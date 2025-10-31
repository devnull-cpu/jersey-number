

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
import timm
from PIL import Image, ImageOps
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, labels_file, processor, use_preprocessing=True, augment=True):
        """
        labels_file: visibility_labels.json with image paths and labels
        processor: DINOv3 image processor
        use_preprocessing: whether to apply contrast/equalize/upscale
        """
        # Load labels
        with open(labels_file, 'r') as f:
            self.labels = json.load(f)
        
        self.image_paths = list(self.labels.keys())
        self.processor = processor
        self.use_preprocessing = use_preprocessing
        
        self.augment = augment
        self.aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomRotation(5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ]) if augment else None
        
        print(f"Found {len(self.image_paths)} labeled images")
        print(f"Preprocessing enabled: {use_preprocessing}")
        
        # Count classes
        label_counts = {}
        for label in self.labels.values():
            label_counts[label] = label_counts.get(label, 0) + 1
        print(f"Label distribution: {label_counts}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def preprocess_image(self, img):
        """Apply contrast enhancement and upscaling"""
        # Auto-contrast (stretches histogram)
        img = ImageOps.autocontrast(img, cutoff=0)
        
        # Histogram equalization (can be aggressive, optional)
        # img = ImageOps.equalize(img)
        
        # Upscale 2x with high-quality resampling
        scale_factor = 2
        # img = img.resize(
        #     (img.width * scale_factor, img.height * scale_factor), 
        #     Image.Resampling.LANCZOS
        # )
        
        return img
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[img_path]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            
            
            # Apply preprocessing if enabled
            if self.use_preprocessing:
                image = self.preprocess_image(image)
                
            # Apply augmentation BEFORE processor (only for training)
            if self.augment and self.aug_transform:
                image = self.aug_transform(image)
            
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            pixel_values = torch.zeros(3, 224, 224)
        
        return pixel_values, label

class BinaryClassifier(nn.Module):
    def __init__(self, dinov3_model, embed_dim=384, num_heads=6, dropout=0.1, freeze_backbone=True):
        super().__init__()
        
        self.backbone = dinov3_model
        
        # Freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Learnable query token for attention pooling
        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Multi-head attention for pooling
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)  # Binary: not visible (0) or visible (1)
        )
    
    def forward(self, pixel_values):
        """
        pixel_values: [batch, 3, H, W] - raw images
        """
        # Extract features from DINOv3
        with torch.no_grad():  # Backbone is frozen
            outputs = self.backbone(pixel_values)
            features = outputs.last_hidden_state  # [batch, 201, 384]
        
        batch_size = features.shape[0]
        
        # Expand query token for batch
        query = self.query_token.expand(batch_size, -1, -1)  # [batch, 1, 384]
        
        # Attention pooling: query attends to all tokens
        pooled, _ = self.attention_pool(query, features, features)  # [batch, 1, 384]
        pooled = pooled.squeeze(1)  # [batch, 384]
        
        # Classify
        logits = self.classifier(pooled)  # [batch, 2]
        
        return logits


class BinaryClassifier2(nn.Module):
    def __init__(self, backbone, embed_dim=384, num_heads=6, dropout=0.1, freeze_backbone=True):
        super().__init__()
        
        self.backbone = backbone
        self.use_timm = hasattr(backbone, 'forward_features')  # Check if it's a timm model
        
        # Freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Learnable query token for attention pooling
        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Multi-head attention for pooling
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Classification head
        """
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )
        """
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),  # Add dropout here
            nn.Linear(embed_dim, 128),  # Skip the 256 layer
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )
    
    def forward(self, pixel_values):
        """
        pixel_values: [batch, 3, H, W] - raw images
        """
        # Extract features
        with torch.no_grad():
            if self.use_timm:
                # For timm models (DeiT, etc.)
                features = self.backbone.forward_features(pixel_values)  # [batch, 197, 384]
            else:
                # For HuggingFace models (DINOv3)
                outputs = self.backbone(pixel_values)
                features = outputs.last_hidden_state  # [batch, 201, 384]
        
        batch_size = features.shape[0]
        
        # Expand query token for batch
        query = self.query_token.expand(batch_size, -1, -1)
        
        # Attention pooling
        pooled, _ = self.attention_pool(query, features, features)
        pooled = pooled.squeeze(1)
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits
        
class SoccerNetProcessor:
    """Preprocessing for DeiT/SoccerNet models"""
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __call__(self, images, return_tensors=None):
        """
        Make it compatible with HuggingFace processor API
        images: PIL Image or list of PIL Images
        """
        if not isinstance(images, list):
            images = [images]
        
        pixel_values = torch.stack([self.transform(img) for img in images])
        
        return {'pixel_values': pixel_values}

def train_model(labels_file, num_epochs=50, batch_size=16, lr=1e-4, 
                val_split=0.2, device='cuda', dino=False):
    
    if dino:
        # Load DINOv3 model
        print("Loading DINOv3 model...")
        model_name = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
        processor = AutoImageProcessor.from_pretrained(model_name)
        dinov3_model = AutoModel.from_pretrained(model_name).to(device)
        dinov3_model.eval()
        print("DINOv3 loaded")
    else:
        deit_backbone = load_soccernet_deit('model.deit_s.pth.tar-16')
        #vit_b_backbone = load_soccernet_vitb('model.vit_b.pth.tar-10')

        deit_backbone = deit_backbone.to(device)
        #vit_b_backbone = vit_b_backbone.to(device)
        processor = SoccerNetProcessor()
        
    # Load labels first
    with open(labels_file, 'r') as f:
        all_labels = json.load(f)
    
    # Split the labels dict into train/val
    image_paths = list(all_labels.keys())
    val_size = int(len(image_paths) * val_split)
    train_size = len(image_paths) - val_size
    
    indices = torch.randperm(len(image_paths)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_labels = {image_paths[i]: all_labels[image_paths[i]] for i in train_indices}
    val_labels = {image_paths[i]: all_labels[image_paths[i]] for i in val_indices}
    
    # Save temporary label files
    with open('train_labels_temp.json', 'w') as f:
        json.dump(train_labels, f)
    with open('val_labels_temp.json', 'w') as f:
        json.dump(val_labels, f)
    
    # Create separate datasets with different augmentation settings
    train_dataset = ImageDataset('train_labels_temp.json', processor, 
                                  use_preprocessing=True, augment=True)
    val_dataset = ImageDataset('val_labels_temp.json', processor, 
                                use_preprocessing=True, augment=False)
    
    print(f"\nTrain size: {train_size}, Val size: {val_size}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # Calculate class weights from training data only
    all_labels = list(train_labels.values())
    class_counts = np.bincount(all_labels)
    class_weights = torch.FloatTensor([1.0 / c for c in class_counts])
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = class_weights.to(device)

    print(f"Class weights: {class_weights.cpu().numpy()}")
    print(f"Train size: {train_size}, Val size: {val_size}")
    
    if dino:
        # Initialize model
        model = BinaryClassifier(dinov3_model, embed_dim=384, num_heads=6, dropout=0.1)
    else:
        model = BinaryClassifier2(deit_backbone, embed_dim=384, num_heads=6, dropout=0.1)
        # model = BinaryClassifier2(vit_b_backbone, embed_dim=768, num_heads=12, dropout=0.12)
        
    model = model.to(device)
    
    # Loss and optimizer (only train the head, not backbone)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_acc = 0.0
    
    # Training loop
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for pixel_values, labels in pbar:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            logits = model(pixel_values)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            _, predicted = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * train_correct / train_total:.2f}%'
            })
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_pred_by_class = {0: 0, 1: 0}  # Count predictions per class
        val_correct_by_class = {0: 0, 1: 0}
        
        with torch.no_grad():
            for pixel_values, labels in val_loader:
                pixel_values = pixel_values.to(device)
                labels = labels.to(device)
                
                logits = model(pixel_values)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Per-class accuracy
                for label in [0, 1]:
                    mask = labels == label
                    val_pred_by_class[label] += mask.sum().item()
                    val_correct_by_class[label] += ((predicted == labels) & mask).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Per-class validation accuracy
        for label in [0, 1]:
            if val_pred_by_class[label] > 0:
                class_acc = 100 * val_correct_by_class[label] / val_pred_by_class[label]
                print(f"  Val Acc (class {label}): {class_acc:.2f}%")
        
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_binary_classifier_sr.pt')
            print(f"  ✓ Saved best model (val_acc: {val_acc:.2f}%)")
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"{'='*60}")
    
    return model
    
    
from torchvision.models import vit_b_16
import torch.nn as nn

class TorchvisionViTWrapper(nn.Module):
    """Wrapper to extract patch tokens from torchvision ViT"""
    def __init__(self, vit_model):
        super().__init__()
        self.vit = vit_model
        
    def forward_features(self, x):
        # Replicate torchvision ViT forward but return hidden states
        x = self.vit._process_input(x)
        n = x.shape[0]
        
        # Expand class token and add position embeddings
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.vit.encoder(x)
        
        # Return all tokens [batch, 197, 768]
        return x
    
    def forward(self, x):
        return self.forward_features(x)

def load_soccernet_vitb(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['state_dict']
    
    # Strip prefix
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.model.'):
            new_key = key.replace('module.model.', '')
            new_state_dict[new_key] = value
    
    # Create torchvision ViT-B/16
    model = vit_b_16(weights=None)
    
    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded ViT-B: {len(missing_keys)} missing, {len(unexpected_keys)} unexpected")
    
    # Wrap to extract features
    wrapper = TorchvisionViTWrapper(model)
    wrapper.eval()
    
    return wrapper


def load_soccernet_deit(checkpoint_path):
    """Load SoccerNet DeiT-S model and prepare for feature extraction"""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['state_dict']
    
    # Strip 'module.model.' prefix
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.model.'):
            new_key = key.replace('module.model.', '')
            new_state_dict[new_key] = value
    
    # Create DeiT-S model
    model = timm.create_model('deit_small_patch16_224', pretrained=False)
    
    # Load weights (ignore head since we'll extract features)
    model.load_state_dict(new_state_dict, strict=False)
    
    # Remove classification head
    model.head = nn.Identity()
    
    model.eval()
    return model

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = train_model(
        labels_file='visibility_labels.json',
        num_epochs=50,
        batch_size=16,  # Smaller batch since we're processing images on the fly
        lr=3e-5,
        val_split=0.2,
        device=device
    )