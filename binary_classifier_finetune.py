"""
Fine-tune a pre-trained binary classifier on a new dataset (offline mode).

This script loads a checkpoint from binary_classifier_train.py and fine-tunes
only the classification head on a new dataset, without requiring network access.

Requirements:
- Trained checkpoint (.pt file)
- Local DINOv3 config files (config.json, preprocessor_config.json)
- New dataset labels file
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, AutoConfig
from PIL import Image, ImageOps
import argparse


class ImageDataset(Dataset):
    def __init__(self, labels_file, processor, use_preprocessing=True):
        """
        labels_file: JSON file with image paths and labels
        processor: DINOv3 image processor
        use_preprocessing: whether to apply contrast/equalize/upscale
        """
        # Load labels
        with open(labels_file, 'r') as f:
            self.labels = json.load(f)

        self.image_paths = list(self.labels.keys())
        self.processor = processor
        self.use_preprocessing = use_preprocessing

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

        # Upscale 2x with high-quality resampling
        scale_factor = 2
        img = img.resize(
            (img.width * scale_factor, img.height * scale_factor),
            Image.Resampling.LANCZOS
        )

        return img

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[img_path]

        try:
            image = Image.open(img_path).convert('RGB')

            # Apply preprocessing if enabled
            if self.use_preprocessing:
                image = self.preprocess_image(image)

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


def load_model_offline(checkpoint_path, config_dir, device='cuda'):
    """
    Load a trained model from checkpoint using local config files only.

    Args:
        checkpoint_path: Path to the .pt checkpoint file
        config_dir: Directory containing config.json and preprocessor_config.json
        device: Device to load model on

    Returns:
        model: Loaded BinaryClassifier model
        processor: Image processor
        checkpoint: Full checkpoint dict (for optimizer state, etc.)
    """
    print(f"Loading model offline from {checkpoint_path}")
    print(f"Using config from {config_dir}")

    # Load config and processor from local files only
    config = AutoConfig.from_pretrained(config_dir, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(config_dir, local_files_only=True)

    # Create model architecture from config (no pretrained weights)
    print("Creating DINOv3 architecture from config...")
    dinov3_model = AutoModel.from_config(config)

    # Create classifier (backbone will be frozen)
    print("Creating BinaryClassifier...")
    model = BinaryClassifier(
        dinov3_model,
        embed_dim=384,
        num_heads=6,
        dropout=0.1,
        freeze_backbone=True
    )

    # Load checkpoint
    print("Loading checkpoint weights...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)

    print(f"Model loaded successfully!")
    print(f"Checkpoint info: Epoch {checkpoint.get('epoch', 'N/A')}, Val Acc: {checkpoint.get('val_acc', 'N/A'):.2f}%")

    # Verify only head is trainable
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    return model, processor, checkpoint


def finetune_model(model, processor, labels_file, num_epochs=20, batch_size=16,
                   lr=1e-4, val_split=0.2, device='cuda', output_name='finetuned_binary_classifier.pt'):
    """
    Fine-tune the classifier head on a new dataset.
    """

    # Load new dataset
    print(f"\nLoading dataset from {labels_file}")
    dataset = ImageDataset(labels_file, processor, use_preprocessing=True)

    # Split train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Train size: {train_size}, Val size: {val_size}")

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

    # Calculate class weights for imbalanced data
    all_labels = list(dataset.labels.values())
    class_counts = np.bincount(all_labels)
    class_weights = torch.FloatTensor([1.0 / c for c in class_counts])
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = class_weights.to(device)

    print(f"Class weights: {class_weights.cpu().numpy()}")

    # Loss and optimizer (only train the head, not backbone)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_acc = 0.0

    print(f"\n{'='*60}")
    print(f"Starting fine-tuning for {num_epochs} epochs")
    print(f"{'='*60}\n")

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
        val_pred_by_class = {0: 0, 1: 0}
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
            }, output_name)
            print(f"  Saved best model to {output_name} (val_acc: {val_acc:.2f}%)")

    print(f"\n{'='*60}")
    print(f"Fine-tuning complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {output_name}")
    print(f"{'='*60}")

    return model


def main():
    parser = argparse.ArgumentParser(description='Fine-tune binary classifier on new dataset (offline)')

    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint .pt file')
    parser.add_argument('--config_dir', type=str, required=True,
                        help='Directory containing DINOv3 config files')
    parser.add_argument('--labels', type=str, required=True,
                        help='Path to new labels JSON file')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of fine-tuning epochs (default: 20)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')
    parser.add_argument('--output', type=str, default='finetuned_binary_classifier.pt',
                        help='Output checkpoint filename (default: finetuned_binary_classifier.pt)')

    # Device
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu, default: auto-detect)')

    args = parser.parse_args()

    # Auto-detect device if not specified
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")
    print(f"\nConfiguration:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Config dir: {args.config_dir}")
    print(f"  Labels: {args.labels}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Val split: {args.val_split}")
    print(f"  Output: {args.output}\n")

    # Load model offline
    model, processor, checkpoint = load_model_offline(
        args.checkpoint,
        args.config_dir,
        device=device
    )

    # Fine-tune on new dataset
    model = finetune_model(
        model=model,
        processor=processor,
        labels_file=args.labels,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        device=device,
        output_name=args.output
    )


if __name__ == '__main__':
    main()
