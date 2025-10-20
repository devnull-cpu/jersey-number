import json
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
import numpy as np
import re
from transformers import pipeline, AutoProcessor, VitPoseForPoseEstimation
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image, ImageOps, ImageEnhance



class BinaryClassifier(nn.Module):
    def __init__(self, dinov3_model, embed_dim=384, num_heads=6, dropout=0.1):
        super().__init__()
        
        self.backbone = dinov3_model
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
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
    
    def forward(self, pixel_values):
        with torch.no_grad():
            outputs = self.backbone(pixel_values)
            features = outputs.last_hidden_state
        
        batch_size = features.shape[0]
        query = self.query_token.expand(batch_size, -1, -1)
        pooled, _ = self.attention_pool(query, features, features)
        pooled = pooled.squeeze(1)
        logits = self.classifier(pooled)
        
        return logits

def preprocess_image(img):
    """Apply same preprocessing as training"""
    img = ImageOps.autocontrast(img, cutoff=0)
    
    scale_factor = 2
    img = img.resize(
        (img.width * scale_factor, img.height * scale_factor), 
        Image.Resampling.LANCZOS
    )
    
    return img

# Import PARSeq dependencies
try:
    from strhub.data.module import SceneTextDataModule
    PARSEQ_AVAILABLE = True
    print("✓ Successfully imported strhub.data.module")
except ImportError as e:
    print(f"✗ Failed to import strhub.data.module: {e}")
    PARSEQ_AVAILABLE = False

def load_validation_data(json_file_path):
    """Load the validation set JSON file with ground truth labels"""
    with open(json_file_path, 'r') as f:
        validation_data = json.load(f)
    return validation_data

def extract_torso_crop(image, padding=5, device=None):
    """Extract torso region from a cropped person image using shoulder and hip keypoints."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Initialize pose estimation model (we'll cache this later)
        if not hasattr(extract_torso_crop, 'processor'):
            extract_torso_crop.processor = AutoProcessor.from_pretrained("usyd-community/vitpose-plus-small")
            extract_torso_crop.model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-plus-small", device_map=device)
        
        image_processor = extract_torso_crop.processor
        model = extract_torso_crop.model
        
        # Create a bounding box for the entire image in COCO format [x, y, w, h]
        image_width, image_height = image.size
        full_image_boxes = np.array([[0, 0, image_width, image_height]], dtype=np.float32)
        
        # Process the image with the full image bounding box
        inputs = image_processor(image, boxes=[full_image_boxes], return_tensors="pt").to(device)
        
        with torch.no_grad():
            # VitPose models with multiple experts require dataset_index as a tensor
            dataset_index = torch.tensor([0], device=device)
            outputs = model(**inputs, dataset_index=dataset_index)
        
        # Post-process pose estimation
        pose_results = image_processor.post_process_pose_estimation(outputs, boxes=[full_image_boxes], threshold=0.3)
        
        if not pose_results or not pose_results[0]:
            return image
        
        person_pose = pose_results[0][0]  # First person in cropped image
        
        # Extract shoulder and hip keypoints
        keypoints = {}
        
        for keypoint, label, score in zip(
            person_pose["keypoints"], 
            person_pose["labels"], 
            person_pose["scores"]
        ):
            keypoint_name = model.config.id2label[label.item()]
            x = keypoint[0].item()
            y = keypoint[1].item()
            confidence = score.item()
            
            # Store relevant keypoints with confidence check
            if confidence > 0.3:
                if keypoint_name == 'L_Shoulder':
                    keypoints['left_shoulder'] = (x, y)
                elif keypoint_name == 'R_Shoulder':
                    keypoints['right_shoulder'] = (x, y)
                elif keypoint_name == 'L_Hip':
                    keypoints['left_hip'] = (x, y)
                elif keypoint_name == 'R_Hip':
                    keypoints['right_hip'] = (x, y)
        
        # Check if we have the required keypoints
        required_points = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        missing_points = [point for point in required_points if point not in keypoints]
        
        if missing_points:
            return image
        
        # Calculate torso bounding box using shoulders and hips
        shoulder_points = [keypoints['left_shoulder'], keypoints['right_shoulder']]
        hip_points = [keypoints['left_hip'], keypoints['right_hip']]
        all_points = shoulder_points + hip_points
        
        # Get bounding box coordinates
        x_coords = [point[0] for point in all_points]
        y_coords = [point[1] for point in all_points]
        
        min_x = int(max(0, min(x_coords) - padding))
        max_x = int(min(image.width, max(x_coords) + padding))
        min_y = int(max(0, min(y_coords) - padding))
        max_y = int(min(image.height, max(y_coords) + padding))
        
        # Crop the torso region
        torso_crop = image.crop((min_x, min_y, max_x, max_y))
        
        return torso_crop
        
    except Exception as e:
        return image

def preprocess_image_siglip(image_path, use_torso_extraction=False, torso_padding=5):
    """Preprocess image for SigLIP model"""
    img = Image.open(image_path)
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Apply auto-contrast
    img = ImageOps.autocontrast(img, cutoff=0)
    
    img_gray = ImageOps.grayscale(img)
    img = img_gray.convert('RGB')
    
    return img

def preprocess_image_parseq(image_path, use_torso_extraction=True, torso_padding=1):
    """Preprocess image for PARSeq model"""
    img = Image.open(image_path)
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    # Apply auto-contrast
    img = ImageOps.autocontrast(img, cutoff=0)
    
    
    # Extract torso region if requested
    if use_torso_extraction:
        img = extract_torso_crop(img, padding=torso_padding)
        
    img_gray = ImageOps.grayscale(img)
    img = img_gray.convert('RGB')
    
    return img

def beam_search_decode(logits, tokenizer, beam_size=5):
    """Beam search decoding with proper confidence calculation"""
    log_probs = torch.log_softmax(logits, dim=-1)[0]  # [T, C]
    eos_id, bos_id, pad_id = tokenizer.eos_id, tokenizer.bos_id, tokenizer.pad_id
    itos = tokenizer._itos  # index-to-string map

    sequences = [([bos_id], 0.0)]  # start with <BOS>
    for t in range(log_probs.size(0)):
        all_candidates = []
        for seq, score in sequences:
            if seq[-1] == eos_id:  # stop expanding if already ended
                all_candidates.append((seq, score))
                continue
            for c in range(log_probs.size(1)):
                new_seq = seq + [c]
                new_score = score + log_probs[t, c].item()
                all_candidates.append((new_seq, new_score))
        # keep top-k
        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_size]
        # if all finished early
        if all(s[-1] == eos_id for s, _ in sequences):
            break

    decoded = []
    for seq, score in sequences:
        # strip BOS/EOS/PAD tokens
        chars = [itos[i] for i in seq if i not in (bos_id, eos_id, pad_id)]
        text = ''.join(chars)
        # Convert log probability to probability-like confidence
        # Normalize by sequence length to get average per-token confidence
        seq_len = len([i for i in seq if i not in (bos_id, eos_id, pad_id)])
        avg_log_prob = score / max(seq_len, 1)
        # Convert to probability (still will be small, but relative ordering is maintained)
        confidence = np.exp(avg_log_prob)
        decoded.append((text, confidence))
    return decoded

def extract_jersey_number_from_parseq(results, valid_numbers):
    """Extract jersey number from PARSeq beam search results with confidence"""
    for text, confidence in results:
        # Look for 1-2 digit numbers in the text
        m = re.search(r'\b\d{1,2}\b', text)
        if m:
            num_str = m.group()
            # Pad single digits with leading zero
            if len(num_str) == 1:
                num_str = '0' + num_str
            # Check if it's a valid jersey number
            if num_str in valid_numbers:
                return num_str, confidence
    
    return None, 0.0

def get_siglip_prediction(image_path, image_classifier, candidate_labels, 
                          score_threshold=0.0002, use_torso_extraction=False):
    """Get SigLIP prediction for a single image"""
    img = preprocess_image_siglip(image_path, use_torso_extraction)
    output = image_classifier(img, candidate_labels)
    
    # Handle output - it's a list of dicts when single image is passed
    if isinstance(output, list) and len(output) > 0:
        if isinstance(output[0], dict):
            # Single image returns [{'label': ..., 'score': ...}, ...]
            predictions = output
        elif isinstance(output[0], list):
            # Batch returns [[{'label': ..., 'score': ...}, ...]]
            predictions = output[0]
        else:
            return None, 0.0
    else:
        return None, 0.0
    
    # Get valid predictions above threshold
    valid_predictions = [item for item in predictions if item['score'] >= score_threshold]
    
    if valid_predictions:
        top_prediction = max(valid_predictions, key=lambda x: x['score'])
        return top_prediction['label'], top_prediction['score']
    else:
        return None, 0.0

def get_parseq_prediction(image_path, parseq_model, img_transform, device, valid_numbers, 
                          use_torso_extraction=True):
    """Get PARSeq prediction for a single image"""
    img = preprocess_image_parseq(image_path, use_torso_extraction)
    img_tensor = img_transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = parseq_model(img_tensor)
        results = beam_search_decode(logits, parseq_model.tokenizer, beam_size=5)
    
    return extract_jersey_number_from_parseq(results, valid_numbers)


        
        
def is_double_digit(num_str):
    """Check if a string is a two-digit number with identical digits."""
    return len(num_str) == 2 and num_str[0] == num_str[1] and num_str.isdigit()

def optimal_fusion_threshold(siglip_pred, siglip_conf, parseq_pred, parseq_conf):
    """
    Fusion strategy: Trust PARSeq when confident (>= 0.40),
    otherwise default to SigLIP.
    Accuracy: 0.7149
    """
    if parseq_pred is not None and parseq_conf >= 0.40:
        return parseq_pred, 'PARSeq', parseq_conf
    else:
        pred = siglip_pred if siglip_pred is not None else "NO_PREDICTION"
        return pred, 'SigLIP', siglip_conf


def optimal_fusion_weighted(siglip_pred, siglip_conf, parseq_pred, parseq_conf):
    """
    Fusion strategy: Weighted voting with SigLIP=0.9, PARSeq=0.1
    """
    # Handle None cases
    if siglip_pred is None and parseq_pred is None:
        return "NO_PREDICTION", 'None', 0.0
    elif siglip_pred is None:
        return parseq_pred, 'PARSeq', parseq_conf
    elif parseq_pred is None:
        return siglip_pred, 'SigLIP', siglip_conf
    
    siglip_weight = 0.9
    parseq_weight = 0.1

    if is_double_digit(parseq_pred):
        siglip_weight = 0.3
        parseq_weight = 0.7

    # Always use weighted scoring, even when they agree
    # This ensures consistent attribution
    siglip_score = siglip_conf * siglip_weight
    parseq_score = parseq_conf * parseq_weight
    
    if siglip_score > parseq_score:
        return siglip_pred, 'SigLIP', siglip_conf
    else:
        return parseq_pred, 'PARSeq', parseq_conf


# Your original (now suboptimal) for comparison
def optimal_fusion_original(siglip_pred, siglip_conf, parseq_pred, parseq_conf):
    """
    Original fusion strategy: Trust PARSeq when very confident (>= 0.88),
    otherwise default to SigLIP.
    Accuracy: 0.6983
    """
    if parseq_pred is not None and parseq_conf >= 0.88:
        return parseq_pred, 'PARSeq', parseq_conf
    else:
        pred = siglip_pred if siglip_pred is not None else "NO_PREDICTION"
        return pred, 'SigLIP', siglip_conf


def optimal_fusion_with_rejection(siglip_pred, siglip_conf, parseq_pred, parseq_conf,
                                  siglip_threshold=0.03,
                                  parseq_threshold=0.65,
                                  agreement_bonus_siglip=0.1,
                                  agreement_bonus_parseq=0.2):
    """
    Fusion with model-specific rejection thresholds
    
    Args:
        siglip_threshold: Minimum confidence for SigLIP (default 0.03)
        parseq_threshold: Minimum confidence for PARSeq (default 0.65)
        agreement_bonus_siglip: Bonus to SigLIP conf when models agree
        agreement_bonus_parseq: Bonus to PARSeq conf when models agree
    
    Returns:
        (prediction, source, confidence) - prediction is "NO_PREDICTION" if not confident
    """

    if is_double_digit(siglip_pred):
        siglip_conf = siglip_conf*0.35

    if siglip_pred is None and parseq_pred is None:
        return "NO_PREDICTION", 'None', 0.0
    elif siglip_pred is None:
        if parseq_conf >= parseq_threshold:
            return parseq_pred, 'PARSeq', parseq_conf
        else:
            return "NO_PREDICTION", 'Rejected_PARSeq', parseq_conf
    elif parseq_pred is None:
        if siglip_conf >= siglip_threshold:
            return siglip_pred, 'SigLIP', siglip_conf
        else:
            return "NO_PREDICTION", 'Rejected_SigLIP', siglip_conf
    
    # Check for agreement - highly reliable (95.9% accurate)
    if siglip_pred == parseq_pred:
        # Boost both confidences when they agree
        effective_siglip_conf = min(1.0, siglip_conf + agreement_bonus_siglip)
        effective_parseq_conf = min(1.0, parseq_conf + agreement_bonus_parseq)
        # Check if either passes threshold with bonus
        if effective_siglip_conf >= siglip_threshold or effective_parseq_conf >= parseq_threshold:
            
            return siglip_pred, 'Agreement', max(siglip_conf, parseq_conf)
        else:
            return "NO_PREDICTION", 'Rejected_Agreement', max(siglip_conf, parseq_conf)
    
    # Disagreement - use weighted scoring
    siglip_score = siglip_conf * 0.9
    parseq_score = parseq_conf * 0.1
    
    if siglip_score > parseq_score:
        # SigLIP wins
        if siglip_conf >= siglip_threshold:
            return siglip_pred, 'SigLIP', siglip_conf
        else:
            return "NO_PREDICTION", 'Rejected_SigLIP', siglip_conf
    else:
        # PARSeq wins
        if parseq_conf >= parseq_threshold:
            return parseq_pred, 'PARSeq', parseq_conf
        else:
            return "NO_PREDICTION", 'Rejected_PARSeq', parseq_conf


def is_visible(image_path, model, processor, device, confidence_threshold, use_preprocessing=True):
    """Check if an image is visible using the binary classifier."""
    try:
        image = Image.open(image_path).convert('RGB')
        if use_preprocessing:
            image = preprocess_image(image)
        
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)
        
        with torch.no_grad():
            logits = model(pixel_values)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred].item()
        
        return pred == 1 and confidence >= confidence_threshold
    except Exception as e:
        print(f"\nError in visibility check for {image_path}: {e}")
        return False

def evaluate_fusion_model(validation_data, 
                         fusion_strategy='weighted',  # 'weighted' or 'rejection'
                         siglip_threshold=0.03,
                         parseq_threshold=0.65,
                         use_torso_extraction_siglip=False,
                         use_torso_extraction_parseq=True,
                         batch_size=8,
                         visibility_model_path='best_binary_classifier.pt',
                         visibility_confidence_threshold=0.67):
    """Evaluate fusion model that runs both models and uses specified fusion strategy"""
    print("=== Evaluating Fusion Model ===")
    print(f"Fusion strategy: {fusion_strategy}")
    if fusion_strategy == 'rejection':
        print(f"SigLIP threshold: {siglip_threshold}")
        print(f"PARSeq threshold: {parseq_threshold}")
    print("Running BOTH models on all images...")
    
    # Load models
    print("\nLoading models...")

    # Load visibility classifier
    print(f"Loading visibility classifier from {visibility_model_path}...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
    visibility_processor = AutoImageProcessor.from_pretrained(model_name)
    dinov3_model = AutoModel.from_pretrained(model_name).to(device)
    dinov3_model.eval()
    
    visibility_model = BinaryClassifier(dinov3_model, embed_dim=384, num_heads=6, dropout=0.1)
    checkpoint = torch.load(visibility_model_path, map_location=device)
    visibility_model.load_state_dict(checkpoint['model_state_dict'])
    visibility_model = visibility_model.to(device)
    visibility_model.eval()
    print(f"Visibility model loaded (val acc: {checkpoint['val_acc']:.2f}%)")

    print("Loading SigLIP model...")
    ckpt = "google/siglip2-so400m-patch16-naflex"

    image_classifier = pipeline(model=ckpt, task="zero-shot-image-classification", device=0)
    candidate_labels = [str(i) for i in [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 36, 44, 50, 55, 62, 93]]
    valid_numbers = ['0' + str(i) if i < 10 else str(i) for i in [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 36, 44, 50, 55, 62, 93]]
    
    parseq_model = None
    img_transform = None
    if PARSEQ_AVAILABLE:
        print("Loading PARSeq model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        parseq_model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval().to(device)
        img_transform = SceneTextDataModule.get_transform(parseq_model.hparams.img_size)
    else:
        print("Warning: PARSeq not available, fusion will only use SigLIP")
        device = None
    
    # Load pose estimation if needed
    if use_torso_extraction_siglip or use_torso_extraction_parseq:
        print("Loading pose estimation model...")
        pose_device = "cuda" if torch.cuda.is_available() else "cpu"
        extract_torso_crop.processor = AutoProcessor.from_pretrained("usyd-community/vitpose-plus-small")
        extract_torso_crop.model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-plus-small", device_map=pose_device)
    
    # Process each image
    prediction_details = []
    model_usage = defaultdict(int)
    
    print(f"Processing {len(validation_data)} images...")
    
    for image_path, true_label in validation_data.items():
        try:
            if not Path(image_path).exists():
                continue

            # Pre-filter for visibility
            if not is_visible(str(image_path), visibility_model, visibility_processor, device, visibility_confidence_threshold):
                prediction_details.append({
                    'image_path': image_path,
                    'true_label': true_label,
                    'predicted_label': 'NOT_VISIBLE',
                    'confidence': 0,
                    'correct': False,
                    'model_used': 'VisibilityClassifier',
                })
                continue
            
            # ALWAYS run both models
            siglip_pred, siglip_conf = get_siglip_prediction(
                image_path, image_classifier, candidate_labels,
                use_torso_extraction=use_torso_extraction_siglip
            )
            
            parseq_pred = None
            parseq_conf = 0.0
            if PARSEQ_AVAILABLE and parseq_model is not None:
                parseq_pred, parseq_conf = get_parseq_prediction(
                    image_path, parseq_model, img_transform, device, valid_numbers,
                    use_torso_extraction=use_torso_extraction_parseq
                )
            
            # Apply fusion strategy
            if fusion_strategy == 'rejection':
                final_prediction, prediction_source, final_confidence = optimal_fusion_with_rejection(
                    siglip_pred, siglip_conf, parseq_pred, parseq_conf,
                    siglip_threshold=siglip_threshold,
                    parseq_threshold=parseq_threshold,
                    agreement_bonus_siglip=0.1,
                    agreement_bonus_parseq=0.2
                )
            else:  # weighted
                final_prediction, prediction_source, final_confidence = optimal_fusion_weighted(
                    siglip_pred, siglip_conf, parseq_pred, parseq_conf
                )
            
            # Count model usage
            model_usage[prediction_source] += 1
            
            # Normalize labels
            true_label_str = str(true_label)
            if final_prediction and final_prediction != "NO_PREDICTION" and final_prediction.startswith('0') and len(final_prediction) == 2:
                final_prediction = final_prediction[1:]
            if siglip_pred and siglip_pred.startswith('0') and len(siglip_pred) == 2:
                siglip_pred = siglip_pred[1:]
            if parseq_pred and parseq_pred.startswith('0') and len(parseq_pred) == 2:
                parseq_pred = parseq_pred[1:]
            
            is_correct = final_prediction == true_label_str
            
            prediction_details.append({
                'image_path': image_path,
                'true_label': true_label_str,
                'predicted_label': final_prediction,
                'confidence': final_confidence,
                'correct': is_correct,
                'model_used': prediction_source,
                'siglip_prediction': siglip_pred,
                'siglip_confidence': siglip_conf,
                'parseq_prediction': parseq_pred,
                'parseq_confidence': parseq_conf
            })
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    # Calculate metrics
    total_predictions = len(prediction_details)
    predictions_made = sum(1 for d in prediction_details if d['predicted_label'] != "NO_PREDICTION")
    no_predictions = total_predictions - predictions_made
    correct_predictions = sum(1 for d in prediction_details if d['correct'] and d['predicted_label'] != "NO_PREDICTION")
    
    overall_accuracy = sum(1 for d in prediction_details if d['correct']) / total_predictions if total_predictions else 0
    accuracy_when_predicted = correct_predictions / predictions_made if predictions_made else 0
    
    accuracy_on_predictions = correct_predictions / predictions_made if predictions_made else 0
    coverage = predictions_made / total_predictions
    
    print(f"\n{'='*60}")
    print(f"=== FUSION MODEL RESULTS ===")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy_on_predictions:.4f} ({accuracy_on_predictions*100:.2f}%)")
    print(f"Coverage: {coverage:.4f} ({coverage*100:.2f}%)")
    print(f"Correct predictions: {correct_predictions}/{predictions_made} (out of {total_predictions} total)")
    print(f"No prediction: {no_predictions} ({no_predictions/total_predictions*100:.1f}%)")
    
    print(f"\nModel usage breakdown:")
    for source, count in sorted(model_usage.items(), key=lambda x: x[1], reverse=True):
        print(f"  {source}: {count} ({count/total_predictions*100:.1f}%)")
    
    # Analyze accuracy by model choice (excluding rejections)
    for source in model_usage.keys():
        if 'Rejected' in source or source == 'None':
            continue
        predictions_by_source = [d for d in prediction_details if d['model_used'] == source]
        if predictions_by_source:
            accuracy_by_source = sum(1 for d in predictions_by_source if d['correct']) / len(predictions_by_source)
            print(f"  {source} accuracy: {accuracy_by_source:.4f}")
    
    # Show standalone accuracies for comparison
    siglip_standalone_correct = sum(1 for d in prediction_details if d['siglip_prediction'] == d['true_label'])
    siglip_standalone_accuracy = siglip_standalone_correct / total_predictions
    
    if PARSEQ_AVAILABLE:
        parseq_standalone_correct = sum(1 for d in prediction_details if d['parseq_prediction'] == d['true_label'])
        parseq_standalone_accuracy = parseq_standalone_correct / total_predictions
        
        print(f"\n--- Standalone Model Performance ---")
        print(f"SigLIP standalone: {siglip_standalone_accuracy:.4f} ({siglip_standalone_accuracy*100:.2f}%)")
        print(f"PARSeq standalone: {parseq_standalone_accuracy:.4f} ({parseq_standalone_accuracy*100:.2f}%)")
        print(f"Fusion improvement over SigLIP: {(accuracy_when_predicted - siglip_standalone_accuracy)*100:.2f} pp")
    
    return overall_accuracy, accuracy_when_predicted, prediction_details


def process_folder(folder_path, 
                   image_classifier, 
                   parseq_model, 
                   img_transform, 
                   device, 
                   valid_numbers, 
                   candidate_labels,
                   visibility_model, 
                   visibility_processor, 
                   fusion_strategy='rejection',
                   siglip_threshold=0.03,
                   parseq_threshold=0.65,
                   use_torso_extraction_siglip=False,
                   use_torso_extraction_parseq=True,
                   vote_threshold=0.6,
                   visibility_confidence_threshold=0.67):
    """
    Process all images in a folder and return predictions with voting
    
    Args:
        folder_path: Path to folder containing images
        fusion_strategy: 'weighted' or 'rejection'
        siglip_threshold: Threshold for SigLIP confidence
        parseq_threshold: Threshold for PARSeq confidence
        use_torso_extraction_siglip: Whether to use torso extraction for SigLIP
        use_torso_extraction_parseq: Whether to use torso extraction for PARSeq
        vote_threshold: Minimum proportion of predictions needed for consensus (0.5 = majority)
        visibility_confidence_threshold: Confidence threshold for visibility
    
    Returns:
        dict with predictions per image and consensus prediction
    """
    print(f"=== Processing folder: {folder_path} ===")
    print(f"Fusion strategy: {fusion_strategy}")
    if fusion_strategy == 'rejection':
        print(f"Thresholds: SigLIP>={siglip_threshold}, PARSeq>={parseq_threshold}")

    # Find all images in folder
    folder = Path(folder_path)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions]
    
    print(f"\nFound {len(image_files)} images to process")
    
    # Process each image
    predictions = []
    valid_predictions = []  # Predictions that aren't "NO_PREDICTION"
    not_visible_count = 0
    
    for image_file in image_files:
        try:
            print(f"Processing {image_file.name}...", end=' ')

            # Pre-filter for visibility
            if not is_visible(str(image_file), visibility_model, visibility_processor, device, visibility_confidence_threshold):
                print("✗ Not visible")
                predictions.append({
                    'image_file': str(image_file),
                    'image_name': image_file.name,
                    'predicted_number': 'NOT_VISIBLE',
                    'confidence': 0,
                    'prediction_source': 'VisibilityClassifier',
                })
                not_visible_count += 1
                continue
            
            # Run both models
            siglip_pred, siglip_conf = get_siglip_prediction(
                str(image_file), image_classifier, candidate_labels,
                use_torso_extraction=use_torso_extraction_siglip
            )
            
            parseq_pred = None
            parseq_conf = 0.0
            if PARSEQ_AVAILABLE and parseq_model is not None:
                parseq_pred, parseq_conf = get_parseq_prediction(
                    str(image_file), parseq_model, img_transform, device, valid_numbers,
                    use_torso_extraction=use_torso_extraction_parseq
                )
            
            # Apply fusion strategy
            if fusion_strategy == 'rejection':
                final_prediction, prediction_source, final_confidence = optimal_fusion_with_rejection(
                    siglip_pred, siglip_conf, parseq_pred, parseq_conf,
                    siglip_threshold=siglip_threshold,
                    parseq_threshold=parseq_threshold,
                    agreement_bonus_siglip=0.1,
                    agreement_bonus_parseq=0.2
                )
            else:  # weighted
                final_prediction, prediction_source, final_confidence = optimal_fusion_weighted(
                    siglip_pred, siglip_conf, parseq_pred, parseq_conf
                )
            
            # Normalize predictions
            if final_prediction and final_prediction != "NO_PREDICTION" and final_prediction.startswith('0') and len(final_prediction) == 2:
                final_prediction = final_prediction[1:]
            if siglip_pred and siglip_pred.startswith('0') and len(siglip_pred) == 2:
                siglip_pred = siglip_pred[1:]
            if parseq_pred and parseq_pred.startswith('0') and len(parseq_pred) == 2:
                parseq_pred = parseq_pred[1:]
            
            prediction_data = {
                'image_file': str(image_file),
                'image_name': image_file.name,
                'predicted_number': final_prediction,
                'confidence': final_confidence,
                'prediction_source': prediction_source,
                'siglip_prediction': siglip_pred,
                'siglip_confidence': siglip_conf,
                'parseq_prediction': parseq_pred,
                'parseq_confidence': parseq_conf
            }
            
            predictions.append(prediction_data)
            
            if final_prediction != "NO_PREDICTION":
                valid_predictions.append(final_prediction)
                print(f"✓ Predicted: {final_prediction} (confidence: {final_confidence:.3f})")
            else:
                print(f"✗ No prediction (rejected)")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    # Voting across all valid predictions
    print(f"\n{'='*60}")
    print("=== VOTING RESULTS ===")
    print(f"{'='*60}")
    print(f"Total images: {len(predictions)}")
    print(f"Valid predictions: {len(valid_predictions)}")
    print(f"Rejected: {len(predictions) - len(valid_predictions)}")
    
    consensus_number = None
    vote_confidence = 0.0
    vote_details = {}
    
    if valid_predictions:
        # Count votes
        vote_counts = Counter(valid_predictions)
        total_votes = len(valid_predictions)
        
        print(f"\nVote breakdown:")
        for number, count in vote_counts.most_common():
            percentage = count / total_votes
            print(f"  #{number}: {count}/{total_votes} votes ({percentage:.1%})")
        
        # Determine consensus
        most_common_number, most_common_count = vote_counts.most_common(1)[0]
        consensus_percentage = most_common_count / total_votes
        
        if consensus_percentage >= vote_threshold:
            consensus_number = most_common_number
            vote_confidence = consensus_percentage
            print(f"\n✓ CONSENSUS: #{consensus_number}")
            print(f"  Confidence: {vote_confidence:.1%} ({most_common_count}/{total_votes} votes)")
        else:
            consensus_number=-1
            print(f"\n✗ NO CONSENSUS (threshold: {vote_threshold:.1%})")
            print(f"  Best candidate: #{most_common_number} with {consensus_percentage:.1%}")
        
        vote_details = {
            'vote_counts': dict(vote_counts),
            'total_votes': total_votes,
            'most_common': most_common_number,
            'most_common_count': most_common_count,
            'consensus_percentage': consensus_percentage
        }
    else:
        print("\n✗ NO VALID PREDICTIONS - all images rejected")
    
    # Calculate prediction source statistics
    source_counts = Counter(p['prediction_source'] for p in predictions)
    print(f"\n--- Prediction Source Breakdown ---")
    for source, count in source_counts.most_common():
        print(f"  {source}: {count} ({count/len(predictions):.1%})")
    
    return {
        'folder_path': str(folder_path),
        'total_images': len(predictions),
        'valid_predictions_count': len(valid_predictions),
        'rejected_count': len(predictions) - len(valid_predictions),
        'consensus_number': consensus_number,
        'consensus_confidence': vote_confidence,
        'vote_threshold_used': vote_threshold,
        'vote_details': vote_details,
        'source_counts': dict(source_counts),
        'individual_predictions': predictions
    }

def benchmark_visibility_filter(root_image_folder, gt_json_path, image_classifier, parseq_model, img_transform, device, valid_numbers, candidate_labels, visibility_model, visibility_processor, max_folders=None):
    """Benchmark the visibility filter by processing all folders in the ground truth file."""
    with open(gt_json_path, 'r') as f:
        ground_truth = json.load(f)

    all_predictions = []
    correct_predictions = 0
    total_predictions = 0
    covered_predictions = 0
    correct_covered_predictions = 0

    i = 0
    for folder_name, true_label in ground_truth.items():
        if max_folders is not None and i >= max_folders:
            break

        #if true_label == -1:
        #    continue

        folder_path = Path(root_image_folder) / folder_name
        if not folder_path.exists():
            continue

        total_predictions += 1
        i += 1

        result = process_folder(
            folder_path=str(folder_path),
            image_classifier=image_classifier,
            parseq_model=parseq_model,
            img_transform=img_transform,
            device=device,
            valid_numbers=valid_numbers,
            candidate_labels=candidate_labels,
            visibility_model=visibility_model,
            visibility_processor=visibility_processor,
        )

        if result['consensus_number'] is not None and result['consensus_number'] != 'NOT_VISIBLE' and result['consensus_number'] != 'NO_PREDICTION':
            covered_predictions += 1
            if result['consensus_number'] == str(true_label):
                correct_covered_predictions += 1

        if result['consensus_number'] == str(true_label):
            correct_predictions += 1
        
        all_predictions.append(result)

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    coverage = covered_predictions / total_predictions if total_predictions > 0 else 0
    accuracy_on_covered = correct_covered_predictions / covered_predictions if covered_predictions > 0 else 0

    print(f"\n{'='*60}")
    print("=== BENCHMARK RESULTS ===")
    print(f"{'='*60}")
    print(f"Total folders processed: {total_predictions}")
    print(f"Correct predictions (overall): {correct_predictions}")
    print(f"Overall Accuracy: {accuracy:.2%}")
    print(f"\nCoverage: {coverage:.2%} ({covered_predictions}/{total_predictions})")
    print(f"Accuracy on Covered: {accuracy_on_covered:.2%} ({correct_covered_predictions}/{covered_predictions})")

    with open('benchmark_results.json', 'w') as f:
        json.dump({
            'overall_accuracy': accuracy,
            'coverage': coverage,
            'accuracy_on_covered': accuracy_on_covered,
            'results': all_predictions
        }, f, indent=2)

if __name__ == "__main__":
    gt_json_path = "train_gt.json"
    root_image_folder = "C:/Temp/jersey-2023/train/images/"
    
    try:
        # Load the ground truth data to dynamically create labels
        with open(gt_json_path, 'r') as f:
            ground_truth = json.load(f)
        
        all_numbers = {label for label in ground_truth.values() if label != -1}
        candidate_labels = [str(num) for num in sorted(list(all_numbers))]
        valid_numbers = ['0' + str(num) if num < 10 else str(num) for num in sorted(list(all_numbers))]

        # Load all models once
        print("Loading all models...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Visibility model
        visibility_model_path = 'best_binary_classifier.pt'
        model_name = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
        visibility_processor = AutoImageProcessor.from_pretrained(model_name)
        dinov3_model = AutoModel.from_pretrained(model_name).to(device)
        dinov3_model.eval()
        visibility_model = BinaryClassifier(dinov3_model, embed_dim=384, num_heads=6, dropout=0.1)
        checkpoint = torch.load(visibility_model_path, map_location=device)
        visibility_model.load_state_dict(checkpoint['model_state_dict'])
        visibility_model = visibility_model.to(device)
        visibility_model.eval()
        print(f"Visibility model loaded (val acc: {checkpoint['val_acc']:.2f}%)")

        # SigLIP model
        ckpt = "google/siglip2-so400m-patch16-naflex"
        image_classifier = pipeline(model=ckpt, task="zero-shot-image-classification", device=0)

        # PARSeq model
        parseq_model = None
        img_transform = None
        if PARSEQ_AVAILABLE:
            parseq_model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval().to(device)
            img_transform = SceneTextDataModule.get_transform(parseq_model.hparams.img_size)

        # Pose estimation model
        extract_torso_crop.processor = AutoProcessor.from_pretrained("usyd-community/vitpose-plus-small")
        extract_torso_crop.model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-plus-small", device_map=device)



        
        # validation_data = load_validation_data(validation_json_path)
        
        result = process_folder(
            folder_path="C:/Temp/jersey-2023/train/images/115",
            image_classifier=image_classifier,
            parseq_model=parseq_model,
            img_transform=img_transform,
            candidate_labels=candidate_labels,
            visibility_model=visibility_model,
            visibility_processor=visibility_processor,
            fusion_strategy='rejection',
            siglip_threshold=0.03,
            parseq_threshold=0.65,
            use_torso_extraction_siglip=False,
            use_torso_extraction_parseq=True,
            vote_threshold=0.6,  # Require majority vote (>50%)
            device=device,
            valid_numbers=valid_numbers,
        )
        """
        # Mode 2: Maximize prediction accuracy with rejection
        print("\n" + "="*60)
        print("MODE 2: MAXIMIZE PREDICTION ACCURACY (WITH REJECTION)")
        print("="*60)
        overall_acc2, pred_acc2, details2 = evaluate_fusion_model(
            validation_data,
            fusion_strategy='rejection',
            siglip_threshold=0.03,  # Optimized threshold
            parseq_threshold=0.65,   # Optimized threshold
            use_torso_extraction_siglip=False,
            use_torso_extraction_parseq=True
        )

        # Save results
        with open('fusion_results_with_rejection.json', 'w') as f:
            json.dump({
                'fusion_strategy': 'rejection',
                'siglip_threshold': 0.03,
                'parseq_threshold': 0.65,
                'overall_accuracy': overall_acc2,
                'accuracy_when_predicted': pred_acc2,
                'details': details2
            }, f, indent=2)
        """
        """
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        print(f"Mode 1 (No rejection): {overall_acc1:.4f} overall, 100% coverage")
        print(f"Mode 2 (With rejection): {pred_acc2:.4f} accuracy, {sum(1 for d in details2 if d['predicted_label'] != 'NO_PREDICTION')/len(details2):.1%} coverage")
        """
        """
        benchmark_visibility_filter(
            root_image_folder=root_image_folder,
            gt_json_path=gt_json_path,
            image_classifier=image_classifier,
            parseq_model=parseq_model,
            img_transform=img_transform,
            device=device,
            valid_numbers=valid_numbers,
            candidate_labels=candidate_labels,
            visibility_model=visibility_model,
            visibility_processor=visibility_processor,
            max_folders=100
        )
        """

    except FileNotFoundError:
        print(f"Error: Could not find {gt_json_path}")
    except Exception as e:
        print(f"Error: {e}")