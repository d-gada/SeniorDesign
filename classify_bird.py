"""
classify_bird.py
----------------
Bird species classification using BirdNET weights.

Usage:
    python classify_bird.py --audio test.wav --checkpoint ./model_output_finetune/checkpoints/birdnet_weights.pt --classes labelled_manifest.json
    
    Or with optional parameters:
    python classify_bird.py \
        --audio test.wav \
        --checkpoint ./model_output_finetune/checkpoints/birdnet_weights.pt \
        --classes labelled_manifest.json \
        --top_k 5 \
        --output results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn.functional as F

# Import from local modules
from preprocess import (
    TARGET_FRAMES,
    TARGET_MELS,
    audio_to_melspec,
    load_audio,
    segment_audio,
)
from model import BirdMAEClassifier, build_classifier, build_mae


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def load_class_names(classes_path: str) -> List[str]:
    """
    Load class names from JSON file.
    Supports both list format and {'classes': [...]} format.
    
    Args:
        classes_path: Path to JSON file containing class names
        
    Returns:
        List of class name strings
    """
    try:
        with open(classes_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle both formats
        if isinstance(data, dict) and "classes" in data:
            classes = [str(c) for c in data["classes"]]
        elif isinstance(data, list):
            classes = [str(c) for c in data]
        else:
            raise ValueError("JSON must be a list or object with 'classes' key")
        
        if not classes:
            raise ValueError("No classes found in JSON file")
        
        print(f"✓ Loaded {len(classes)} classes from {classes_path}")
        return classes
    
    except FileNotFoundError:
        print(f"✗ Error: Classes file not found: {classes_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"✗ Error: Invalid JSON in {classes_path}")
        sys.exit(1)


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[dict, int]:
    """
    Load checkpoint and extract metadata.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: torch device (cpu/cuda)
        
    Returns:
        Tuple of (checkpoint dict, num_classes)
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
        
        # Try to get num_classes from checkpoint
        num_classes = checkpoint.get("num_classes")
        if num_classes:
            print(f"✓ Checkpoint contains {num_classes} classes")
        
        return checkpoint, num_classes
    
    except FileNotFoundError:
        print(f"✗ Error: Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        sys.exit(1)


def audio_to_tensor(audio_path: str, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    """
    Convert audio file to normalized mel spectrogram tensor.
    
    Args:
        audio_path: Path to audio file
        mean: Mean for normalization
        std: Std dev for normalization
        
    Returns:
        Tensor of shape (num_segments, 1, 128, 224)
    """
    try:
        # Load and segment audio
        waveform, sr = load_audio(audio_path)
        segments = segment_audio(waveform, sr)
        
        if not segments:
            print(f"✗ Error: No valid audio segments extracted from {audio_path}")
            sys.exit(1)
        
        print(f"✓ Extracted {len(segments)} audio segments from {audio_path}")
        
        # Convert each segment to mel spectrogram
        specs = []
        for seg in segments:
            spec = audio_to_melspec(seg, sr)
            
            # Ensure correct shape
            spec = _pad_or_crop_spec(spec)
            
            # Normalize
            spec = (spec.astype(np.float32) - mean) / (std + 1e-8)
            specs.append(spec)
        
        # Stack into batch
        batch = np.stack(specs, axis=0)  # (num_segments, 128, 224)
        tensor = torch.from_numpy(batch).unsqueeze(1).float()  # (num_segments, 1, 128, 224)
        
        return tensor
    
    except FileNotFoundError:
        print(f"✗ Error: Audio file not found: {audio_path}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error processing audio: {e}")
        sys.exit(1)


def _pad_or_crop_spec(spec: np.ndarray) -> np.ndarray:
    """
    Ensure spectrogram has correct shape (TARGET_MELS, TARGET_FRAMES).
    Crops or pads as needed.
    """
    # Handle height (mel bins)
    if spec.shape[0] > TARGET_MELS:
        spec = spec[:TARGET_MELS, :]
    elif spec.shape[0] < TARGET_MELS:
        pad_h = TARGET_MELS - spec.shape[0]
        spec = np.pad(spec, ((0, pad_h), (0, 0)))
    
    # Handle width (time frames)
    if spec.shape[1] > TARGET_FRAMES:
        spec = spec[:, :TARGET_FRAMES]
    elif spec.shape[1] < TARGET_FRAMES:
        pad_w = TARGET_FRAMES - spec.shape[1]
        spec = np.pad(spec, ((0, 0), (0, pad_w)))
    
    return spec


def build_model(checkpoint: dict, num_classes: int, device: torch.device) -> BirdMAEClassifier:
    """
    Create and load the BirdMAE classifier model.
    
    Args:
        checkpoint: Loaded checkpoint dictionary
        num_classes: Number of output classes
        device: torch device
        
    Returns:
        Loaded BirdMAEClassifier model in eval mode
    """
    try:
        cfg = checkpoint.get("cfg", {})
        
        # Build MAE backbone and classifier head
        mae = build_mae(cfg)
        model = build_classifier(mae, num_classes, cfg)
        
        # Load weights
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
            print(f"✓ Loaded model weights from checkpoint")
        else:
            print("⚠ Warning: Checkpoint has no 'model_state', using untrained model")
        
        model = model.to(device)
        model.eval()
        
        return model
    
    except Exception as e:
        print(f"✗ Error building model: {e}")
        sys.exit(1)


@torch.no_grad()
def predict(
    model: BirdMAEClassifier,
    x: torch.Tensor,
    classes: List[str],
    device: torch.device,
    top_k: int = 5,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Run inference on audio segments.
    
    Args:
        model: BirdMAEClassifier model
        x: Input tensor of shape (num_segments, 1, 128, 224)
        classes: List of class names
        device: torch device
        top_k: Number of top predictions to return
        
    Returns:
        Tuple of (per_segment_predictions, aggregated_predictions)
    """
    x = x.to(device)
    
    # Forward pass
    logits = model(x)
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    
    # Per-segment predictions
    segment_predictions = []
    for i in range(probs.shape[0]):
        p = probs[i]
        top_idx = np.argsort(-p)[:top_k]
        
        segment_predictions.append({
            "segment": i,
            "predictions": [
                {
                    "rank": r + 1,
                    "species": classes[int(idx)],
                    "confidence": float(p[int(idx)]),
                    "class_id": int(idx),
                }
                for r, idx in enumerate(top_idx)
            ],
        })
    
    # Aggregate across segments (average probabilities)
    avg_probs = probs.mean(axis=0)
    top_idx = np.argsort(-avg_probs)[:top_k]
    
    aggregated = [
        {
            "rank": r + 1,
            "species": classes[int(idx)],
            "confidence": float(avg_probs[int(idx)]),
            "class_id": int(idx),
        }
        for r, idx in enumerate(top_idx)
    ]
    
    return segment_predictions, aggregated


def print_results(
    audio_path: str,
    aggregated: List[Dict],
    segment_predictions: List[Dict],
    top_k: int,
) -> None:
    """Pretty print classification results."""
    print("\n" + "=" * 70)
    print(f"Classification Results: {Path(audio_path).name}")
    print("=" * 70)
    
    print(f"\nTop {top_k} Predictions (Aggregated):")
    print("-" * 70)
    
    for pred in aggregated:
        print(
            f"  [{pred['rank']}] {pred['species']:<35} "
            f"Confidence: {pred['confidence']:.4f}"
        )
    
    if len(segment_predictions) > 1:
        print(f"\nTop Prediction by Segment:")
        print("-" * 70)
        for seg in segment_predictions:
            top = seg["predictions"][0]
            print(
                f"  Segment {seg['segment']}: {top['species']:<30} "
                f"({top['confidence']:.4f})"
            )
    
    print("=" * 70 + "\n")


def save_results(
    output_path: str,
    audio_path: str,
    checkpoint_path: str,
    aggregated: List[Dict],
    segment_predictions: List[Dict],
) -> None:
    """Save results to JSON file."""
    results = {
        "audio_file": str(Path(audio_path).absolute()),
        "checkpoint": str(Path(checkpoint_path).absolute()),
        "num_segments": len(segment_predictions),
        "aggregated_predictions": aggregated,
        "segment_predictions": segment_predictions,
    }
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {output_file}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Classify bird species from audio using BirdNET weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python classify_bird.py --audio recording.wav --checkpoint birdnet_weights.pt --classes labels.json
  python classify_bird.py --audio recording.wav --checkpoint birdnet_weights.pt --classes labels.json --output predictions.json
        """,
    )
    
    parser.add_argument(
        "--audio",
        required=True,
        help="Path to audio file (.wav, .mp3, .flac, etc.)",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint file (.pt) with BirdNET weights",
    )
    parser.add_argument(
        "--classes",
        required=True,
        help="Path to JSON file with class names",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top predictions to show (default: 5)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional: Save results to JSON file",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for inference (default: auto)",
    )
    
    args = parser.parse_args()
    
    # Select device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load classes
    classes = load_class_names(args.classes)
    num_classes = len(classes)
    
    # Load checkpoint
    checkpoint, ckpt_num_classes = load_checkpoint(args.checkpoint, device)
    
    # Verify class count matches
    if ckpt_num_classes and ckpt_num_classes != num_classes:
        print(f"⚠ Warning: Checkpoint expects {ckpt_num_classes} classes, "
              f"but {num_classes} provided")
    
    # Process audio
    print(f"\nProcessing audio: {args.audio}")
    x = audio_to_tensor(args.audio)
    
    # Build model
    model = build_model(checkpoint, num_classes, device)
    
    # Run inference
    print(f"\nRunning inference...")
    segment_predictions, aggregated = predict(
        model, x, classes, device, top_k=args.top_k
    )
    
    # Display results
    print_results(args.audio, aggregated, segment_predictions, args.top_k)
    
    # Save if requested
    if args.output:
        save_results(
            args.output,
            args.audio,
            args.checkpoint,
            aggregated,
            segment_predictions,
        )


if __name__ == "__main__":
    main()
