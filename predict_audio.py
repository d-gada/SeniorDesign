"""
predict_audio.py
----------------
Simple CLI for predicting bird species from one audio file using a trained .pt checkpoint.

Supports fine-tuned checkpoints from train.py (best_finetune.pt).
Pretraining checkpoints (best_model.pt) do not contain a classifier head and cannot
produce species predictions.

Examples:
  python predict_audio.py \
      --audio ../OUTSIDE_AUDIO/example.wav \
      --checkpoint ../models/best_finetune.pt \
      --classes_json ./labelled_manifest.json \
      --stats ./new_processed_data/stats.json

  python predict_audio.py \
      --audio ../OUTSIDE_AUDIO/example.wav \
      --checkpoint ../models/best_finetune.pt \
      --classes_csv ../species.csv
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from SeniorDesign.model import BirdMAEClassifier, build_classifier, build_mae
from SeniorDesign.preprocess import (
    TARGET_FRAMES,
    TARGET_MELS,
    audio_to_melspec,
    load_audio,
    segment_audio,
)


def _load_classes(args: argparse.Namespace, ckpt_num_classes: Optional[int]) -> list[str]:
    classes: list[str] = []

    if args.classes_json:
        with open(args.classes_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "classes" in data:
            classes = [str(c) for c in data["classes"]]
        elif isinstance(data, list):
            classes = [str(c) for c in data]
        else:
            raise ValueError("--classes_json must be a JSON list or an object containing a 'classes' list")

    elif args.classes_csv:
        with open(args.classes_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError("--classes_csv appears to have no header row")

            # Prefer scientific names because training labels are often species-level canonical names.
            field = None
            lowered = {name.lower(): name for name in reader.fieldnames}
            for candidate in ["scientific name", "label", "class", "species"]:
                if candidate in lowered:
                    field = lowered[candidate]
                    break
            if field is None:
                field = reader.fieldnames[0]

            classes = [str(row[field]).strip() for row in reader if str(row.get(field, "")).strip()]

    if not classes:
        if ckpt_num_classes is None:
            raise ValueError(
                "No class names provided and checkpoint does not contain num_classes. "
                "Provide --classes_json or --classes_csv."
            )
        classes = [f"class_{i}" for i in range(int(ckpt_num_classes))]

    if ckpt_num_classes is not None and len(classes) != int(ckpt_num_classes):
        raise ValueError(
            f"Class count mismatch: checkpoint expects {ckpt_num_classes}, but provided class list has {len(classes)}"
        )

    return classes


def _load_stats(stats_path: Optional[str]) -> tuple[float, float]:
    if not stats_path:
        return 0.0, 1.0
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    return float(stats.get("mean", 0.0)), float(stats.get("std", 1.0))


def _audio_to_tensor(audio_path: str, mean: float, std: float) -> torch.Tensor:
    waveform, sr = load_audio(audio_path)
    segments = segment_audio(waveform, sr)
    if not segments:
        raise ValueError("No valid 3-second segments were extracted from the audio file")

    specs: list[np.ndarray] = []
    for seg in segments:
        spec = audio_to_melspec(seg, sr)

        if spec.shape != (TARGET_MELS, TARGET_FRAMES):
            if spec.shape[0] > TARGET_MELS:
                spec = spec[:TARGET_MELS, :]
            elif spec.shape[0] < TARGET_MELS:
                pad_h = TARGET_MELS - spec.shape[0]
                spec = np.pad(spec, ((0, pad_h), (0, 0)))

            if spec.shape[1] > TARGET_FRAMES:
                spec = spec[:, :TARGET_FRAMES]
            elif spec.shape[1] < TARGET_FRAMES:
                pad_w = TARGET_FRAMES - spec.shape[1]
                spec = np.pad(spec, ((0, 0), (0, pad_w)))

        spec = (spec.astype(np.float32) - mean) / (std + 1e-8)
        specs.append(spec)

    batch = np.stack(specs, axis=0)
    return torch.from_numpy(batch).unsqueeze(1).float()


def _load_model(checkpoint_path: str, num_classes: int, device: torch.device) -> BirdMAEClassifier:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("cfg", {})

    if "model_state" not in ckpt:
        raise ValueError("Checkpoint is missing 'model_state'; expected a fine-tuned classifier checkpoint")

    mae = build_mae(cfg)
    model = build_classifier(mae, num_classes, cfg)

    try:
        model.load_state_dict(ckpt["model_state"])
    except RuntimeError as exc:
        raise ValueError(
            "Could not load checkpoint weights into classifier model. "
            "You may be using a pretraining checkpoint instead of a fine-tuned checkpoint."
        ) from exc

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def _predict(
    model: BirdMAEClassifier,
    x: torch.Tensor,
    classes: list[str],
    device: torch.device,
    top_k: int,
) -> tuple[list[dict], list[dict]]:
    x = x.to(device)
    logits = model(x)
    probs = F.softmax(logits, dim=-1).cpu().numpy()

    segment_predictions: list[dict] = []
    for i in range(probs.shape[0]):
        p = probs[i]
        top_idx = np.argsort(-p)[:top_k]
        segment_predictions.append(
            {
                "segment": i,
                "top": [
                    {
                        "rank": r + 1,
                        "class_index": int(idx),
                        "species": classes[int(idx)],
                        "confidence": float(p[int(idx)]),
                    }
                    for r, idx in enumerate(top_idx)
                ],
            }
        )

    avg_probs = probs.mean(axis=0)
    top_idx = np.argsort(-avg_probs)[:top_k]
    aggregate = [
        {
            "rank": r + 1,
            "class_index": int(idx),
            "species": classes[int(idx)],
            "confidence": float(avg_probs[int(idx)]),
        }
        for r, idx in enumerate(top_idx)
    ]

    return segment_predictions, aggregate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict bird species from one audio file")
    p.add_argument("--audio", required=True, help="Path to input audio file (.wav/.mp3/.flac/etc.)")
    p.add_argument("--checkpoint", required=True, help="Fine-tuned .pt checkpoint (e.g. best_finetune.pt)")
    p.add_argument("--stats", default=None, help="Optional stats.json with mean/std")

    cls = p.add_mutually_exclusive_group(required=False)
    cls.add_argument("--classes_json", default=None, help="JSON list of class names or {'classes': [...]} file")
    cls.add_argument("--classes_csv", default=None, help="CSV containing class names (tries 'Scientific Name' first)")

    p.add_argument("--top_k", type=int, default=5, help="How many predictions to show")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
    p.add_argument("--output", default=None, help="Optional JSON output file")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_num_classes = ckpt.get("num_classes")

    classes = _load_classes(args, ckpt_num_classes)
    mean, std = _load_stats(args.stats)

    x = _audio_to_tensor(args.audio, mean, std)
    model = _load_model(args.checkpoint, len(classes), device)

    segment_predictions, aggregate = _predict(model, x, classes, device, top_k=max(1, args.top_k))

    print("\nTop predictions (aggregated across segments):")
    for item in aggregate:
        print(
            f"  [{item['rank']}] {item['species']:<40} "
            f"{item['confidence']:.4f} (class {item['class_index']})"
        )

    result = {
        "audio": str(Path(args.audio)),
        "checkpoint": str(Path(args.checkpoint)),
        "num_segments": len(segment_predictions),
        "aggregate": aggregate,
        "segments": segment_predictions,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved prediction JSON to: {out_path}")


if __name__ == "__main__":
    main()
