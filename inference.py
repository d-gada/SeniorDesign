"""
inference.py
------------
Inference utilities for the trained BirdMAE classifier.

Modes:
  single    — classify a single audio file
  batch     — classify all audio files in a directory, output CSV
  xeno      — fetch audio from xeno-canto and classify it

Usage:
    python inference.py single  --audio bird.mp3 --checkpoint runs/finetune/best_finetune.pt --classes classes.json
    python inference.py batch   --input_dir /data/field_recordings --checkpoint ... --classes ...
    python inference.py explain --audio bird.mp3 --checkpoint ... --classes ...  # attention viz
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from SeniorDesign.preprocess import load_audio, segment_audio, audio_to_melspec, \
                       SAMPLE_RATE, CLIP_DURATION, HOP_DURATION, \
                       TARGET_MELS, TARGET_FRAMES
from SeniorDesign.model import build_mae, build_classifier, BirdMAEClassifier, patchify

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    checkpoint_path: str,
    classes_path:    str,
    device:          torch.device,
) -> tuple[BirdMAEClassifier, list[str], dict]:
    """Load model + class list from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt.get("cfg", {})

    with open(classes_path) as f:
        classes = json.load(f)["classes"]

    mae   = build_mae(cfg)
    model = build_classifier(mae, len(classes), cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    log.info(f"Loaded model ({len(classes)} classes) from {checkpoint_path}")
    return model, classes, cfg


# ─────────────────────────────────────────────────────────────────────────────
# Audio → predictions
# ─────────────────────────────────────────────────────────────────────────────

def audio_to_spectrogram_tensor(
    audio_path: str,
    stats:      Optional[dict] = None,
) -> torch.Tensor:
    """Load audio and return stacked spectrogram tensor (N, 1, H, W)."""
    waveform, sr = load_audio(audio_path)
    segments     = segment_audio(waveform, sr)
    if not segments:
        raise ValueError(f"No segments extracted from {audio_path}")

    specs = []
    mean  = stats["mean"] if stats else 0.0
    std   = stats["std"]  if stats else 1.0

    for seg in segments:
        spec = audio_to_melspec(seg, sr)
        if spec.shape != (TARGET_MELS, TARGET_FRAMES):
            spec = spec[:TARGET_MELS, :TARGET_FRAMES] if spec.shape[1] >= TARGET_FRAMES else \
                   np.pad(spec, ((0, 0), (0, TARGET_FRAMES - spec.shape[1])))
        spec = (spec - mean) / (std + 1e-8)
        specs.append(spec)

    tensor = torch.from_numpy(np.stack(specs)).float().unsqueeze(1)  # (N, 1, H, W)
    return tensor


@torch.no_grad()
def predict(
    model:      BirdMAEClassifier,
    classes:    list[str],
    specs:      torch.Tensor,
    device:     torch.device,
    top_k:      int = 5,
    threshold:  float = 0.01,
) -> list[dict]:
    """
    Run model on a batch of spectrograms.
    Returns a list of per-segment dicts with top-k predictions.
    """
    specs = specs.to(device)
    logits = model(specs)                        # (N, C)
    probs  = F.softmax(logits, dim=-1).cpu()     # (N, C)

    results = []
    for i in range(probs.shape[0]):
        p          = probs[i]
        top_ids    = p.argsort(descending=True)[:top_k].tolist()
        top_probs  = p[top_ids].tolist()
        segment_res = {
            "segment": i,
            "predictions": [
                {"rank": r + 1, "species": classes[j], "confidence": round(c, 4)}
                for r, (j, c) in enumerate(zip(top_ids, top_probs))
                if c >= threshold
            ],
        }
        results.append(segment_res)
    return results


def aggregate_predictions(results: list[dict], classes: list[str]) -> dict:
    """Average probabilities across segments and return ranked species."""
    score: dict[str, float] = {c: 0.0 for c in classes}
    for seg in results:
        for pred in seg["predictions"]:
            score[pred["species"]] += pred["confidence"]
    total = sum(score.values()) or 1.0
    ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)
    return {sp: round(v / total, 4) for sp, v in ranked if v > 0}


# ─────────────────────────────────────────────────────────────────────────────
# Attention / saliency visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualize_attention(
    model:      BirdMAEClassifier,
    spec:       torch.Tensor,    # (1, 1, H, W)
    save_path:  str,
    device:     torch.device,
) -> None:
    """
    Approximate saliency map via gradient × input (vanilla backprop).
    Highlights which spectrogram regions influenced the prediction most.
    """
    spec = spec.to(device).requires_grad_(True)
    model.zero_grad()
    logits = model(spec)                          # (1, C)
    top_class = logits.argmax(dim=-1)
    logits[0, top_class].backward()

    saliency = spec.grad.data.abs().squeeze().cpu().numpy()  # (H, W)
    spec_np   = spec.detach().squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4),
                             facecolor="#0D1B2A")
    for ax in axes:
        ax.set_facecolor("#1B2838")

    axes[0].imshow(spec_np, origin="lower", aspect="auto",
                   cmap="magma", interpolation="nearest")
    axes[0].set_title("Input Spectrogram", color="#E0E0E0", fontsize=11)
    axes[0].set_xlabel("Time Frames")
    axes[0].set_ylabel("Mel Bins")

    im = axes[1].imshow(saliency, origin="lower", aspect="auto",
                        cmap="hot", interpolation="nearest")
    axes[1].set_title("Saliency Map (Gradient × Input)", color="#E0E0E0", fontsize=11)
    axes[1].set_xlabel("Time Frames")
    plt.colorbar(im, ax=axes[1])

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Attention map saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Inference modes
# ─────────────────────────────────────────────────────────────────────────────

def run_single(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, cfg = load_model(args.checkpoint, args.classes, device)

    stats = None
    if args.stats:
        with open(args.stats) as f:
            stats = json.load(f)

    specs   = audio_to_spectrogram_tensor(args.audio, stats)
    results = predict(model, classes, specs, device, top_k=args.top_k)
    summary = aggregate_predictions(results, classes)

    print("\n── Segment-level predictions ──────────────────────────────")
    for seg in results:
        print(f"\nSegment {seg['segment']}:")
        for p in seg["predictions"]:
            bar = "█" * int(p["confidence"] * 30)
            print(f"  [{p['rank']}] {p['species']:<35} {p['confidence']:.3f}  {bar}")

    print("\n── Aggregated (whole file) ─────────────────────────────────")
    for sp, score in list(summary.items())[:args.top_k]:
        bar = "█" * int(score * 30)
        print(f"  {sp:<40} {score:.3f}  {bar}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"file": args.audio, "segments": results, "summary": summary}, f, indent=2)
        log.info(f"Results saved → {args.output}")


def run_batch(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, cfg = load_model(args.checkpoint, args.classes, device)

    stats = None
    if args.stats:
        with open(args.stats) as f:
            stats = json.load(f)

    input_dir  = Path(args.input_dir)
    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    files      = [p for p in sorted(input_dir.rglob("*")) if p.suffix.lower() in audio_exts]
    log.info(f"Found {len(files)} audio files")

    rows = []
    for afile in files:
        try:
            specs   = audio_to_spectrogram_tensor(str(afile), stats)
            results = predict(model, classes, specs, device, top_k=3)
            summary = aggregate_predictions(results, classes)
            top1    = list(summary.items())[0] if summary else ("unknown", 0.0)
            top2    = list(summary.items())[1] if len(summary) > 1 else ("", 0.0)
            top3    = list(summary.items())[2] if len(summary) > 2 else ("", 0.0)
            rows.append({
                "file":         str(afile),
                "top1_species": top1[0], "top1_conf": top1[1],
                "top2_species": top2[0], "top2_conf": top2[1],
                "top3_species": top3[0], "top3_conf": top3[1],
            })
            log.info(f"{afile.name} → {top1[0]} ({top1[1]:.3f})")
        except Exception as exc:
            log.warning(f"Failed {afile.name}: {exc}")
            rows.append({"file": str(afile), "error": str(exc)})

    out_csv = args.output or str(input_dir / "predictions.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Batch results saved → {out_csv}")


def run_explain(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, cfg = load_model(args.checkpoint, args.classes, device)

    stats = None
    if args.stats:
        with open(args.stats) as f:
            stats = json.load(f)

    specs     = audio_to_spectrogram_tensor(args.audio, stats)   # (N, 1, H, W)
    save_path = args.output or Path(args.audio).stem + "_saliency.png"
    visualize_attention(model, specs[:1], save_path, device)

    # Also print top predictions for the first segment
    preds = predict(model, classes, specs[:1], device, top_k=5)
    print("\nTop predictions (first segment):")
    for p in preds[0]["predictions"]:
        print(f"  [{p['rank']}] {p['species']:<35} {p['confidence']:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="BirdMAE Inference")
    p.add_argument("--checkpoint", required=True, help="Fine-tuned model checkpoint (.pt)")
    p.add_argument("--classes",    required=True, help="classes.json with species list")
    p.add_argument("--stats",      default=None,  help="Dataset stats JSON (mean/std)")
    p.add_argument("--top_k",      type=int, default=5)

    sub = p.add_subparsers(dest="mode", required=True)

    sg = sub.add_parser("single",  help="Classify a single audio file")
    sg.add_argument("--audio",  required=True)
    sg.add_argument("--output", default=None, help="Save JSON results")

    bt = sub.add_parser("batch",   help="Classify a directory of audio files")
    bt.add_argument("--input_dir", required=True)
    bt.add_argument("--output",    default=None, help="Output CSV path")

    ex = sub.add_parser("explain", help="Saliency map for one audio file")
    ex.add_argument("--audio",  required=True)
    ex.add_argument("--output", default=None, help="Output PNG path")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Re-attach shared args to sub-parsers (workaround for argparse subparser scoping)
    if args.mode == "single":
        run_single(args)
    elif args.mode == "batch":
        run_batch(args)
    elif args.mode == "explain":
        run_explain(args)
