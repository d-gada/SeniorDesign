"""
preprocess.py
-------------
Audio preprocessing pipeline for Amazonian bird call recordings.
Converts raw audio files into mel spectrogram patches ready for MAE training.

Usage:
    python preprocess.py --input_dir /path/to/audio --output_dir /path/to/output
"""

import os
import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import librosa
import librosa.display
from tqdm import tqdm

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Constants (BirdNET-compatible defaults) ───────────────────────────────────
SAMPLE_RATE       = 48_000       # BirdNET native sample rate
CLIP_DURATION     = 3.0          # seconds per clip
HOP_DURATION      = 1.5          # overlap stride (50%)
N_FFT             = 1024
HOP_LENGTH        = 320          # ~150 frames for 3 s clip at 48 kHz
N_MELS            = 128
F_MIN             = 50.0
F_MAX             = 15_000.0
PATCH_SIZE        = 16           # pixels; spectrogram split into 16×16 patches
TARGET_FRAMES     = 224          # time axis after resize
TARGET_MELS       = 128          # freq axis (= N_MELS, no resize needed usually)
SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


# ─────────────────────────────────────────────────────────────────────────────
# Audio utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> Tuple[torch.Tensor, int]:
    """Load audio file and resample to target_sr. Returns mono waveform."""
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)   # stereo → mono
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform, target_sr


def segment_audio(
    waveform: torch.Tensor,
    sr: int,
    clip_duration: float = CLIP_DURATION,
    hop_duration: float  = HOP_DURATION,
) -> list[torch.Tensor]:
    """Slice waveform into overlapping clips of fixed length."""
    clip_samples = int(clip_duration * sr)
    hop_samples  = int(hop_duration * sr)
    segments = []
    start = 0
    while start + clip_samples <= waveform.shape[-1]:
        seg = waveform[:, start : start + clip_samples]
        segments.append(seg)
        start += hop_samples
    return segments


def audio_to_melspec(
    waveform: torch.Tensor,
    sr:       int = SAMPLE_RATE,
    n_fft:    int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels:   int = N_MELS,
    f_min:    float = F_MIN,
    f_max:    float = F_MAX,
    top_db:   float = 80.0,
) -> np.ndarray:
    """
    Compute a log-mel spectrogram from a waveform tensor.
    Returns a 2-D numpy array (n_mels × time_frames), normalised to [0, 1].
    """
    mel_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
    )
    mel = mel_transform(waveform.squeeze(0))          # (n_mels, T)
    mel_db = T.AmplitudeToDB(top_db=top_db)(mel)      # log scale
    mel_np = mel_db.numpy()

    # Normalise to [0, 1]
    mel_np = (mel_np - mel_np.min()) / (mel_np.max() - mel_np.min() + 1e-8)
    return mel_np.astype(np.float32)


def resize_spectrogram(spec: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize spectrogram to (target_h, target_w) using librosa."""
    return librosa.util.fix_length(
        librosa.resample(spec, orig_sr=spec.shape[1], target_sr=target_w, axis=1),
        size=target_h,
        axis=0,
    )


def augment_spectrogram(spec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Lightweight augmentations suitable for tropical soundscape recordings:
      - Time shift
      - Frequency shift
      - Additive Gaussian noise (simulates rain / insect background)
      - Random gain
    """
    # Time shift (±10 %)
    shift = int(spec.shape[1] * rng.uniform(-0.1, 0.1))
    spec  = np.roll(spec, shift, axis=1)

    # Frequency shift (±5 mel bins)
    f_shift = int(rng.uniform(-5, 5))
    spec    = np.roll(spec, f_shift, axis=0)

    # Additive noise (tropical background noise simulation)
    noise_level = rng.uniform(0.0, 0.04)
    spec = spec + noise_level * rng.standard_normal(spec.shape).astype(np.float32)

    # Random gain
    gain = rng.uniform(0.8, 1.2)
    spec = spec * gain

    return np.clip(spec, 0.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main preprocessing pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_file(
    audio_path:    str,
    output_dir:    str,
    augment:       bool = True,
    n_augmentations: int = 2,
    seed:          int  = 42,
) -> list[str]:
    """
    Process a single audio file → list of .npy spectrogram paths.
    Returns paths of all saved spectrograms.
    """
    rng = np.random.default_rng(seed)
    saved = []

    try:
        waveform, sr = load_audio(audio_path)
    except Exception as exc:
        log.warning(f"Could not load {audio_path}: {exc}")
        return []

    segments = segment_audio(waveform, sr)
    if not segments:
        log.warning(f"No segments extracted from {audio_path}")
        return []

    stem     = Path(audio_path).stem
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for idx, seg in enumerate(segments):
        spec = audio_to_melspec(seg, sr)

        # Resize to standard dimensions
        if spec.shape != (TARGET_MELS, TARGET_FRAMES):
            try:
                spec = resize_spectrogram(spec, TARGET_MELS, TARGET_FRAMES)
            except Exception:
                spec = spec[:TARGET_MELS, :TARGET_FRAMES] if spec.shape[1] >= TARGET_FRAMES else \
                       np.pad(spec, ((0, 0), (0, TARGET_FRAMES - spec.shape[1])))

        # Save original
        fname = out_path / f"{stem}_{idx:04d}.npy"
        np.save(fname, spec)
        saved.append(str(fname))

        # Save augmentations
        if augment:
            for aug_idx in range(n_augmentations):
                aug_spec = augment_spectrogram(spec.copy(), rng)
                aug_fname = out_path / f"{stem}_{idx:04d}_aug{aug_idx}.npy"
                np.save(aug_fname, aug_spec)
                saved.append(str(aug_fname))

    return saved


def build_manifest(
    audio_dir:   str,
    output_dir:  str,
    augment:     bool = True,
    n_aug:       int  = 2,
) -> str:
    """
    Process all audio files in audio_dir and write a JSON manifest listing
    all produced spectrogram .npy file paths.
    """
    audio_dir  = Path(audio_dir)
    output_dir = Path(output_dir)
    spec_dir   = output_dir / "spectrograms"

    audio_files = [
        p for p in sorted(audio_dir.rglob("*"))
        if p.suffix.lower() in SUPPORTED_FORMATS
    ]
    log.info(f"Found {len(audio_files)} audio files in {audio_dir}")

    all_paths = []
    for afile in tqdm(audio_files, desc="Processing audio"):
        paths = process_file(str(afile), str(spec_dir), augment=augment, n_augmentations=n_aug)
        all_paths.extend(paths)

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"spectrograms": all_paths, "count": len(all_paths)}, f, indent=2)

    log.info(f"Manifest written → {manifest_path}  ({len(all_paths)} spectrograms)")
    return str(manifest_path)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset statistics (mean / std for normalisation)
# ─────────────────────────────────────────────────────────────────────────────

def compute_dataset_stats(manifest_path: str, sample_frac: float = 0.1) -> dict:
    """Compute per-pixel mean and std over a random subset of spectrograms."""
    with open(manifest_path) as f:
        paths = json.load(f)["spectrograms"]

    rng     = np.random.default_rng(0)
    sample  = rng.choice(paths, size=max(1, int(len(paths) * sample_frac)), replace=False)
    running = []
    for p in tqdm(sample, desc="Computing stats"):
        spec = np.load(p)
        running.append(spec)

    data  = np.stack(running)
    stats = {"mean": float(data.mean()), "std": float(data.std())}
    log.info(f"Dataset stats — mean: {stats['mean']:.4f}, std: {stats['std']:.4f}")

    stats_path = Path(manifest_path).parent / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"Stats saved → {stats_path}")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Preprocess Amazonian bird audio for MAE training")
    p.add_argument("--input_dir",  required=True, help="Root directory of raw audio files")
    p.add_argument("--output_dir", required=True, help="Destination directory for spectrograms + manifest")
    p.add_argument("--no_augment", action="store_true", help="Disable data augmentation")
    p.add_argument("--n_aug",      type=int, default=2, help="Number of augmented copies per clip")
    p.add_argument("--stats",      action="store_true", help="Compute dataset mean/std after processing")
    return p.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    manifest = build_manifest(
        args.input_dir,
        args.output_dir,
        augment=not args.no_augment,
        n_aug=args.n_aug,
    )
    if args.stats:
        compute_dataset_stats(manifest)
