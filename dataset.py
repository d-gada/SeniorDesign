"""
dataset.py
----------
PyTorch Dataset classes for the Amazonian Bird MAE pipeline.

  - BirdSpectrogramDataset  : unlabelled, for MAE pre-training
  - LabelledBirdDataset     : labelled, for supervised fine-tuning
"""

import json
import random
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# ─────────────────────────────────────────────────────────────────────────────
# Unlabelled dataset (MAE pre-training)
# ─────────────────────────────────────────────────────────────────────────────

class BirdSpectrogramDataset(Dataset):
    """
    Loads pre-computed mel-spectrogram .npy files produced by preprocess.py.

    Each item is a (1, H, W) float32 tensor, normalised with the dataset
    mean / std if stats are provided.
    """

    def __init__(
        self,
        manifest_path: str,
        stats_path:    Optional[str] = None,
        transform:     Optional[Callable] = None,
    ):
        with open(manifest_path) as f:
            data = json.load(f)
        self.paths = data["spectrograms"]

        self.mean = 0.0
        self.std  = 1.0
        if stats_path and Path(stats_path).exists():
            with open(stats_path) as f:
                stats     = json.load(f)
                self.mean = stats["mean"]
                self.std  = stats["std"]

        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        spec = np.load(self.paths[idx]).astype(np.float32)   # (H, W)
        spec = (spec - self.mean) / (self.std + 1e-8)
        spec = torch.from_numpy(spec).unsqueeze(0)           # (1, H, W)
        if self.transform:
            spec = self.transform(spec)
        return spec


# ─────────────────────────────────────────────────────────────────────────────
# Labelled dataset (supervised fine-tuning)
# ─────────────────────────────────────────────────────────────────────────────

class LabelledBirdDataset(Dataset):
    """
    Expects a JSON manifest like:
        {
          "samples": [
            {"path": "/abs/path/to/spec.npy", "label": "Pipra_filicauda"},
            ...
          ],
          "classes": ["Lepidothrix_coronata", "Pipra_filicauda", ...]
        }

    Exposes class_to_idx and idx_to_class dicts for downstream use.
    """

    def __init__(
        self,
        manifest_path: str,
        stats_path:    Optional[str] = None,
        transform:     Optional[Callable] = None,
        split:         str = "train",          # "train" | "val" | "test"
        val_frac:      float = 0.10,
        test_frac:     float = 0.05,
        seed:          int   = 42,
    ):
        with open(manifest_path) as f:
            data = json.load(f)

        samples = data["samples"]
        classes = data["classes"]
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        self.num_classes  = len(classes)

        # Deterministic train / val / test split
        rng = random.Random(seed)
        rng.shuffle(samples)
        n        = len(samples)
        n_test   = int(n * test_frac)
        n_val    = int(n * val_frac)
        if split == "test":
            samples = samples[:n_test]
        elif split == "val":
            samples = samples[n_test : n_test + n_val]
        else:
            samples = samples[n_test + n_val :]

        self.samples = samples

        self.mean = 0.0
        self.std  = 1.0
        if stats_path and Path(stats_path).exists():
            with open(stats_path) as f:
                s         = json.load(f)
                self.mean = s["mean"]
                self.std  = s["std"]

        self.transform = transform

    # ── Class-balance helpers ─────────────────────────────────────────────────

    def class_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for s in self.samples:
            idx = self.class_to_idx[s["label"]]
            counts[idx] = counts.get(idx, 0) + 1
        return counts

    def weighted_sampler(self) -> WeightedRandomSampler:
        """Returns a WeightedRandomSampler for handling class imbalance."""
        counts  = self.class_counts()
        weights = [1.0 / counts[self.class_to_idx[s["label"]]] for s in self.samples]
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    # ── Dataset protocol ─────────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        spec   = np.load(sample["path"]).astype(np.float32)
        spec   = (spec - self.mean) / (self.std + 1e-8)
        spec   = torch.from_numpy(spec).unsqueeze(0)           # (1, H, W)
        label  = self.class_to_idx[sample["label"]]
        if self.transform:
            spec = self.transform(spec)
        return spec, label


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def make_pretrain_loaders(
    manifest_path: str,
    stats_path:    Optional[str]  = None,
    batch_size:    int  = 64,
    num_workers:   int  = 0,
    val_frac:      float = 0.05,
    seed:          int  = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Split the unlabelled manifest into train/val loaders for MAE pre-training.
    """
    with open(manifest_path) as f:
        paths = json.load(f)["spectrograms"]

    rng = random.Random(seed)
    rng.shuffle(paths)
    n_val    = max(1, int(len(paths) * val_frac))
    val_paths   = paths[:n_val]
    train_paths = paths[n_val:]

    def _write_tmp_manifest(p_list, suffix):
        import tempfile, json as _json
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=f"_{suffix}.json", delete=False
        )
        _json.dump({"spectrograms": p_list, "count": len(p_list)}, tmp)
        tmp.close()
        return tmp.name

    train_manifest = _write_tmp_manifest(train_paths, "train")
    val_manifest   = _write_tmp_manifest(val_paths,   "val")

    train_ds = BirdSpectrogramDataset(train_manifest, stats_path)
    val_ds   = BirdSpectrogramDataset(val_manifest,   stats_path)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


def make_finetune_loaders(
    manifest_path: str,
    stats_path:    Optional[str] = None,
    batch_size:    int  = 32,
    num_workers:   int  = 0,
    balanced:      bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Returns (train_loader, val_loader, test_loader, num_classes).
    """
    train_ds = LabelledBirdDataset(manifest_path, stats_path, split="train")
    val_ds   = LabelledBirdDataset(manifest_path, stats_path, split="val")
    test_ds  = LabelledBirdDataset(manifest_path, stats_path, split="test")

    sampler = train_ds.weighted_sampler() if balanced else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader, train_ds.num_classes
