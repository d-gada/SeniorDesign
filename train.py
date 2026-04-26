"""
train.py
--------
Two-phase training script for the Amazonian Bird MAE pipeline.

Phase 1 — MAE pre-training  (self-supervised, unlabelled data)
Phase 2 — Fine-tuning       (supervised, labelled data)

Produces live loss / F1 visualisations saved to <output_dir>/plots/.

Usage:
    # Phase 1
    python train.py pretrain \
        --manifest data/manifest.json \
        --stats    data/stats.json \
        --output   runs/pretrain

    # Phase 2
    python train.py finetune \
        --manifest  data/labelled_manifest.json \
        --stats     data/stats.json \
        --checkpoint runs/pretrain/best_model.pt \
        --output    runs/finetune
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score, classification_report
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from SeniorDesign.dataset import make_pretrain_loaders, make_finetune_loaders
from SeniorDesign.model import build_mae, build_classifier, BirdMAE, BirdMAEClassifier

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Plotting utilities
# ─────────────────────────────────────────────────────────────────────────────

BIRD_PALETTE = {
    "bg":       "#0D1B2A",
    "surface":  "#1B2838",
    "accent1":  "#4FC3F7",   # sky blue
    "accent2":  "#81C784",   # leaf green
    "accent3":  "#FFB74D",   # amber
    "accent4":  "#F06292",   # pink
    "text":     "#E0E0E0",
    "grid":     "#263040",
}

plt.rcParams.update({
    "figure.facecolor":  BIRD_PALETTE["bg"],
    "axes.facecolor":    BIRD_PALETTE["surface"],
    "axes.edgecolor":    BIRD_PALETTE["grid"],
    "axes.labelcolor":   BIRD_PALETTE["text"],
    "xtick.color":       BIRD_PALETTE["text"],
    "ytick.color":       BIRD_PALETTE["text"],
    "text.color":        BIRD_PALETTE["text"],
    "grid.color":        BIRD_PALETTE["grid"],
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "legend.framealpha": 0.3,
    "legend.facecolor":  BIRD_PALETTE["surface"],
    "legend.edgecolor":  BIRD_PALETTE["grid"],
    "font.family":       "monospace",
})


def _smooth(values: list[float], weight: float = 0.6) -> list[float]:
    """Exponential moving average smoothing."""
    out, last = [], values[0] if values else 0.0
    for v in values:
        last = last * weight + v * (1 - weight)
        out.append(last)
    return out


def plot_pretrain_curves(
    train_losses: list[float],
    val_losses:   list[float],
    save_path:    str,
    title:        str = "MAE Pre-training — Reconstruction Loss",
) -> None:
    """Save a polished loss-curve plot for the pre-training phase."""
    epochs = list(range(1, len(train_losses) + 1))
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold", color=BIRD_PALETTE["text"], y=1.01)

    ax.plot(epochs, train_losses, color=BIRD_PALETTE["accent1"],
            linewidth=1.2, alpha=0.4, label="Train (raw)")
    ax.plot(epochs, _smooth(train_losses), color=BIRD_PALETTE["accent1"],
            linewidth=2.2, label="Train (smoothed)")

    if val_losses:
        ax.plot(epochs, val_losses, color=BIRD_PALETTE["accent2"],
                linewidth=1.2, alpha=0.4, label="Val (raw)")
        ax.plot(epochs, _smooth(val_losses), color=BIRD_PALETTE["accent2"],
                linewidth=2.2, label="Val (smoothed)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(True)

    # Annotate best val
    if val_losses:
        best_ep  = int(np.argmin(val_losses)) + 1
        best_val = min(val_losses)
        ax.axvline(best_ep, color=BIRD_PALETTE["accent3"], linestyle=":", linewidth=1.5)
        ax.annotate(
            f"Best val\n{best_val:.4f}",
            xy=(best_ep, best_val),
            xytext=(best_ep + 0.5, best_val * 1.05),
            color=BIRD_PALETTE["accent3"],
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved plot → {save_path}")


def plot_finetune_curves(
    train_losses: list[float],
    val_losses:   list[float],
    train_f1s:    list[float],
    val_f1s:      list[float],
    save_path:    str,
    title:        str = "Fine-tuning — Loss & Macro F1",
) -> None:
    """Dual-panel plot: loss (top) and macro F1 (bottom)."""
    epochs = list(range(1, len(train_losses) + 1))

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold",
                 color=BIRD_PALETTE["text"], y=1.01)
    gs  = gridspec.GridSpec(2, 1, hspace=0.4, figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # ── Loss panel ───────────────────────────────────────────────────────────
    ax1.plot(epochs, train_losses, color=BIRD_PALETTE["accent1"],
             linewidth=1.0, alpha=0.35, label="Train loss (raw)")
    ax1.plot(epochs, _smooth(train_losses), color=BIRD_PALETTE["accent1"],
             linewidth=2.0, label="Train loss")
    ax1.plot(epochs, val_losses, color=BIRD_PALETTE["accent2"],
             linewidth=1.0, alpha=0.35, label="Val loss (raw)")
    ax1.plot(epochs, _smooth(val_losses), color=BIRD_PALETTE["accent2"],
             linewidth=2.0, label="Val loss")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend(fontsize=8)
    ax1.grid(True)

    # ── F1 panel ─────────────────────────────────────────────────────────────
    ax2.plot(epochs, train_f1s, color=BIRD_PALETTE["accent3"],
             linewidth=1.0, alpha=0.35, label="Train F1 (raw)")
    ax2.plot(epochs, _smooth(train_f1s), color=BIRD_PALETTE["accent3"],
             linewidth=2.0, label="Train F1")
    ax2.plot(epochs, val_f1s, color=BIRD_PALETTE["accent4"],
             linewidth=1.0, alpha=0.35, label="Val F1 (raw)")
    ax2.plot(epochs, _smooth(val_f1s), color=BIRD_PALETTE["accent4"],
             linewidth=2.0, label="Val F1")

    # Annotate best val F1
    if val_f1s:
        best_ep = int(np.argmax(val_f1s)) + 1
        best_f1 = max(val_f1s)
        ax2.axvline(best_ep, color="white", linestyle=":", linewidth=1.5)
        ax2.annotate(
            f"Best\n{best_f1:.3f}",
            xy=(best_ep, best_f1),
            xytext=(best_ep + 0.3, best_f1 - 0.03),
            color="white", fontsize=8,
        )

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro F1 Score")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved plot → {save_path}")


def plot_reconstruction(
    originals:     torch.Tensor,   # (N, 1, H, W)
    reconstructed: torch.Tensor,   # (N, 1, H, W)
    masked_input:  torch.Tensor,   # (N, 1, H, W)
    save_path:     str,
    n_show:        int = 4,
) -> None:
    """Side-by-side visualisation: original | masked | reconstruction."""
    N   = min(n_show, originals.shape[0])
    fig, axes = plt.subplots(3, N, figsize=(N * 3.5, 9))
    fig.suptitle("Masked Autoencoder — Reconstruction Examples",
                 fontsize=13, fontweight="bold", color=BIRD_PALETTE["text"])

    row_labels = ["Original", "Masked Input", "Reconstructed"]
    imgs_rows  = [originals, masked_input, reconstructed]

    for row, (label, imgs) in enumerate(zip(row_labels, imgs_rows)):
        for col in range(N):
            ax  = axes[row, col] if N > 1 else axes[row]
            img = imgs[col, 0].cpu().numpy()
            ax.imshow(img, origin="lower", aspect="auto", cmap="magma",
                      interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label, fontsize=10, color=BIRD_PALETTE["text"])

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved reconstruction plot → {save_path}")


def plot_per_class_f1(
    class_names:  list[str],
    f1_scores:    list[float],
    save_path:    str,
    title:        str = "Per-Class F1 Score (Test Set)",
) -> None:
    """Horizontal bar chart of per-class F1 scores."""
    sorted_pairs = sorted(zip(f1_scores, class_names))
    scores, names = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.35)))
    fig.suptitle(title, fontsize=13, fontweight="bold", color=BIRD_PALETTE["text"])

    cmap   = plt.cm.RdYlGn
    colors = [cmap(s) for s in scores]
    bars   = ax.barh(names, scores, color=colors, edgecolor="none", height=0.7)

    for bar, score in zip(bars, scores):
        ax.text(
            score + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{score:.2f}", va="center", ha="left",
            fontsize=7, color=BIRD_PALETTE["text"],
        )

    ax.set_xlim(0, 1.12)
    ax.set_xlabel("F1 Score")
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="y", labelsize=7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved per-class F1 plot → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler / optimiser helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> optim.AdamW:
    # Don't apply weight decay to LayerNorm or bias
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )


def warmup_cosine_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — MAE Pre-training
# ─────────────────────────────────────────────────────────────────────────────

def train_pretrain_epoch(
    model:     BirdMAE,
    loader,
    optimizer: optim.Optimizer,
    scaler:    GradScaler,
    device:    torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_steps = len(loader)
    t0 = time.time()
    for step, batch in enumerate(loader, start=1):
        imgs = batch.to(device)
        optimizer.zero_grad()
        with autocast():
            _, _, loss = model(imgs)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        if log_interval > 0 and (step % log_interval == 0 or step == total_steps):
            elapsed = time.time() - t0
            steps_left = max(total_steps - step, 0)
            sec_per_step = elapsed / max(step, 1)
            eta_sec = sec_per_step * steps_left
            log.info(
                f"  [Pretrain][Train] step {step}/{total_steps}  "
                f"loss={loss.item():.4f}  "
                f"eta={eta_sec/60.0:.1f}m"
            )
    return total_loss / len(loader)


@torch.no_grad()
def eval_pretrain_epoch(
    model:  BirdMAE,
    loader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    for batch in loader:
        imgs = batch.to(device)
        with autocast():
            _, _, loss = model(imgs)
        total_loss += loss.item()
    return total_loss / len(loader)


def run_pretrain(args) -> str:
    """Full pre-training loop. Returns path to best checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Pre-training on {device}")

    out_dir   = Path(args.output)
    plot_dir  = out_dir / "plots"
    ckpt_dir  = out_dir / "checkpoints"
    plot_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    cfg = {}
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            cfg = json.load(f)

    log.info("STEP 1: Building dataloaders...")
    train_loader, val_loader = make_pretrain_loaders(
        args.manifest,
        stats_path   = args.stats,
        batch_size   = cfg.get("batch_size", 64),
        num_workers  = cfg.get("num_workers", 0),
    )

    log.info("STEP 2: Building model...")
    model = build_mae(cfg).to(device)
    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    log.info("STEP 3: Creating optimizer...")
    optimizer = build_optimizer(model, cfg.get("lr", 1e-6), cfg.get("weight_decay", 0.05))
    scheduler = warmup_cosine_scheduler(
        optimizer,
        warmup_epochs = cfg.get("warmup_epochs", 3),
        total_epochs  = cfg.get("epochs", 10),
    )
    scaler  = GradScaler()
    epochs  = cfg.get("epochs", 3)

    train_losses, val_losses = [], []
    best_val = float("inf")
    best_ckpt = str(ckpt_dir / "best_model.pt")

    # Optional: save example reconstructions
    vis_batch = next(iter(val_loader)).to(device)[:8]

    log.info("STEP 4: Starting training loop...")
    for epoch in range(1, epochs + 1):
        t0         = time.time()
        train_loss = train_pretrain_epoch(model, train_loader, optimizer, scaler, device)
        val_loss   = eval_pretrain_epoch(model, val_loader, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        lr = optimizer.param_groups[0]["lr"]
        log.info(
            f"[Pretrain] Epoch {epoch:03d}/{epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"lr={lr:.2e}  t={time.time()-t0:.1f}s"
        )

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "val_loss": best_val, "cfg": cfg},
                best_ckpt,
            )
            log.info(f"  ✓ New best checkpoint saved ({best_val:.4f})")

        # Periodic checkpoint
        if epoch % cfg.get("save_every", 10) == 0:
            ep_ckpt = ckpt_dir / f"epoch_{epoch:04d}.pt"
            torch.save({"epoch": epoch, "model_state": model.state_dict()}, ep_ckpt)

        # Update loss plot every epoch
        plot_pretrain_curves(
            train_losses, val_losses,
            save_path=str(plot_dir / "pretrain_loss.png"),
        )

        # Reconstruction visualisation every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            rec_img, masked_img = model.reconstruct(vis_batch)
            plot_reconstruction(
                vis_batch, rec_img, masked_img,
                save_path=str(plot_dir / f"reconstruction_ep{epoch:04d}.png"),
            )

    log.info(f"Pre-training complete. Best val loss: {best_val:.4f}")
    log.info(f"Best checkpoint: {best_ckpt}")
    log.info("Training complete.")
    return best_ckpt

    


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Supervised Fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

def compute_f1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1).cpu().numpy()
    labs  = labels.cpu().numpy()
    return f1_score(labs, preds, average="macro", zero_division=0)


def train_finetune_epoch(
    model:     BirdMAEClassifier,
    loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler:    GradScaler,
    device:    torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss, all_logits, all_labels = 0.0, [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            logits = model(imgs)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss  += loss.item()
        all_logits.append(logits.detach())
        all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    f1 = compute_f1(all_logits, all_labels)
    return total_loss / len(loader), f1


@torch.no_grad()
def eval_finetune_epoch(
    model:     BirdMAEClassifier,
    loader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, all_logits, all_labels = 0.0, [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast():
            logits = model(imgs)
            loss   = criterion(logits, labels)
        total_loss  += loss.item()
        all_logits.append(logits)
        all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    preds      = all_logits.argmax(dim=-1).cpu().numpy()
    labs       = all_labels.cpu().numpy()
    f1         = f1_score(labs, preds, average="macro", zero_division=0)
    return total_loss / len(loader), f1, preds, labs


def run_finetune(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Fine-tuning on {device}")

    out_dir  = Path(args.output)
    plot_dir = out_dir / "plots"
    ckpt_dir = out_dir / "checkpoints"
    plot_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cfg = {}
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            cfg = json.load(f)

    train_loader, val_loader, test_loader, num_classes = make_finetune_loaders(
        args.manifest,
        stats_path  = args.stats,
        batch_size  = cfg.get("batch_size", 64),
        num_workers = cfg.get("num_workers", 0),
        balanced    = cfg.get("balanced_sampling", True),
    )

    # Rebuild MAE and load pretrained weights
    mae = build_mae(cfg)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        missing, unexpected = mae.load_state_dict(ckpt["model_state"], strict=False)
        log.info(f"Loaded pretrained weights from {args.checkpoint}")
        if missing:
            log.warning(f"  Missing keys ({len(missing)}): {missing[:5]}…")
    mae = mae.to(device)

    model = build_classifier(mae, num_classes, cfg).to(device)
    log.info(f"Classifier parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Label smoothing helps with multi-class Amazonian datasets
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.1))

    # Two-stage LR: low LR for backbone, higher for head
    backbone_lr = cfg.get("backbone_lr", 1e-6)
    head_lr     = cfg.get("head_lr", 5e-6)
    optimizer   = optim.AdamW([
        {"params": model.mae.parameters(), "lr": backbone_lr},
        {"params": model.head.parameters(), "lr": head_lr},
    ], weight_decay=cfg.get("weight_decay", 0.05))

    epochs    = cfg.get("finetune_epochs", 10)
    scheduler = warmup_cosine_scheduler(optimizer, warmup_epochs=3, total_epochs=epochs)
    scaler    = GradScaler()

    train_losses, val_losses = [], []
    train_f1s,    val_f1s   = [], []
    best_val_f1  = 0.0
    best_ckpt    = str(ckpt_dir / "best_finetune.pt")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_f1 = train_finetune_epoch(
            model, train_loader, criterion, optimizer, scaler, device)
        vl_loss, vl_f1, _, _ = eval_finetune_epoch(
            model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        train_f1s.append(tr_f1)
        val_f1s.append(vl_f1)

        lr = optimizer.param_groups[-1]["lr"]
        log.info(
            f"[Finetune] Epoch {epoch:03d}/{epochs}  "
            f"loss={tr_loss:.3f}/{vl_loss:.3f}  "
            f"F1={tr_f1:.3f}/{vl_f1:.3f}  "
            f"lr={lr:.2e}  t={time.time()-t0:.1f}s"
        )

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_f1": best_val_f1,
                "num_classes": num_classes,
                "cfg": cfg,
            }, best_ckpt)
            log.info(f"  ✓ New best F1={best_val_f1:.4f}  checkpoint saved")

        # Update plots every epoch
        plot_finetune_curves(
            train_losses, val_losses, train_f1s, val_f1s,
            save_path=str(plot_dir / "finetune_curves.png"),
        )

    # ── Test evaluation ───────────────────────────────────────────────────────
    log.info("\nRunning test evaluation…")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    _, test_f1, test_preds, test_labs = eval_finetune_epoch(
        model, test_loader, criterion, device)
    log.info(f"Test Macro F1: {test_f1:.4f}")

    # Retrieve class names from dataset
    from SeniorDesign.dataset import LabelledBirdDataset
    test_ds     = LabelledBirdDataset(args.manifest, args.stats, split="test")
    class_names = [test_ds.idx_to_class[i] for i in range(test_ds.num_classes)]

    # Per-class F1 report
    report = classification_report(
        test_labs, test_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    per_class_f1 = [report[c]["f1-score"] for c in class_names if c in report]
    plot_per_class_f1(
        class_names, per_class_f1,
        save_path=str(plot_dir / "per_class_f1.png"),
        title=f"Per-Class F1 — Test Set (Macro F1={test_f1:.3f})",
    )

    # Save text report
    report_path = out_dir / "test_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(classification_report(test_labs, test_preds,
                                      target_names=class_names, zero_division=0))
    log.info(f"Classification report → {report_path}")
    log.info("Fine-tuning complete.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="BirdMAE training script")
    sub = p.add_subparsers(dest="phase", required=True)

    # Pre-train
    pt = sub.add_parser("pretrain", help="MAE self-supervised pre-training")
    pt.add_argument("--manifest",   required=True, help="Unlabelled manifest JSON")
    pt.add_argument("--stats",      default=None,  help="Dataset stats JSON")
    pt.add_argument("--output",     required=True, help="Output directory")
    pt.add_argument("--config",     default=None,  help="JSON config file")

    # Fine-tune
    ft = sub.add_parser("finetune", help="Supervised fine-tuning")
    ft.add_argument("--manifest",    required=True, help="Labelled manifest JSON")
    ft.add_argument("--stats",       default=None,  help="Dataset stats JSON")
    ft.add_argument("--checkpoint",  default=None,  help="Pretrained MAE checkpoint")
    ft.add_argument("--output",      required=True, help="Output directory")
    ft.add_argument("--config",      default=None,  help="JSON config file")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.phase == "pretrain":
        run_pretrain(args)
    elif args.phase == "finetune":
        run_finetune(args)
