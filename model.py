"""
model.py
--------
BirdMAE — Masked Autoencoder built on a BirdNET-compatible EfficientNet backbone.

"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ─────────────────────────────────────────────────────────────────────────────
# Positional encoding
# ─────────────────────────────────────────────────────────────────────────────
# This is important for the to have a spacial understanding of the spectrogram

def sinusoidal_2d_pos_embed(embed_dim: int, grid_h: int, grid_w: int) -> torch.Tensor:
    """2-D sinusoidal pos. embedding: (grid_h * grid_w, embed_dim)."""
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"
    half = embed_dim // 2

    def _1d(length, d):
        pos   = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        omega = 1.0 / (10000 ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
        enc   = torch.zeros(length, d)
        enc[:, 0::2] = torch.sin(pos * omega)
        enc[:, 1::2] = torch.cos(pos * omega)
        return enc

    row = _1d(grid_h, half).unsqueeze(1).expand(-1, grid_w, -1)
    col = _1d(grid_w, half).unsqueeze(0).expand(grid_h, -1, -1)
    return torch.cat([row, col], dim=-1).reshape(grid_h * grid_w, embed_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Patchify / Unpatchify
# ─────────────────────────────────────────────────────────────────────────────
# Patchify turns the spectrogram into non-overlapping patches, which are the input tokens for the transformer. 
# Unpatchify reconstructs the spectrogram from the predicted patches during decoding.

def patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B, 1, H, W) → (B, num_patches, patch_size²)"""
    B, C, H, W = imgs.shape
    h, w = H // patch_size, W // patch_size
    x = imgs.reshape(B, C, h, patch_size, w, patch_size)
    x = x.permute(0, 2, 4, 3, 5, 1)
    return x.reshape(B, h * w, patch_size * patch_size * C)


def unpatchify(patches: torch.Tensor, patch_size: int, H: int, W: int) -> torch.Tensor:
    """(B, num_patches, patch_size²) → (B, 1, H, W)"""
    B = patches.shape[0]
    h, w = H // patch_size, W // patch_size
    x = patches.reshape(B, h, w, patch_size, patch_size, 1)
    x = x.permute(0, 5, 1, 3, 2, 4)
    return x.reshape(B, 1, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# ConvStem  (processes the FULL spectrogram — fixes the BatchNorm crash)
# ─────────────────────────────────────────────────────────────────────────────
# I am using batchnorm in the conv stem, but it was crashing when the batch size was 1 during fine-tuning.
# This ConvStem processes the entire spectrogram at once, which avoids this issue.

class ConvStem(nn.Module):
    """
    Runs EfficientNet-B0 stages 0-1 (stride 8 total) over the entire
    spectrogram, then AdaptiveAvgPool2d maps the feature map to
    (grid_h, grid_w) so each cell corresponds to exactly one patch token.

    BatchNorm always sees the full spatial map — no 1×1 collapse, no crash.
    Pretrained BirdNET weights are preserved; only the first conv is adapted
    from 3-channel RGB to 1-channel by averaging across the channel dim.
    """

    def __init__(
        self,
        img_h:      int  = 128,
        img_w:      int  = 224,
        patch_size: int  = 16,
        embed_dim:  int  = 256,
        freeze:     bool = False,
    ):
        super().__init__()
        self.grid_h = img_h // patch_size
        self.grid_w = img_w // patch_size

        # EfficientNet-B0 up to stage-1 (stride-8, 24 output channels)
        backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            features_only=True,
            out_indices=(1,),
        )

        # Adapt stem conv: 3-channel → 1-channel
        old = backbone.conv_stem
        new_conv = nn.Conv2d(
            1, old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        backbone.conv_stem = new_conv

        self.backbone = backbone
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        # Probe output channels — use batch=2 so BatchNorm doesn't crash here
        with torch.no_grad():
            dummy       = torch.zeros(2, 1, img_h, img_w)
            out_channels = self.backbone(dummy)[0].shape[1]

        self.pool = nn.AdaptiveAvgPool2d((self.grid_h, self.grid_w))
        self.proj = nn.Conv2d(out_channels, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs   : (B, 1, H, W)
        returns: (B, grid_h * grid_w, embed_dim)
        """
        feat = self.backbone(imgs)[0]          # (B, C, h', w')
        feat = self.pool(feat)                 # (B, C, grid_h, grid_w)
        feat = self.proj(feat)                 # (B, embed_dim, grid_h, grid_w)
        B, D, gh, gw = feat.shape
        tokens = feat.permute(0, 2, 3, 1).reshape(B, gh * gw, D)
        return self.norm(tokens)               # (B, num_patches, embed_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer block
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim    = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


# ─────────────────────────────────────────────────────────────────────────────
# BirdMAE
# ─────────────────────────────────────────────────────────────────────────────

class BirdMAE(nn.Module):
    """
    Masked Autoencoder for Amazonian bird call spectrograms.

    Parameters
    ----------
    img_h, img_w    : spectrogram size in pixels  (default 128 × 224)
    patch_size      : grid cell size               (default 16)
    embed_dim       : encoder token dimension
    enc_depth       : transformer blocks in encoder
    dec_depth       : transformer blocks in decoder
    dec_dim         : decoder token dimension (< embed_dim)
    n_heads_enc/dec : attention heads
    mask_ratio      : fraction of patches masked during pre-training
    freeze_backbone : freeze ConvStem weights
    """

    def __init__(
        self,
        img_h:           int   = 128,
        img_w:           int   = 224,
        patch_size:      int   = 16,
        embed_dim:       int   = 256,
        enc_depth:       int   = 6,
        dec_depth:       int   = 4,
        dec_dim:         int   = 128,
        n_heads_enc:     int   = 8,
        n_heads_dec:     int   = 4,
        mask_ratio:      float = 0.75,
        freeze_backbone: bool  = False,
    ):
        super().__init__()
        assert img_h % patch_size == 0 and img_w % patch_size == 0, (
            f"img_h={img_h} and img_w={img_w} must both be divisible by patch_size={patch_size}"
        )

        self.img_h       = img_h
        self.img_w       = img_w
        self.patch_size  = patch_size
        self.mask_ratio  = mask_ratio
        self.embed_dim   = embed_dim
        self.grid_h      = img_h // patch_size
        self.grid_w      = img_w // patch_size
        self.num_patches = self.grid_h * self.grid_w

        # ── Encoder ──────────────────────────────────────────────────────────
        self.conv_stem = ConvStem(img_h, img_w, patch_size, embed_dim, freeze_backbone)

        pos = sinusoidal_2d_pos_embed(embed_dim, self.grid_h, self.grid_w)
        self.register_buffer("pos_embed", pos)

        self.enc_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads_enc) for _ in range(enc_depth)
        ])
        self.enc_norm = nn.LayerNorm(embed_dim)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.enc_to_dec = nn.Linear(embed_dim, dec_dim, bias=False)

        pos_dec = sinusoidal_2d_pos_embed(dec_dim, self.grid_h, self.grid_w)
        self.register_buffer("pos_embed_dec", pos_dec)

        self.dec_blocks = nn.ModuleList([
            TransformerBlock(dec_dim, n_heads_dec) for _ in range(dec_depth)
        ])
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.dec_head = nn.Linear(dec_dim, patch_size * patch_size)

    # ── Masking ───────────────────────────────────────────────────────────────

    def random_masking(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, D  = x.shape
        len_keep = int(L * (1 - self.mask_ratio))
        noise    = torch.rand(B, L, device=x.device)
        ids_shuf = torch.argsort(noise, dim=1)
        ids_rest = torch.argsort(ids_shuf, dim=1)
        ids_keep = ids_shuf[:, :len_keep]
        x_vis    = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask     = torch.ones(B, L, device=x.device)
        mask[:, :len_keep] = 0
        mask     = torch.gather(mask, 1, ids_rest)
        return x_vis, mask, ids_rest

    # ── Encode ────────────────────────────────────────────────────────────────

    def encode(
        self, imgs: torch.Tensor, mask: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L = imgs.shape[0], self.num_patches
        tokens = self.conv_stem(imgs) + self.pos_embed.unsqueeze(0)

        if mask:
            tokens, mask_bin, ids_restore = self.random_masking(tokens)
        else:
            mask_bin    = torch.zeros(B, L, device=imgs.device)
            ids_restore = torch.arange(L, device=imgs.device).unsqueeze(0).expand(B, -1)

        for blk in self.enc_blocks:
            tokens = blk(tokens)
        return self.enc_norm(tokens), mask_bin, ids_restore

    # ── Decode ────────────────────────────────────────────────────────────────

    def decode(self, latent: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        B, L    = latent.shape[0], self.num_patches
        len_vis = latent.shape[1]

        latent      = self.enc_to_dec(latent)
        mask_tokens = self.mask_token.expand(B, L - len_vis, -1)
        full        = torch.cat([latent, mask_tokens], dim=1)
        idx         = ids_restore.unsqueeze(-1).expand(-1, -1, full.shape[-1])
        full        = torch.gather(full, 1, idx)
        full        = full + self.pos_embed_dec.unsqueeze(0)

        for blk in self.dec_blocks:
            full = blk(full)
        return self.dec_head(self.dec_norm(full))   # (B, L, P²)

    # ── Loss ──────────────────────────────────────────────────────────────────

    def reconstruction_loss(
        self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        target = patchify(imgs, self.patch_size)
        loss   = F.mse_loss(pred, target, reduction="none").mean(-1)  # (B, L)
        return (loss * mask).sum() / mask.sum()

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, mask, ids_restore = self.encode(imgs, mask=True)
        pred = self.decode(latent, ids_restore)
        loss = self.reconstruction_loss(imgs, pred, mask)
        return pred, mask, loss

    # ── Reconstruction visualisation ──────────────────────────────────────────

    @torch.no_grad()
    def reconstruct(self, imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pred, mask, _ = self.forward(imgs)
        P = self.patch_size
        rec_img        = unpatchify(pred, P, self.img_h, self.img_w)
        mask_vis       = mask.unsqueeze(-1).expand(-1, -1, P * P)
        orig_patches   = patchify(imgs, P)
        masked_patches = orig_patches * (1 - mask_vis) + 0.5 * mask_vis
        masked_img     = unpatchify(masked_patches, P, self.img_h, self.img_w)
        return rec_img, masked_img


# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning head
# ─────────────────────────────────────────────────────────────────────────────

class BirdMAEClassifier(nn.Module):
    """GAP over encoder tokens → linear classifier."""

    def __init__(self, mae: BirdMAE, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.mae  = mae
        self.head = nn.Sequential(
            nn.LayerNorm(mae.embed_dim),
            nn.Dropout(dropout),
            nn.Linear(mae.embed_dim, num_classes),
        )

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        latent, _, _ = self.mae.encode(imgs, mask=False)
        return self.head(latent.mean(dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_mae(cfg: dict) -> BirdMAE:
    return BirdMAE(
        img_h           = cfg.get("img_h",           128),
        img_w           = cfg.get("img_w",           224),
        patch_size      = cfg.get("patch_size",       16),
        embed_dim       = cfg.get("embed_dim",        256),
        enc_depth       = cfg.get("enc_depth",          6),
        dec_depth       = cfg.get("dec_depth",          4),
        dec_dim         = cfg.get("dec_dim",          128),
        n_heads_enc     = cfg.get("n_heads_enc",        8),
        n_heads_dec     = cfg.get("n_heads_dec",        4),
        mask_ratio      = cfg.get("mask_ratio",      0.75),
        freeze_backbone = cfg.get("freeze_backbone", False),
    )


def build_classifier(mae: BirdMAE, num_classes: int, cfg: dict) -> BirdMAEClassifier:
    return BirdMAEClassifier(mae, num_classes, dropout=cfg.get("dropout", 0.3))