# Amazonian Bird MAE

A **Masked Autoencoder** built on a BirdNET-compatible EfficientNet backbone,
fine-tuned for Amazonian bird call classification.

---

## File overview

| File | Purpose |
|---|---|
| `preprocess.py` | Audio → mel spectrogram pipeline with augmentation |
| `dataset.py` | PyTorch Datasets + DataLoader factories |
| `model.py` | BirdMAE architecture + classifier head |
| `train.py` | Two-phase training: MAE pre-train + supervised fine-tune |
| `inference.py` | Single-file, batch, and saliency-map inference |
| `config.json` | Hyperparameter config |
| `requirements.txt` | Python dependencies |

---

## Quick-start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Prepare data

```bash
# Unlabelled Amazonian recordings (e.g. downloaded from xeno-canto)
python preprocess.py \
    --input_dir  /data/amazonian_audio \
    --output_dir /data/processed \
    --n_aug 2 \
    --stats

# For fine-tuning, your labelled manifest should look like:
# {
#   "samples": [{"path": "/data/processed/spectrograms/xyz.npy", "label": "Pipra_filicauda"}, ...],
#   "classes": ["Lepidothrix_coronata", "Pipra_filicauda", ...]
# }
```

### 3. Pre-train (MAE)

```bash
python train.py pretrain \
    --manifest /data/processed/manifest.json \
    --stats    /data/processed/stats.json \
    --output   runs/pretrain \
    --config   config.json
```

Plots are written to `runs/pretrain/plots/` after every epoch:
- `pretrain_loss.png` — train vs. val MSE loss curve
- `reconstruction_ep*.png` — visual reconstruction quality

### 4. Fine-tune (classification)

```bash
python train.py finetune \
    --manifest   /data/labelled_manifest.json \
    --stats      /data/processed/stats.json \
    --checkpoint runs/pretrain/checkpoints/best_model.pt \
    --output     runs/finetune \
    --config     config.json
```

Plots written to `runs/finetune/plots/`:
- `finetune_curves.png` — loss + macro F1 curves (train/val)
- `per_class_f1.png` — horizontal bar chart of per-species F1

### 5. Inference

```bash
# Classify one file
python inference.py \
    --checkpoint runs/finetune/checkpoints/best_finetune.pt \
    --classes    classes.json \
    --stats      /data/processed/stats.json \
    single \
    --audio  field_recording.mp3 \
    --output results.json

# Batch classify a directory
python inference.py \
    --checkpoint runs/finetune/checkpoints/best_finetune.pt \
    --classes    classes.json \
    batch \
    --input_dir /data/field_recordings \
    --output    predictions.csv

# Saliency map (which time-frequency regions drove the prediction)
python inference.py \
    --checkpoint runs/finetune/checkpoints/best_finetune.pt \
    --classes    classes.json \
    explain \
    --audio  field_recording.mp3 \
    --output saliency.png
```

---

## Architecture summary

```
Audio (48 kHz)
  └─▶ Mel Spectrogram (128 × 224)
        └─▶ Patchify (16 × 16 patches → 112 tokens)
              └─▶ BirdNET EfficientNet-B0 stem  (per-patch feature extraction)
                    └─▶ Sinusoidal 2-D pos. embedding
                          └─▶ Random masking (75 % of patches dropped)
                                └─▶ 6-layer Transformer encoder
                                      └─▶ 4-layer Transformer decoder
                                            └─▶ MSE loss on masked patches
                                                    (pre-training)
                                            OR
                                            └─▶ GAP → Linear head
                                                    (fine-tuning)
```

---

## Key design decisions

| Choice | Rationale |
|---|---|
| **EfficientNet-B0 stem as patch encoder** | Reuses BirdNET pretrained weights; adapts to 1-channel spectrograms via weight-averaging of RGB stem |
| **75 % mask ratio** | Forces the encoder to learn rich acoustic representations; higher than ViT defaults because spectrograms are sparser than natural images |
| **Time-frequency 2-D positional encoding** | Encodes both spectral (frequency axis) and temporal structure explicitly |
| **Two-stage LR in fine-tuning** | Low backbone LR (1e-5) + higher head LR (1e-4) prevents catastrophic forgetting of pretrained representations |
| **Label smoothing (0.1)** | Improves calibration on rare Amazonian species with few recordings |
| **Balanced sampling** | WeightedRandomSampler corrects the heavy class imbalance typical of biodiversity datasets |

---

## Data sources for Amazonian birds

- [xeno-canto](https://xeno-canto.org) — community recordings, filterable by country / region
- [Macaulay Library](https://www.macaulaylibrary.org) — Cornell Lab archive
- [WikiAves](https://www.wikiaves.com.br) — Brazilian species focus
- [INPA collections](http://www.inpa.gov.br) — Instituto Nacional de Pesquisas da Amazônia

---

## Citation

If you use this project, please cite:

```
He, K. et al. (2021). Masked Autoencoders Are Scalable Vision Learners.
Kahl, S. et al. (2021). BirdNET: A deep learning solution for avian diversity monitoring.
```
