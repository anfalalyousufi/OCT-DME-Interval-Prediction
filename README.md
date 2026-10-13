# OLIVES DME — Uncertainty-Aware Injection Interval Prediction

Predicts how many weeks until a DME patient's next anti-VEGF injection is
needed (treat-and-extend "wait vs. inject" decision), from OCT + clinical/
biomarker data, with **calibrated prediction intervals** via conformal
prediction — not just a point estimate.

## Why this exists

Treat-and-extend regimens (as validated in the TREX-DME trials) decide, at
every visit, whether to shorten or extend the interval to the next
injection based on OCT findings. A model that outputs "6 weeks" with no
sense of its own confidence is not safe to use as decision support. This
scaffold instead outputs, e.g., "6 weeks, 90% confidence interval [4, 9]" —
narrow when the model is confident, wide when the case is ambiguous.

## Project layout

```
olives_dme_uncertainty/
├── data/
│   └── olives_dataset.py   # PyTorch Dataset: builds (OCT, tabular, target) per visit
├── models/
│   └── fusion_model.py     # CNN/EfficientNet backbone + tabular MLP, fused regression head
├── uncertainty/
│   └── conformal.py        # Split conformal + conformalized quantile regression (CQR)
├── train.py                 # Training loop, patient-level split
├── evaluate.py               # MAE/RMSE + coverage + sharpness metrics
└── requirements.txt
```

## IMPORTANT — you must adapt the data loader to your actual OLIVES export

OLIVES is distributed as OCT volumes/B-scans + CSVs of per-visit clinical
labels (BCVA, CST/CRT, biomarker flags), keyed by patient/eye ID and visit
week. Exact column names differ slightly between the released CSV and the
Hugging Face mirror, and some patients have missing visits. `olives_dataset.py`
is written against a **reasonable, clearly-documented schema** — open the
`REQUIRED COLUMNS` block at the top of that file and rename/remap to match
whatever CSV you actually download from
https://github.com/olivesgatech/OLIVES_Dataset before running anything.

## Suggested first run

1. `pip install -r requirements.txt`
2. Point `DATA_ROOT` / `CSV_PATH` in `train.py` at your local OLIVES copy.
3. `python train.py` — trains the fusion model, saves checkpoint + validation
   predictions.
4. `python evaluate.py` — calibrates the conformal wrapper on a held-out
   calibration split, evaluates coverage/sharpness on the test split.

## 3D / volume-level variant (new)

The original pipeline above (`train.py`) uses one B-scan per visit. OLIVES
is actually distributed as full volumes (~49 B-scans/visit), so there's a
parallel volume-level pipeline that uses all of them:

- `data/olives_volume_dataset.py` — loads the whole B-scan sequence per
  visit, resampled to a fixed depth so volumes of different lengths batch
  together.
- `models/volume_encoder.py` — two interchangeable encoders:
  - `Slice2DAggregator`: runs a shared 2D pretrained CNN on every slice,
    then combines them with learned attention pooling (a small
    multiple-instance-learning head). Cheap, reuses ImageNet weights, and
    the attention weights double as a free interpretability signal —
    which B-scans the model actually looked at.
  - `Conv3DEncoder`: a real 3D CNN (Conv3d/BatchNorm3d/MaxPool3d blocks)
    convolving across x, y, and depth jointly. No pretrained weights exist
    at this scale for OCT, so it needs more data/epochs than the
    slice-aggregation model to converge.
- `models/fusion_model_3d.py` — same tabular fusion pattern as before, now
  fused with a volume feature instead of a single-image feature.
- `train_3d.py` — same training loop shape as `train.py`, with much
  smaller default `BATCH_SIZE`/`IMAGE_SIZE` since a batch of full volumes
  is far more memory-hungry than a batch of single images.

**Start with `encoder_type="slice2d"`** — it's the one nearly every
published DME-treatment paper actually uses in practice (full 3D CNNs are
rare in this literature specifically because no 3D-pretrained backbone
exists at ImageNet scale). Only try `"conv3d"` once the slice-aggregation
pipeline is working and you want to test whether true volumetric
convolution beats it on your data — that comparison is itself a valid
ablation study for a report/paper.

`evaluate.py`'s conformal-calibration logic (split conformal / CQR) is
model-agnostic and works unchanged with `VolumeFusionRegressor` — just
swap the model class and dataset/loader when adapting it.

## What to change as you make this your own project

- **Target definition**: currently regresses "weeks until next visit/injection".
  Swap to a binary "inject now vs extend" head in `fusion_model.py` if you'd
  rather frame it as classification (see the docstring there for the change).
- **Backbone**: swapped easily — `models/fusion_model.py` takes any
  `torchvision` CNN or a ViT; start with ResNet-18 for speed, move to
  EfficientNet/ViT once the pipeline works end-to-end (this mirrors the
  backbones benchmarked in the original OLIVES paper).
- **Uncertainty method**: `uncertainty/conformal.py` ships split conformal
  (simple, fixed-width intervals) and CQR (quantile-based, adapts width per
  sample). Start with split conformal to sanity check the pipeline, then
  switch to CQR for the "harder cases get wider intervals" behavior that's
  the actual point of the project.
- **Patient-level splitting is non-negotiable**: OLIVES has ~7 visits per
  eye. Splitting by visit instead of by patient leaks information and will
  make your numbers look better than they are. `train.py` already splits by
  `patient_id` — don't change that without a good reason.
