"""
Trains the volume-based fusion model (full OCT volume + tabular ->
weeks-until-next-visit). This is the 3D counterpart of train.py.

Usage:
    python train_3d.py

Edit the config block below before running. Note the smaller BATCH_SIZE
and IMAGE_SIZE compared to train.py — volumes are memory-heavy
(batch, depth, 3, H, W all multiply together), so start conservative and
scale up once you've confirmed it fits on your GPU.
"""

import os

import torch
from torch.utils.data import DataLoader

from data.olives_dataset import patient_level_split
from data.olives_volume_dataset import OlivesVolumeDataset
from models.fusion_model_3d import VolumeFusionRegressor, pinball_loss

# ---- Config: EDIT THESE ----------------------------------------------
CSV_PATH = "path/to/olives_visits.csv"       # must include an 'oct_dir' column
VOLUME_ROOT = "path/to/olives_volumes"
ENCODER_TYPE = "slice2d"          # "slice2d" (start here) | "conv3d"
BACKBONE = "resnet18"             # only used when ENCODER_TYPE == "slice2d"
TARGET_DEPTH = 32                 # resampled number of B-scans per volume
IMAGE_SIZE = 128                  # smaller than the 2D pipeline's 224 - volumes are memory-heavy
QUANTILE_MODE = True
BATCH_SIZE = 4                    # start small; volumes are much heavier than single images
EPOCHS = 30
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/fusion_model_3d.pt"
# ------------------------------------------------------------------------


def build_loaders():
    train_ids, val_ids, test_ids = patient_level_split(CSV_PATH)

    common = dict(csv_path=CSV_PATH, volume_root=VOLUME_ROOT, target_depth=TARGET_DEPTH, image_size=IMAGE_SIZE)
    train_ds = OlivesVolumeDataset(patient_ids=train_ids, **common)
    val_ds = OlivesVolumeDataset(patient_ids=val_ids, **common)
    test_ds = OlivesVolumeDataset(patient_ids=test_ids, **common)

    print(f"Patients -> train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")
    print(f"Visit-pairs -> train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    return train_ds, train_loader, val_loader, test_loader


def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, n = 0.0, 0

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            volume = batch["volume"].to(DEVICE)      # (B, D, 3, H, W)
            tabular = batch["tabular"].to(DEVICE)
            target = batch["target_weeks"].to(DEVICE)

            preds = model(volume, tabular)

            if QUANTILE_MODE:
                loss = pinball_loss(preds, target)
            else:
                loss = torch.nn.functional.l1_loss(preds.squeeze(-1), target)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * volume.size(0)
            n += volume.size(0)

    return total_loss / n


def main():
    train_ds, train_loader, val_loader, test_loader = build_loaders()

    model = VolumeFusionRegressor(
        tabular_dim=train_ds.tabular_dim,
        encoder_type=ENCODER_TYPE,
        backbone_name=BACKBONE,
        pretrained=True,
        quantile_mode=QUANTILE_MODE,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val = float("inf")
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer)
        val_loss = run_epoch(model, val_loader, optimizer=None)
        scheduler.step()

        print(f"epoch {epoch:03d} | train_loss {train_loss:.3f} | val_loss {val_loss:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> saved new best checkpoint (val_loss={val_loss:.3f})")

    print(f"Training done. Best val_loss={best_val:.3f}.")
    print("Reuse evaluate.py's conformal-calibration logic (swap in VolumeFusionRegressor) for coverage/sharpness.")


if __name__ == "__main__":
    main()
