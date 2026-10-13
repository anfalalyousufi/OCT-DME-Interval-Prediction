"""
Trains the OCT + tabular fusion model to predict weeks-until-next-visit.

Usage:
    python train.py

Edit DATA_ROOT / CSV_PATH below to point at your local OLIVES export before
running. See data/olives_dataset.py for the expected CSV schema.
"""

import os

import torch
from torch.utils.data import DataLoader

from data.olives_dataset import OlivesConfig, OlivesInjectionIntervalDataset, patient_level_split
from models.fusion_model import FusionRegressor, pinball_loss

# ---- Config: EDIT THESE ----------------------------------------------
CSV_PATH = "path/to/olives_visits.csv"
IMAGE_ROOT = "path/to/olives_images"
BACKBONE = "resnet18"          # "resnet18" | "resnet50" | "efficientnet_b0" | "vit_b_16"
QUANTILE_MODE = True           # True -> also trains q_low/q_high heads for CQR
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/fusion_model.pt"
# ------------------------------------------------------------------------


def build_loaders():
    train_ids, val_ids, test_ids = patient_level_split(CSV_PATH)

    train_ds = OlivesInjectionIntervalDataset(
        OlivesConfig(csv_path=CSV_PATH, image_root=IMAGE_ROOT, train=True), patient_ids=train_ids
    )
    val_ds = OlivesInjectionIntervalDataset(
        OlivesConfig(csv_path=CSV_PATH, image_root=IMAGE_ROOT, train=False), patient_ids=val_ids
    )
    test_ds = OlivesInjectionIntervalDataset(
        OlivesConfig(csv_path=CSV_PATH, image_root=IMAGE_ROOT, train=False), patient_ids=test_ids
    )

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
            image = batch["image"].to(DEVICE)
            tabular = batch["tabular"].to(DEVICE)
            target = batch["target_weeks"].to(DEVICE)

            preds = model(image, tabular)  # (batch, 3) if quantile_mode else (batch, 1)

            if QUANTILE_MODE:
                loss = pinball_loss(preds, target)
            else:
                loss = torch.nn.functional.l1_loss(preds.squeeze(-1), target)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * image.size(0)
            n += image.size(0)

    return total_loss / n


def main():
    train_ds, train_loader, val_loader, test_loader = build_loaders()

    model = FusionRegressor(
        tabular_dim=train_ds.tabular_dim,
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

    print(f"Training done. Best val_loss={best_val:.3f}. Run evaluate.py next.")


if __name__ == "__main__":
    main()
