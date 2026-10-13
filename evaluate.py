"""
Evaluates the trained model two ways:
  1. Point-prediction accuracy (MAE, RMSE) - is the estimate itself good?
  2. Conformal calibration (coverage, sharpness) - are the confidence
     intervals honest and useful?

Coverage should land close to your target (e.g. 90% intervals should
contain the true value ~90% of the time on the test set - not more, not
less). Sharpness (mean interval width) is the metric you try to minimize
*subject to* hitting your coverage target - a model that always predicts
"somewhere between 0 and 52 weeks" has perfect coverage and is useless.

Usage:
    python evaluate.py
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.olives_dataset import OlivesConfig, OlivesInjectionIntervalDataset, patient_level_split
from models.fusion_model import FusionRegressor
from uncertainty.conformal import ConformalizedQuantileRegressor, SplitConformalRegressor
from train import CSV_PATH, IMAGE_ROOT, BACKBONE, QUANTILE_MODE, CHECKPOINT_PATH, DEVICE, BATCH_SIZE

ALPHA = 0.1  # 1 - ALPHA = 90% target coverage


def get_predictions(model, loader):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(DEVICE)
            tabular = batch["tabular"].to(DEVICE)
            preds = model(image, tabular).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(batch["target_weeks"].numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


def point_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    return {"MAE_weeks": round(mae, 3), "RMSE_weeks": round(rmse, 3)}


def main():
    train_ids, val_ids, test_ids = patient_level_split(CSV_PATH)
    # Split the val set in half: one half for calibration, other half unused here
    # (already used for early stopping during training).
    n_calib = len(val_ids) // 2
    calib_ids, _ = val_ids[:n_calib], val_ids[n_calib:]

    calib_ds = OlivesInjectionIntervalDataset(
        OlivesConfig(csv_path=CSV_PATH, image_root=IMAGE_ROOT, train=False), patient_ids=calib_ids
    )
    test_ds = OlivesInjectionIntervalDataset(
        OlivesConfig(csv_path=CSV_PATH, image_root=IMAGE_ROOT, train=False), patient_ids=test_ids
    )
    calib_loader = DataLoader(calib_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = FusionRegressor(
        tabular_dim=calib_ds.tabular_dim, backbone_name=BACKBONE, quantile_mode=QUANTILE_MODE
    ).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))

    calib_preds, calib_targets = get_predictions(model, calib_loader)
    test_preds, test_targets = get_predictions(model, test_loader)

    if QUANTILE_MODE:
        # preds columns are [q_low, q_median, q_high]
        point_test = test_preds[:, 1]
        print("Point-prediction metrics (median head):")
        print(point_metrics(point_test, test_targets))

        cqr = ConformalizedQuantileRegressor(alpha=ALPHA)
        cqr.calibrate(calib_preds[:, 0], calib_preds[:, 2], calib_targets)
        result = cqr.predict(test_preds[:, 0], test_preds[:, 2], point=point_test)

    else:
        point_test = test_preds.squeeze(-1)
        print("Point-prediction metrics:")
        print(point_metrics(point_test, test_targets))

        scr = SplitConformalRegressor(alpha=ALPHA)
        scr.calibrate(calib_preds.squeeze(-1), calib_targets)
        result = scr.predict(point_test)

    print(f"\nTarget coverage: {1 - ALPHA:.0%}")
    print(f"Empirical coverage on test set: {result.coverage(test_targets):.1%}")
    print(f"Mean interval width: {result.mean_width():.2f} weeks")
    print(
        "\nInterpretation: coverage close to target = intervals are trustworthy. "
        "Narrower mean width at the same coverage = intervals are more useful. "
        "If coverage is far below target, the model is overconfident - do not "
        "deploy the point predictions alone as decision support."
    )


if __name__ == "__main__":
    main()
