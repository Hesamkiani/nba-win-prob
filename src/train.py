"""
Phase 4: Model Training (PyTorch)
-----------------------------------
Trains a small feedforward neural net (WinProbNet) on the train tensors
built in Phase 3, evaluates accuracy / AUC / Brier score / calibration,
and saves the trained weights for later use in live inference (Phase 5).

Run:
    python src/train.py
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

EPOCHS = 60
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
VAL_FRACTION = 0.1     # slice out of the TRAIN set for validation during training
RANDOM_STATE = 42
EARLY_STOP_PATIENCE = 8   # stop if val loss doesn't improve for this many epochs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class WinProbNet(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),  # raw logit — sigmoid applied at inference, not here
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_tensors():
    X_train_full = torch.load(PROCESSED_DIR / "X_train.pt")
    y_train_full = torch.load(PROCESSED_DIR / "y_train.pt")
    X_test = torch.load(PROCESSED_DIR / "X_test.pt")
    y_test = torch.load(PROCESSED_DIR / "y_test.pt")
    log.info(f"Loaded tensors — train: {X_train_full.shape}, test: {X_test.shape}")
    return X_train_full, y_train_full, X_test, y_test


def carve_out_validation(X_train_full, y_train_full):
    """Hold out a slice of the TRAIN set for validation during training.
    This is a random row-level split, which is fine here — leakage across
    train/val doesn't matter the way it does across train/test, since val
    is only used to watch for overfitting, not for the final reported metrics."""
    n = X_train_full.shape[0]
    g = torch.Generator().manual_seed(RANDOM_STATE)
    perm = torch.randperm(n, generator=g)
    n_val = int(n * VAL_FRACTION)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return (
        X_train_full[train_idx], y_train_full[train_idx],
        X_train_full[val_idx], y_train_full[val_idx],
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_model(X_train, y_train, X_val, y_val):
    n_features = X_train.shape[1]
    model = WinProbNet(n_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    X_val_dev = X_val.to(DEVICE)
    y_val_dev = y_val.to(DEVICE)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(train_ds)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_dev)
            val_loss = loss_fn(val_logits, y_val_dev).item()

        log.info(f"Epoch {epoch:3d}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                log.info(f"Early stopping at epoch {epoch} (no val improvement for {EARLY_STOP_PATIENCE} epochs)")
                break

    model.load_state_dict(best_state)
    log.info(f"Restored best model weights (val_loss={best_val_loss:.4f})")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(X_test.to(DEVICE))
        probs = torch.sigmoid(logits).cpu().numpy()

    y_true = y_test.numpy()
    preds = (probs >= 0.5).astype(int)

    accuracy = (preds == y_true).mean()
    auc = roc_auc_score(y_true, probs)
    brier = brier_score_loss(y_true, probs)

    frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10)

    log.info(f"Test Accuracy: {accuracy:.4f}")
    log.info(f"Test AUC:      {auc:.4f}")
    log.info(f"Brier Score:   {brier:.4f}  (lower is better; 0 = perfect, 0.25 = coin flip)")
    log.info("Reliability curve (predicted bin -> actual win rate):")
    for mp, fp in zip(mean_pred, frac_pos):
        log.info(f"    predicted ~{mp:.2f}  ->  actual {fp:.2f}")

    return {"accuracy": accuracy, "auc": auc, "brier": brier}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log.info(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    X_train_full, y_train_full, X_test, y_test = load_tensors()
    X_train, y_train, X_val, y_val = carve_out_validation(X_train_full, y_train_full)
    log.info(f"Train: {X_train.shape[0]:,} rows | Val: {X_val.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows")

    model = train_model(X_train, y_train, X_val, y_val)
    metrics = evaluate(model, X_test, y_test)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "win_prob_net.pt")
    joblib.dump(metrics, MODELS_DIR / "test_metrics.joblib")
    log.info(f"Saved trained model -> {MODELS_DIR / 'win_prob_net.pt'}")
    log.info("Phase 4 complete.")