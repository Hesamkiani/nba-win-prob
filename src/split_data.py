"""
Phase 3: Train/Test Split & PyTorch Data Preparation
-------------------------------------------------------
Splits the feature table into train/test sets by game_id (never by row,
to avoid data leakage), fits a StandardScaler on the train set only,
and saves everything needed for training (Phase 4).

Run:
    python src/split_data.py
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

TEST_SIZE = 0.2       # 20% of GAMES go to test (not rows)
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "score_diff",
    "seconds_remaining",
    "possession_home",
    "home_team_fouls",
    "away_team_fouls",
    "home_in_bonus",
    "away_in_bonus",
]
LABEL_COLUMN = "home_win"
GROUP_COLUMN = "game_id"


def load_features() -> pd.DataFrame:
    log.info(f"Loading feature table from {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)
    missing = [c for c in FEATURE_COLUMNS + [LABEL_COLUMN, GROUP_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Feature table is missing expected columns: {missing}")
    log.info(f"Loaded {len(df):,} rows across {df[GROUP_COLUMN].nunique():,} games")
    return df


def split_by_game(df: pd.DataFrame):
    """Group-aware split: an entire game's rows go to either train or test,
    never both. This is the #1 leakage trap in this kind of project."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, groups=df[GROUP_COLUMN]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Sanity check: zero overlap in game_id between train and test
    overlap = set(train_df[GROUP_COLUMN]) & set(test_df[GROUP_COLUMN])
    if overlap:
        raise RuntimeError(f"Leakage detected! {len(overlap)} games appear in both splits.")

    log.info(
        f"Train: {len(train_df):,} rows / {train_df[GROUP_COLUMN].nunique():,} games | "
        f"Test: {len(test_df):,} rows / {test_df[GROUP_COLUMN].nunique():,} games | "
        f"Overlap check passed (0 shared games)"
    )
    return train_df, test_df


def scale_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Fit the scaler on TRAIN ONLY, then apply to both. Fitting on the
    full dataset (including test) would leak test-set statistics into
    training — a subtler but still real form of leakage."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURE_COLUMNS].values.astype(np.float32))
    X_test = scaler.transform(test_df[FEATURE_COLUMNS].values.astype(np.float32))

    y_train = train_df[LABEL_COLUMN].values.astype(np.float32)
    y_test = test_df[LABEL_COLUMN].values.astype(np.float32)

    return X_train, X_test, y_train, y_test, scaler


def save_tensors(X_train, X_test, y_train, y_test, scaler):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(torch.from_numpy(X_train), PROCESSED_DIR / "X_train.pt")
    torch.save(torch.from_numpy(X_test), PROCESSED_DIR / "X_test.pt")
    torch.save(torch.from_numpy(y_train), PROCESSED_DIR / "y_train.pt")
    torch.save(torch.from_numpy(y_test), PROCESSED_DIR / "y_test.pt")

    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(FEATURE_COLUMNS, MODELS_DIR / "feature_columns.joblib")

    log.info(f"Saved tensors to {PROCESSED_DIR}")
    log.info(f"Saved scaler + feature column order to {MODELS_DIR}")


if __name__ == "__main__":
    df = load_features()
    train_df, test_df = split_by_game(df)
    X_train, X_test, y_train, y_test, scaler = scale_features(train_df, test_df)

    log.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    log.info(f"X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")
    log.info(f"Feature columns (order matters!): {FEATURE_COLUMNS}")

    save_tensors(X_train, X_test, y_train, y_test, scaler)
    log.info("Phase 3 complete.")