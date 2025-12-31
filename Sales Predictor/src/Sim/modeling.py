"""
modeling.py

This module handles:

    - Recency-weighted hyperparameter tuning (λ tuning)
    - SKU density classification (dense vs sparse)
    - Training a single global XGBoost model
    - Predicting weekly sales using dense vs sparse rules
    - Predicting weekly sales using dense/sparse rules

Option C improvements:
    • More robust WAPE-based λ tuning
    • Require minimum sample size for stability
    • Tier-specific training with safe fallbacks
    • Log1p targets for smoother variance
"""

import os
import re
import numpy as np
import pandas as pd
from joblib import dump
from xgboost import XGBRegressor
from datetime import datetime

from .config import KEEP_ALL_MODELS, MODEL_RETENTION_MONTHS, WEEKLY_MODEL_DIR
from .utils import notify

from .config import (
    RECENCY_LAM_DEFAULT,
    RECENCY_LAM_CANDIDATES,
    SKUDENSE_AVG_THRESHOLD,
    SKUDENSE_ZERO_RATE_THRESHOLD,
    HIGH_VOLUME_THRESHOLD,
    HIGH_VOLUME_LAM_BOOST,
    RECENCY_LAM_POST_BIAS,
    RECENCY_LAM_MAX,
)
from .utils import notify


# ------------------------------------------------------------------------------
# SKU DENSITY CLASSIFICATION
# ------------------------------------------------------------------------------
def compute_sku_density_flags(
    weekly: pd.DataFrame,
    sim_start: pd.Timestamp,
    avg_threshold=SKUDENSE_AVG_THRESHOLD,
    zero_rate_threshold=SKUDENSE_ZERO_RATE_THRESHOLD,
) -> dict:
    """
    Dense SKU if:
        • avg weekly >= avg_threshold
        • zero-rate <= zero_rate_threshold
    """
    notify("[DENSITY] Computing SKU density flags...")

    hist = weekly[weekly["Date"] < sim_start]
    if hist.empty:
        notify("[DENSITY] No history before sim_start — all SKUs treated as sparse.")
        return {}

    stats = (
        hist.groupby(["Cadena", "Tienda", "Producto"])
            .agg(
                total_sales=("WeeklyTotal", "sum"),
                weeks=("Week", "nunique"),
                nonzero=("WeeklyTotal", lambda s: (s > 0).sum()),
            )
            .reset_index()
    )

    stats["avg_weekly"] = stats["total_sales"] / stats["weeks"].clip(lower=1)
    stats["zero_rate"] = 1.0 - stats["nonzero"] / stats["weeks"].clip(lower=1)

    dense_mask = (
        (stats["avg_weekly"] >= avg_threshold)
        & (stats["zero_rate"] <= zero_rate_threshold)
    )

    result = {}
    dense_count = 0

    for idx, row in stats.iterrows():
        key = (int(row["Cadena"]), int(row["Tienda"]), int(row["Producto"]))
        is_dense = bool(dense_mask.loc[idx])
        result[key] = is_dense
        dense_count += int(is_dense)

    notify(f"[DENSITY] Dense SKUs: {dense_count} / {len(stats)}")
    return result

# ------------------------------------------------------------------------------
# TRAINING HELPERS
# ------------------------------------------------------------------------------
def _fit_xgb_with_lambda(train_df, feature_cols, cutoff_date, lam):
    # --------------------------------------------------
    # STRICT TRAINING FILTER (CRITICAL FIX)
    # --------------------------------------------------
    df = train_df.copy()

    # Remove rows with invalid target
    df = df[np.isfinite(df["WeeklyTotal"])]
    df = df[df["WeeklyTotal"] >= 0]

    # Remove rows with missing features (lags / rollings)
    required_cols = [c for c in feature_cols if c != "lag_52w"]
    df = df.dropna(subset=required_cols)

    if len(df) < 200:
        return None

    X = df[feature_cols].astype(float).to_numpy()
    y = df["WeeklyTotal"].astype(float).to_numpy()

    # Safe log transform
    y_log = np.log1p(y)

    # --------------------------------------------------
    # RECENCY WEIGHTS
    # --------------------------------------------------
    weeks_diff = (cutoff_date - df["Date"]).dt.days / 7.0
    weeks_diff = weeks_diff.clip(lower=0.0).to_numpy(dtype=float)

    avg_nonzero = df.get("avg_nonzero_12w", 0.0)
    roll12 = df.get("roll12w", 0.0)

    avg_nonzero = np.asarray(avg_nonzero, dtype=float)
    roll12 = np.asarray(roll12, dtype=float)

    high_volume_mask = (
        (avg_nonzero > HIGH_VOLUME_THRESHOLD)
        | (roll12 > HIGH_VOLUME_THRESHOLD)
    )

    lam_effective = lam * np.where(
        high_volume_mask,
        1.0 + HIGH_VOLUME_LAM_BOOST,
        1.0,
    )

    sample_weight = np.exp(-lam_effective * weeks_diff)

    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------
    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=1.0,
        reg_lambda=5.0,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X, y_log, sample_weight=sample_weight)
    # In-sample fitted values (for residual tracking only)
    y_log_hat = model.predict(X)
    y_hat = np.expm1(y_log_hat).clip(0, None)

    fitted_df = df[[
        "Date", "Cadena", "Tienda", "Producto", "WeeklyTotal"
    ]].copy()

    fitted_df["y_hat"] = y_hat

    return model, fitted_df

def _fit_xgb(df, feature_cols, cutoff_date, lam):
    return _fit_xgb_with_lambda(df, feature_cols, cutoff_date, lam)


# ------------------------------------------------------------------------------
# RECENCY-LAMBDA TUNING
# ------------------------------------------------------------------------------
def tune_recency_lambda(weekly: pd.DataFrame, feature_cols, sim_start: pd.Timestamp) -> float:
    """
    Try several lambda candidates, evaluate on last 52 weeks, pick best.
    """
    notify("[TUNE] Starting recency-lambda tuning...")

    hist = weekly[
        (weekly["Date"] < sim_start) &
        (weekly["has_actual"])
    ]
    if len(hist) < 500:
        notify("[TUNE] Not enough data — using default λ.")
        return RECENCY_LAM_DEFAULT

    cal_start = sim_start - pd.Timedelta(weeks=52)
    calib = hist[hist["Date"] >= cal_start]
    if len(calib) < 200:
        notify("[TUNE] Calibration window too small — using default λ.")
        return RECENCY_LAM_DEFAULT

    best_lam = None
    best_wape = None

    for lam in RECENCY_LAM_CANDIDATES:
        notify(f"[TUNE] Testing λ={lam}...")
        res = _fit_xgb(hist, feature_cols, sim_start, lam)
        if res is None:
            continue

        model, _ = res  # ignore fitted_df during lambda tuning

        X = calib[feature_cols].astype(float).to_numpy()
        y = calib["WeeklyTotal"].clip(lower=0).to_numpy()
        if y.sum() <= 0:
            continue

        pred = np.expm1(model.predict(X)).clip(0, None)
        wape = np.sum(np.abs(y - pred)) / np.sum(y)

        notify(f"[TUNE] λ={lam} → WAPE={wape:.4f}")

        if best_wape is None or wape < best_wape:
            best_wape = wape
            best_lam = lam

    if best_lam is None:
        notify("[TUNE] All lambda candidates failed; using default.")
        lam_final = RECENCY_LAM_DEFAULT
    else:
        # Slightly bias lambda upwards (stronger recency), with a cap
        lam_biased = float(best_lam) * RECENCY_LAM_POST_BIAS
        lam_final = min(lam_biased, RECENCY_LAM_MAX)
        notify(
            f"[TUNE] Selected base lambda={best_lam:.4f}, "
            f"biased={lam_final:.4f} (WAPE={best_wape:.4f})"
        )

    return lam_final

# ------------------------------------------------------------------------------
# TRAIN MODELS FOR A GIVEN CUTOFF DATE
# ------------------------------------------------------------------------------
def train_models_for_cutoff(
    weekly: pd.DataFrame,
    feature_cols: list,
    cutoff_date: pd.Timestamp,
    recency_lambda: float,
    model_dir: str,
) -> dict:
    """
    Trains a single global XGBoost model for a given cutoff date.
    """
    notify(f"[MODEL] Training model for cutoff={cutoff_date.date()}...")

    df = weekly[
        (weekly["Date"] < cutoff_date) &
        (weekly["has_actual"])
    ]

    if df.empty:
        notify("[MODEL] No training data for cutoff.")
        return {"model": None, "fitted_global": None}

    res = _fit_xgb(df, feature_cols, cutoff_date, recency_lambda)
    if res is None:
        notify("[MODEL] Insufficient data after filtering.")
        return {"model": None, "fitted_global": None}

    global_model, fitted_global = res

    # Save model
    month_label = (cutoff_date - pd.Timedelta(days=1)).strftime("%Y%m")
    os.makedirs(model_dir, exist_ok=True)

    path = f"{model_dir}/model_global_{month_label}.pkl"
    dump({"model": global_model, "feature_cols": feature_cols}, path)
    notify(f"[MODEL] Saved global model → {path}")

    _cleanup_old_models(cutoff_date)

    return {
        "model": global_model,
        "fitted_global": fitted_global,
    }

# ------------------------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------------------------
def predict_components_for_df(
    df: pd.DataFrame,
    feature_cols: list,
    global_model,
    sku_dense_map: dict,
):
    """
    Returns:
      pred_model_all: model-based prediction, length=len(df)
      pred_base_all : baseline prediction, length=len(df)
      dense_mask    : bool mask, length=len(df)
    """
    if df.empty:
        z = np.zeros(0, dtype=float)
        return z, z, z.astype(bool)

    # -------------------------
    # Dense / sparse mask
    # -------------------------
    dense_mask = df.apply(
        lambda r: sku_dense_map.get(
            (int(r["Cadena"]), int(r["Tienda"]), int(r["Producto"])),
            False,
        ),
        axis=1,
    ).to_numpy(dtype=bool)

    # -------------------------
    # Baseline (hierarchical, stable)
    # -------------------------
    lag52 = df["lag_52w"].to_numpy(float)
    lag3  = df["lag_3w"].to_numpy(float)
    lag2  = df["lag_2w"].to_numpy(float)
    lag1  = df["lag_1w"].to_numpy(float)
    roll12 = df["roll12w"].to_numpy(float)

    pred_base_all = np.select(
        [
            lag52 > 0,
            lag3 > 0,
            lag2 > 0,
            lag1 > 0,
            roll12 > 0,
        ],
        [
            lag52,
            lag3,
            lag2,
            lag1,
            roll12,
        ],
        default=0.0,
    )

    pred_base_all = np.clip(pred_base_all, 0, None)

    # -------------------------
    # Model prediction (safe)
    # -------------------------
    pred_model_all = np.zeros(len(df), dtype=float)

    if global_model is None:
        return pred_model_all, pred_base_all, dense_mask

    # lag_52w optional
    required_cols = [c for c in feature_cols if c != "lag_52w"]
    valid_mask = df[required_cols].notna().all(axis=1).to_numpy()

    if not valid_mask.any():
        return pred_model_all, pred_base_all, dense_mask

    X_df = df.loc[valid_mask, feature_cols].copy()

    # Fallback for missing lag_52w
    if "lag_52w" in X_df.columns:
        X_df["lag_52w"] = (
            X_df["lag_52w"]
            .fillna(X_df["roll12w"])
            .fillna(X_df["lag_3w"])
            .fillna(X_df["lag_1w"])
            .fillna(0.0)
        )

    X = X_df.astype(float).to_numpy()
    preds = np.expm1(global_model.predict(X)).clip(0, None)

    pred_model_all[valid_mask] = preds

    return pred_model_all, pred_base_all, dense_mask

def _cleanup_old_models(current_cutoff: pd.Timestamp) -> None:
    """
    Delete old model .pkl files in WEEKLY_MODEL_DIR based on MODEL_RETENTION_MONTHS.

    Keeps models for months >= (current_cutoff - MODEL_RETENTION_MONTHS).
    Filenames expected like:
        model_global_YYYYMM.pkl
    """
    if KEEP_ALL_MODELS or MODEL_RETENTION_MONTHS <= 0:
        return

    os.makedirs(WEEKLY_MODEL_DIR, exist_ok=True)

    cutoff_month = current_cutoff.to_period("M")
    keep_from_month = cutoff_month - MODEL_RETENTION_MONTHS + 1

    pattern = re.compile(r".*_(\d{6})\.pkl$")  # capture YYYYMM
    deleted = 0

    for fname in os.listdir(WEEKLY_MODEL_DIR):
        if not fname.endswith(".pkl"):
            continue

        m = pattern.match(fname)
        if not m:
            continue

        yyyymm = m.group(1)
        try:
            file_period = pd.Period(f"{yyyymm[:4]}-{yyyymm[4:]}", freq="M")
        except Exception:
            continue

        if file_period < keep_from_month:
            try:
                os.remove(os.path.join(WEEKLY_MODEL_DIR, fname))
                deleted += 1
            except OSError:
                pass

    if deleted > 0:
        notify(f"[MODEL] Cleanup: deleted {deleted} old model file(s).")
