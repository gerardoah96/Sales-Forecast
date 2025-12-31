"""
compute_bias_table.py

Computes Cadena × MonthNum bias factors from CLOSED historical data only.

Features:
- Alpha backtesting (Step 2)
- Minimum-years-per-chain gate (Step 3)
- Bias drift logging (Step 4)

Output:
- bias_table.pkl      → consumed by simulator.py
- bias_drift_log.csv  → audit & monitoring
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime

from Testing.DB.T_DB import SQL_fetcher
from utils import notify


# ==============================================================================
# CONFIGURATION
# ==============================================================================
EXCLUDED_CHAINS = {1, 2, 3}     # weak 2024 history
ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25]
MIN_IMPROVEMENT = 0.02         # 2% required improvement
CLIP_LOW = 0.90
CLIP_HIGH = 1.10
MIN_ACTUAL = 500.0             # min volume per chain×month
MIN_YEARS = 2
MIN_WEEKS = 52 * MIN_YEARS


# ==============================================================================
# METRICS
# ==============================================================================
def weighted_mape(actual, forecast):
    actual = np.asarray(actual)
    forecast = np.asarray(forecast)
    mask = actual > 0
    if not np.any(mask):
        return np.inf
    return np.average(
        np.abs(actual[mask] - forecast[mask]) / actual[mask],
        weights=actual[mask]
    )


# ==============================================================================
# ALPHA SELECTION
# ==============================================================================
def select_best_alpha(grp: pd.DataFrame):
    """
    grp must contain:
        Actual_sum, Forecast_sum
    """
    base_error = weighted_mape(
        grp["Actual_sum"],
        grp["Forecast_sum"]
    )

    best_alpha = None
    best_error = base_error

    for alpha in ALPHAS:
        adj_forecast = grp["Forecast_sum"] * (
            1.0 + alpha * (grp["Actual_sum"] / grp["Forecast_sum"] - 1.0)
        )

        err = weighted_mape(grp["Actual_sum"], adj_forecast)

        if err < best_error * (1 - MIN_IMPROVEMENT):
            best_error = err
            best_alpha = alpha

    return best_alpha, base_error, best_error


# ==============================================================================
# BIAS COMPUTATION
# ==============================================================================
def compute_bias_table() -> pd.DataFrame:
    notify("[BIAS] Loading historical data...")

    df = SQL_fetcher()
    if df.empty:
        raise RuntimeError("SQL_fetcher returned empty DataFrame")

    df["Date"] = pd.to_datetime(df["Date"])
    df["Cantidad"] = (
        pd.to_numeric(df["Cantidad"], errors="coerce")
          .fillna(0.0)
          .clip(lower=0.0)
    )

    # --------------------------------------------------------------------------
    # WEEKLY AGGREGATION
    # --------------------------------------------------------------------------
    df["Week"] = df["Date"].dt.to_period("W-SUN")

    weekly_actual = (
        df.groupby(["Week", "Cadena"])
          .agg(Actual=("Cantidad", "sum"))
          .reset_index()
    )

    weekly_actual["Date"] = weekly_actual["Week"].dt.to_timestamp()
    weekly_actual["MonthNum"] = weekly_actual["Date"].dt.month

    # --------------------------------------------------------------------------
    # LOAD HISTORICAL FORECASTS (MATCHING SIMULATOR OUTPUT)
    # --------------------------------------------------------------------------
    notify("[BIAS] Loading historical forecasts...")

    fc = pd.read_csv("Weekly_2025_Forecast.csv")
    fc["Date"] = pd.to_datetime(fc["Date"].astype(str))
    fc = (
        fc.groupby(["Date", "Cadena"])
          .agg(Forecast=("Forecast", "sum"))
          .reset_index()
    )

    # --------------------------------------------------------------------------
    # MERGE ACTUAL VS FORECAST
    # --------------------------------------------------------------------------
    merged = weekly_actual.merge(
        fc,
        on=["Date", "Cadena"],
        how="inner",
    )

    # --------------------------------------------------------------------------
    # EXCLUSIONS & HISTORY GATES
    # --------------------------------------------------------------------------
    merged = merged[~merged["Cadena"].isin(EXCLUDED_CHAINS)]

    chain_week_counts = (
        merged.groupby("Cadena")["Week"]
              .nunique()
              .to_dict()
    )

    merged = merged[
        merged["Cadena"].map(chain_week_counts).fillna(0) >= MIN_WEEKS
    ]

    # --------------------------------------------------------------------------
    # AGGREGATE TO CHAIN × MONTH
    # --------------------------------------------------------------------------
    grp = (
        merged.groupby(["Cadena", "MonthNum"])
              .agg(
                  Actual_sum=("Actual", "sum"),
                  Forecast_sum=("Forecast", "sum"),
              )
              .reset_index()
    )

    grp = grp[
        (grp["Actual_sum"] >= MIN_ACTUAL) &
        (grp["Forecast_sum"] > 0)
    ]

    if grp.empty:
        notify("[BIAS] No valid data after filtering — skipping bias")
        return pd.DataFrame(columns=["Cadena", "MonthNum", "bias_factor"])

    grp["raw_ratio"] = grp["Actual_sum"] / grp["Forecast_sum"]

    # --------------------------------------------------------------------------
    # STEP 2: ALPHA BACKTESTING
    # --------------------------------------------------------------------------
    best_alpha, base_err, best_err = select_best_alpha(grp)

    if best_alpha is None:
        notify("[BIAS] No alpha improved results — skipping bias update")
        return pd.DataFrame(columns=["Cadena", "MonthNum", "bias_factor"])

    notify(
        f"[BIAS] Alpha selected: {best_alpha:.2f} | "
        f"WMAPE {base_err:.4f} → {best_err:.4f}"
    )

    # --------------------------------------------------------------------------
    # COMPUTE FINAL BIAS
    # --------------------------------------------------------------------------
    grp["bias_factor"] = 1.0 + best_alpha * (grp["raw_ratio"] - 1.0)
    grp["bias_factor"] = grp["bias_factor"].clip(CLIP_LOW, CLIP_HIGH)

    return grp[["Cadena", "MonthNum", "bias_factor"]]


# ==============================================================================
# DRIFT LOGGING
# ==============================================================================
def log_bias_drift(bias_table: pd.DataFrame):
    if bias_table.empty:
        return

    log = bias_table.copy()
    log["run_date"] = datetime.utcnow().strftime("%Y-%m-%d")

    path = "bias_drift_log.csv"
    header = not os.path.exists(path)

    log.to_csv(path, mode="a", header=header, index=False)


# ==============================================================================
# ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    notify("[BIAS] Starting bias computation...")

    bias_table = compute_bias_table()

    if bias_table.empty:
        notify("[BIAS] No bias written (neutral behavior).")
    else:
        bias_table.to_pickle("bias_table.pkl")
        log_bias_drift(bias_table)
        notify(f"[BIAS] Saved {len(bias_table)} bias factors → bias_table.pkl")
