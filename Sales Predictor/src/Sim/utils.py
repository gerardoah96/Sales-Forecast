"""
utils.py

Shared utilities for the forecast simulation system:
    - notify()              → logging / stdout messages
    - month_to_season()     → map month to one of 4 seasons
    - safe_div()            → safe division helper
    - clip_or_default()     → numeric clipping helper
    - append_forecast_history() → local history store that simulates DB behavior
"""

import os
import numpy as np
import pandas as pd

# ==============================================================================
# LOGGING / MESSAGING
# ==============================================================================
def notify(msg: str):
    print(msg, flush=True)

# ==============================================================================
# MONTH → 4-SEASON MAPPING
# ==============================================================================
def month_to_season(month: int) -> int:
    if month in (12, 1, 2):
        return 1
    elif month in (3, 4, 5):
        return 2
    elif month in (6, 7, 8):
        return 3
    else:
        return 4

# ==============================================================================
# SAFE NUMERIC HELPERS
# ==============================================================================
def safe_div(a, b, default=0.0):
    if b is None or b == 0 or b is np.nan:
        return default
    try:
        return a / b
    except Exception:
        return default

def clip_or_default(val, minv=0.0, maxv=None, default=1.0):
    if val is None or not np.isfinite(val):
        return default
    if maxv is None:
        return max(minv, float(val))
    return float(np.clip(val, minv, maxv))

# ==============================================================================
# FORECAST HISTORY (LOCAL DB SIMULATOR)
# ==============================================================================
def _ensure_date_int_yyyymmdd(series: pd.Series) -> pd.Series:
    """
    Ensure Date is stored as int yyyymmdd (safe, stable, no epoch-nanosecond bugs).
    Accepts datetime, string, or int-ish.
    """
    if np.issubdtype(series.dtype, np.datetime64):
        return series.dt.strftime("%Y%m%d").astype(int)

    s = series.astype(str).str.strip()

    # If already yyyymmdd
    mask_8 = s.str.fullmatch(r"\d{8}", na=False)
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out.loc[mask_8] = s.loc[mask_8].astype(int)

    # Otherwise parse as datetime
    mask_other = ~mask_8
    parsed = pd.to_datetime(s.loc[mask_other], errors="coerce")
    out.loc[mask_other] = parsed.dt.strftime("%Y%m%d").astype("float64")

    return out.fillna(0).astype(int)

def append_forecast_history(df_new: pd.DataFrame,
                            run_date: pd.Timestamp,
                            path: str = "forecast_history.csv"):
    """
    Append forecast results to local history store.
    Ensures:
        - date-only (no time)
        - no duplicate future rows
        - safe for repeated same-day runs
    """

    if df_new is None or df_new.empty:
        notify("[HISTORY] Nothing to append — skipping")
        return

    df_new = df_new.copy()

    # ✅ DATE ONLY — NO TIME
    df_new["Date"] = pd.to_datetime(df_new["Date"]).dt.date
    df_new["forecast_run_date"] = pd.to_datetime(run_date).date()

    # Load existing history if present
    if os.path.exists(path):
        df_old = pd.read_csv(path)

        # Normalize old data safely
        df_old["Date"] = pd.to_datetime(
            df_old["Date"], format="mixed", errors="coerce"
        ).dt.date

        df_old["forecast_run_date"] = pd.to_datetime(
            df_old["forecast_run_date"], format="mixed", errors="coerce"
        ).dt.date

        min_new_date = df_new["Date"].min()
        df_old = df_old[df_old["Date"] < min_new_date]

        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(path, index=False)
    notify(f"[HISTORY] Appended {len(df_new)} rows to {path}")

def compute_smoothed_residuals(df, span=8):
    df = df.sort_values("Date").copy()

    df["residual"] = df["WeeklyTotal"] - df["y_hat"]

    df["resid_ewma"] = (
        df.groupby(["Cadena", "Tienda", "Producto"])["residual"]
          .transform(lambda x: x.ewm(span=span, min_periods=3).mean())
    )

    return df


def compute_confidence_score(df):
    nz_12 = (
        df.groupby(["Cadena", "Tienda", "Producto"])["WeeklyTotal"]
          .transform(lambda x: x.rolling(12, min_periods=3)
                              .apply(lambda v: (v > 0).mean()))
    )

    hist_len = (
        df.groupby(["Cadena", "Tienda", "Producto"])["WeeklyTotal"]
          .transform("count")
    )

    confidence = (
        0.6 * nz_12.clip(0, 1) +
        0.4 * (hist_len / 52).clip(0, 1)
    )

    return confidence.clip(0, 1)

