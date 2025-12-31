"""
data_prep.py

Efficient, vectorized weekly dataset construction and feature engineering.

Outputs:
    weekly_df, feature_cols

The weekly_df includes:
    - WeeklyTotal
    - Calendar fields (year, month, week_of_year)
    - Lag features
    - Rolling mean features
    - Intermittency features
    - Category encodings

NOT included here:
    - Seasonality features (added in seasonality.py)
    - Density flags / tiers (handled in modeling.py)
"""

import pandas as pd
import numpy as np

from .config import KEYS
from .utils import notify


# ------------------------------------------------------------------------------
# Internal helper for rolling computations
# ------------------------------------------------------------------------------
def _rolling_apply(group, window, func, shift=1, fill=0.0):
    """
    Generic helper: compute a rolling window statistic with shift.
    """
    s = group.shift(shift).rolling(window, min_periods=1).apply(func, raw=False)
    return s.fillna(fill)


# ------------------------------------------------------------------------------
# Build weekly dataset
# ------------------------------------------------------------------------------
def build_weekly_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Convert daily-level sales into weekly aggregates and engineer features.

    Returns
    -------
    weekly : pd.DataFrame
        Weekly records for each (Cadena, Tienda, Producto, Week)
    feature_cols : list[str]
        Feature column names used for model input
    """
    notify("[PREP] Building weekly dataset...")

    df = df.copy()

    # ----------------------------------------------------------------------
    # Clean & normalize input
    # ----------------------------------------------------------------------
    df["Date"] = pd.to_datetime(df["Date"])
    df["Cantidad"] = pd.to_numeric(df["Cantidad"], errors="coerce").clip(lower=0.0)
    df = df.dropna(subset=["Cantidad"])

    # Weekly period key
    df["Week"] = df["Date"].dt.to_period("W-SUN")

    # Weekly aggregate
    weekly = (
        df.groupby(KEYS + ["Week"])["Cantidad"]
          .sum()
          .reset_index()
          .rename(columns={"Cantidad": "WeeklyTotal"})
    )

    # Represent each week by its starting Sunday date
    weekly["Date"] = weekly["Week"].dt.to_timestamp()

    # ----------------------------------------------------------------------
    # Sort and compute calendar features
    # ----------------------------------------------------------------------
    weekly = weekly.sort_values(KEYS + ["Date"]).reset_index(drop=True)

    weekly["year"] = weekly["Date"].dt.year
    weekly["month"] = weekly["Date"].dt.month
    weekly["week_of_year"] = weekly["Date"].dt.isocalendar().week.astype(int)
    weekly["week_index"] = weekly["year"] * 53 + weekly["week_of_year"]

    # Cyclical encoding of week_of_year
    weekly["sin_woy"] = np.sin(2 * np.pi * weekly["week_of_year"] / 52.0)
    weekly["cos_woy"] = np.cos(2 * np.pi * weekly["week_of_year"] / 52.0)

    # Group accessor
    grp = weekly.groupby(KEYS)

    # ----------------------------------------------------------------------
    # Lag features
    # ----------------------------------------------------------------------
    weekly["lag_1w"] = grp["WeeklyTotal"].shift(1).fillna(0.0)
    weekly["lag_2w"] = grp["WeeklyTotal"].shift(2).fillna(0.0)
    weekly["lag_3w"] = grp["WeeklyTotal"].shift(3).fillna(0.0)
    weekly["lag_52w"] = grp["WeeklyTotal"].shift(52).fillna(0.0)

    # ----------------------------------------------------------------------
    # Rolling means
    # ----------------------------------------------------------------------
    weekly["roll3w"] = grp["WeeklyTotal"].shift(1).rolling(3, min_periods=1).mean().reset_index(level=KEYS, drop=True).fillna(0.0)
    weekly["roll12w"] = grp["WeeklyTotal"].shift(1).rolling(12, min_periods=1).mean().reset_index(level=KEYS, drop=True).fillna(0.0)

    # ----------------------------------------------------------------------
    # Nonzero rolling windows
    # ----------------------------------------------------------------------
    shifted = grp["WeeklyTotal"].shift(1)

    nonzero_count_12w = (
        shifted.rolling(12, min_periods=1).apply(lambda x: (x > 0).sum(), raw=False)
               .reset_index(level=KEYS, drop=True)
               .fillna(0.0)
    )
    weekly["nonzero_12w"] = nonzero_count_12w

    roll_sum_12w = (
        shifted.rolling(12, min_periods=1).sum()
               .reset_index(level=KEYS, drop=True)
               .fillna(0.0)
    )

    weekly["avg_nonzero_12w"] = np.where(
        nonzero_count_12w > 0,
        roll_sum_12w / nonzero_count_12w,
        0.0
    )

    # ----------------------------------------------------------------------
    # Weeks since last sale (vectorized scan)
    # ----------------------------------------------------------------------
    def _weeks_since_last_sale(series):
        last = -1
        out = []
        for i, v in enumerate(series):
            if v > 0:
                last = i
            out.append(i - last if last != -1 else np.nan)
        return pd.Series(out, index=series.index)

    weekly["weeks_since_last_sale"] = (
        grp["WeeklyTotal"]
        .apply(_weeks_since_last_sale)
        .reset_index(level=KEYS, drop=True)
        .fillna(999.0)
    )

    # ----------------------------------------------------------------------
    # Volatility (std over rolling 3w of shifted data)
    # ----------------------------------------------------------------------
    weekly["volatility_3w"] = (
        shifted.rolling(3, min_periods=1).std()
               .reset_index(level=KEYS, drop=True)
               .fillna(0.0)
    )

    # ----------------------------------------------------------------------
    # Zero-run-length (past 6 weeks)
    # ----------------------------------------------------------------------
    weekly["zero_run_length_6w"] = (
        (shifted == 0).astype(float)
                      .rolling(6, min_periods=1)
                      .sum()
                      .reset_index(level=KEYS, drop=True)
                      .fillna(0.0)
    )

    # ----------------------------------------------------------------------
    # Lag diffs
    # ----------------------------------------------------------------------
    weekly["diff_1w"] = weekly["lag_1w"] - weekly["lag_2w"]
    weekly["diff_2w"] = weekly["lag_2w"] - weekly["lag_3w"]

    # ----------------------------------------------------------------------
    # Category encoding
    # ----------------------------------------------------------------------
    weekly["Cadena_cat"] = weekly["Cadena"].astype("category").cat.codes
    weekly["Tienda_cat"] = weekly["Tienda"].astype("category").cat.codes
    weekly["Producto_cat"] = weekly["Producto"].astype("category").cat.codes

    # ----------------------------------------------------------------------
    # Final feature list
    # ----------------------------------------------------------------------
    feature_cols = [
        # Calendar
        "year", "month", "week_of_year", "week_index",
        "sin_woy", "cos_woy",

        # Lags
        "lag_1w", "lag_2w", "lag_3w", "lag_52w",

        # Rolling stats
        "roll3w", "roll12w",

        # Intermittency
        "nonzero_12w", "avg_nonzero_12w", "weeks_since_last_sale",
        "volatility_3w", "zero_run_length_6w",

        # Diff features
        "diff_1w", "diff_2w",

        # Encodings
        "Cadena_cat", "Tienda_cat", "Producto_cat",
    ]

    notify("[PREP] Weekly dataset built successfully.")

    return weekly, feature_cols
