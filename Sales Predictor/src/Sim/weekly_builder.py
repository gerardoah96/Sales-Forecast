"""
weekly_builder.py

Build the weekly-aggregated dataset with all necessary lag, rolling, and
calendar features for training XGBoost forecasting models.

Output:
    weekly_df, feature_cols
"""

import numpy as np
import pandas as pd

from .utils import notify
from .config import TRANSITION_MONTHS

# Consistent key order for SKUs
KEYS = ["Cadena", "Tienda", "Producto"]

# ==============================================================================
# MAIN FUNCTION
# ==============================================================================
def build_weekly_dataset(df: pd.DataFrame):
    """
    Convert raw daily DF into weekly totals, then add:
        - calendar features
        - cyclical encodings
        - lag features
        - rolling windows (3w, 12w)
        - non-zero stats
        - volatility & zero-run features
        - transition-season flags
        - 3-month local level features per SKU
        - categorical encodings

    Returns:
        weekly_df (pd.DataFrame)
        feature_cols (list[str])
    """
    notify("[WEEKLY] Building weekly dataset...")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # Clean Cantidad
    df["Cantidad"] = (
        pd.to_numeric(df["Cantidad"], errors="coerce")
          .fillna(0.0)
          .clip(lower=0.0)
    )

    # Week key: weeks start on Sunday
    df["Week"] = df["Date"].dt.to_period("W-SUN")

    # --- Weekly totals per SKU ------------------------------------------------
    weekly = (
        df.groupby(KEYS + ["Week"])["Cantidad"]
          .sum()
          .reset_index()
          .rename(columns={"Cantidad": "WeeklyTotal"})
    )

    # Represent week by Sunday date (timestamp)
    weekly["Date"] = weekly["Week"].dt.to_timestamp()

    # Sort time order
    weekly = weekly.sort_values(KEYS + ["Week"]).reset_index(drop=True)

    # ==============================================================================
    # CALENDAR FEATURES
    # ==============================================================================
    notify("[WEEKLY] Adding calendar features...")

    weekly["year"]         = weekly["Date"].dt.year
    weekly["month"]        = weekly["Date"].dt.month
    weekly["week_of_year"] = weekly["Date"].dt.isocalendar().week.astype(int)
    weekly["week_index"]   = weekly["year"] * 53 + weekly["week_of_year"]

    # cyclic week-of-year encoding
    weekly["sin_woy"] = np.sin(2 * np.pi * weekly["week_of_year"] / 52.0)
    weekly["cos_woy"] = np.cos(2 * np.pi * weekly["week_of_year"] / 52.0)

    # --------------------------------------------------------------------------
    # Transition-season features
    # --------------------------------------------------------------------------
    # Flag months that are near a season boundary
    weekly["transition_flag"] = weekly["month"].isin(TRANSITION_MONTHS).astype(float)

    # Simple 0/1 position inside each transition pair:
    # (2,3), (5,6), (8,9), (11,12)
    transition_pos_map = {
        2: 0.0, 3: 1.0,
        5: 0.0, 6: 1.0,
        8: 0.0, 9: 1.0,
        11: 0.0, 12: 1.0,
    }
    weekly["transition_pos"] = (
        weekly["month"]
        .map(transition_pos_map)
        .fillna(0.0)
        .astype(float)
    )

    # ==============================================================================
    # LAG FEATURES
    # ==============================================================================
    notify("[WEEKLY] Adding lag/rolling features...")

    grp = weekly.groupby(KEYS)["WeeklyTotal"]

    weekly["lag_1w"]  = grp.shift(1)
    weekly["lag_2w"]  = grp.shift(2)
    weekly["lag_3w"]  = grp.shift(3)
    weekly["lag_52w"] = grp.shift(52)

    # ==============================================================================
    # ROLLING WINDOWS
    # ==============================================================================
    # NOTE: shift(1) ensures no leakage from same week.
    weekly["roll3w"] = (
        grp.shift(1)
           .rolling(3, min_periods=1)
           .mean()
    )

    weekly["roll12w"] = (
        grp.shift(1)
           .rolling(12, min_periods=1)
           .mean()
    )

    # ==============================================================================
    # NON-ZERO STATS
    # ==============================================================================
    # non-zero count in past 12 weeks (shifted)
    nonzero_12 = (
        grp.shift(1)
           .rolling(12, min_periods=1)
           .apply(lambda x: (x > 0).sum(), raw=False)
    )

    weekly["nonzero_12w"] = nonzero_12

    # Avoid repeated recomputation
    roll_sum_12 = (
        grp.shift(1)
           .rolling(12, min_periods=1)
           .sum()
    )

    weekly["avg_nonzero_12w"] = np.where(
        nonzero_12 > 0,
        roll_sum_12 / nonzero_12,
        0.0
    )

    # ==============================================================================
    # WEEKS SINCE LAST SALE
    # ==============================================================================
    def _weeks_since_last_sale(arr):
        last = -1
        out = []
        for i, v in enumerate(arr):
            if v > 0:
                last = i
            out.append(np.nan if last == -1 else (i - last))
        return out

    weekly["weeks_since_last_sale"] = (
        weekly.groupby(KEYS)["WeeklyTotal"]
              .transform(lambda x: pd.Series(_weeks_since_last_sale(x)))
    )

    # ==============================================================================
    # VOLATILITY & ZERO RUN LENGTH
    # ==============================================================================
    weekly["volatility_3w"] = (
        grp.shift(1)
           .rolling(3, min_periods=1)
           .std()
    )

    weekly["zero_run_length_6w"] = (
        (grp.shift(1) == 0)
           .astype(float)
           .rolling(6, min_periods=1)
           .sum()
    )

    # ==============================================================================
    # DIFFERENCE FEATURES
    # ==============================================================================
    weekly["diff_1w"] = weekly["lag_1w"] - weekly["lag_2w"]
    weekly["diff_2w"] = weekly["lag_2w"] - weekly["lag_3w"]

    # ==============================================================================
    # CATEGORICAL ENCODINGS
    # ==============================================================================
    weekly["Cadena_cat"]   = weekly["Cadena"].astype("category").cat.codes
    weekly["Tienda_cat"]   = weekly["Tienda"].astype("category").cat.codes
    weekly["Producto_cat"] = weekly["Producto"].astype("category").cat.codes

    # ==============================================================================
    # 3-MONTH LOCAL LEVEL FEATURES PER SKU (Cadena, Tienda, Producto)
    # ==============================================================================
    # We aggregate to YearMonth per SKU (including store), compute a
    # rolling 3-month sum and mean, and then merge back to weekly rows.
    weekly["YearMonth"] = weekly["Date"].dt.to_period("M")

    monthly = (
        weekly.groupby(KEYS + ["YearMonth"])["WeeklyTotal"]
              .sum()
              .reset_index()
              .rename(columns={"WeeklyTotal": "MonthlyTotal"})
    )

    # Ensure sorted by SKU and time
    monthly = monthly.sort_values(KEYS + ["YearMonth"]).reset_index(drop=True)

    # Rolling 3-month sum and mean per SKU
    monthly["sku_3m_sum"] = (
        monthly.groupby(KEYS)["MonthlyTotal"]
               .transform(lambda s: s.rolling(3, min_periods=1).sum())
    )
    monthly["sku_3m_mean"] = (
        monthly.groupby(KEYS)["MonthlyTotal"]
               .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )

    # Merge back to weekly rows
    weekly = weekly.merge(
        monthly[KEYS + ["YearMonth", "sku_3m_sum", "sku_3m_mean"]],
        on=KEYS + ["YearMonth"],
        how="left",
    )

    weekly["sku_3m_sum"] = weekly["sku_3m_sum"].fillna(0.0).astype(float)
    weekly["sku_3m_mean"] = weekly["sku_3m_mean"].fillna(0.0).astype(float)

    # Drop helper column if not needed downstream
    weekly.drop(columns=["YearMonth"], inplace=True)

    # ==============================================================================
    # FINAL CLEANUP & FEATURE LIST
    # ==============================================================================
    weekly = weekly.fillna(0.0)

    feature_cols = [
        "year", "month", "week_of_year", "week_index",
        "sin_woy", "cos_woy",
        "lag_1w", "lag_2w", "lag_3w", "lag_52w",
        "roll3w", "roll12w",
        "nonzero_12w", "avg_nonzero_12w",
        "weeks_since_last_sale",
        "volatility_3w", "zero_run_length_6w",
        "diff_1w", "diff_2w",
        "Cadena_cat", "Tienda_cat", "Producto_cat",
        # transition / boundary features
        "transition_flag", "transition_pos",
        # 3-month local level features
        "sku_3m_sum", "sku_3m_mean",
    ]

    notify(f"[WEEKLY] Weekly dataset ready with {len(weekly)} rows and {len(feature_cols)} features.")

    return weekly, feature_cols
