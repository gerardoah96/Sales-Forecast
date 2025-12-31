"""
seasonality.py

Compute historical seasonality indices at different aggregation levels:

    - Global month-level seasonality
    - Chain-level month seasonality
    - SKU-level month seasonality
    - Product-level 4-season index

All functions use *only history prior to sim_start* to avoid leakage.

Important improvements:
    - Shrink seasonal indices toward 1.0 to reduce volatility
    - Clip extremes to avoid instability at season boundaries
    - Vectorized logic where possible
"""

import numpy as np
import pandas as pd

from .config import (
    GLOBAL_SZN_LOOKBACK_MONTHS,
    CHAIN_SZN_LOOKBACK_MONTHS,
    SKU_SZN_LOOKBACK_MONTHS,
    PRODUCT_SZN_LOOKBACK_MONTHS,
)
from .utils import notify, month_to_season


# ------------------------------------------------------------------------------
# Helper: shrink + clip seasonal indices
# ------------------------------------------------------------------------------
def _shrink_and_clip(val: float, shrink_alpha=0.5, clip_min=0.8, clip_max=1.2) -> float:
    """
    Move val toward 1.0 using shrink_alpha, then clip to safe bounds.

    shrink_alpha:
        1.0 = keep full amplitude
        0.5 = cut amplitude in half
        0.0 = flatten to 1.0
    """
    if not np.isfinite(val) or val <= 0:
        return 1.0

    # shrink toward 1.0
    v = 1.0 + shrink_alpha * (val - 1.0)

    # clip
    v = float(np.clip(v, clip_min, clip_max))
    return v


# ------------------------------------------------------------------------------
# Global monthly seasonality
# ------------------------------------------------------------------------------
def compute_global_seasonality_index(weekly: pd.DataFrame, sim_start: pd.Timestamp) -> dict[int, float]:
    """
    Compute month -> global seasonality index using recent history prior to sim_start.
    """
    notify("[SZN] Computing global seasonality...")

    hist = weekly[weekly["Date"] < sim_start].copy()
    if hist.empty:
        return {}

    start = sim_start - pd.DateOffset(months=GLOBAL_SZN_LOOKBACK_MONTHS)
    hist = hist[hist["Date"] >= start].copy()
    if hist.empty:
        return {}

    hist["MonthNum"] = hist["Date"].dt.month

    month_totals = (
        hist.groupby("MonthNum")["WeeklyTotal"]
            .sum()
            .reset_index()
    )

    if month_totals["WeeklyTotal"].sum() <= 0:
        return {}

    month_totals["share"] = month_totals["WeeklyTotal"] / month_totals["WeeklyTotal"].sum()
    mean_share = month_totals["share"].mean()

    # global seasonal index = share / mean_share
    month_totals["global_szn"] = month_totals["share"] / (mean_share if mean_share > 0 else 1.0)

    # shrink + clip
    idx = {}
    for _, r in month_totals.iterrows():
        m = int(r["MonthNum"])
        val = _shrink_and_clip(r["global_szn"], shrink_alpha=0.5, clip_min=0.85, clip_max=1.15)
        idx[m] = val

    notify(f"[SZN] Global seasonality computed for {len(idx)} months.")
    return idx


# ------------------------------------------------------------------------------
# Chain-level monthly seasonality
# ------------------------------------------------------------------------------
def compute_chain_seasonality_index(weekly: pd.DataFrame, sim_start: pd.Timestamp) -> dict[tuple[int, int], float]:
    """
    Compute chain × month seasonality index.

    Returns:
        (Cadena, MonthNum) -> seasonality_index
    """
    notify("[SZN] Computing chain-level seasonality...")

    hist = weekly[weekly["Date"] < sim_start].copy()
    if hist.empty:
        return {}

    start = sim_start - pd.DateOffset(months=CHAIN_SZN_LOOKBACK_MONTHS)
    hist = hist[hist["Date"] >= start].copy()
    if hist.empty:
        return {}

    hist["MonthNum"] = hist["Date"].dt.month

    chain_month = (
        hist.groupby(["Cadena", "MonthNum"])["WeeklyTotal"]
            .sum()
            .reset_index()
            .rename(columns={"WeeklyTotal": "sum_month"})
    )

    chain_total = (
        hist.groupby("Cadena")["WeeklyTotal"]
            .sum()
            .reset_index()
            .rename(columns={"WeeklyTotal": "sum_chain"})
    )

    merged = chain_month.merge(chain_total, on="Cadena", how="left")

    # month_share = sum_month / sum_chain
    merged["month_share"] = merged["sum_month"] / merged["sum_chain"].replace(0.0, np.nan)

    # normalize: index = share / mean(share) for each chain
    merged["szn_index"] = merged.groupby("Cadena")["month_share"].transform(
        lambda s: s / (s.mean() if s.mean() > 0 else 1.0)
    )

    # shrink + clip
    idx = {}
    for _, r in merged.iterrows():
        c = int(r["Cadena"])
        m = int(r["MonthNum"])
        val = _shrink_and_clip(r["szn_index"], shrink_alpha=0.5, clip_min=0.8, clip_max=1.2)
        idx[(c, m)] = val

    notify(f"[SZN] Chain seasonality generated for {len(idx)} chain-month pairs.")
    return idx


# ------------------------------------------------------------------------------
# SKU-level monthly seasonality
# ------------------------------------------------------------------------------
def compute_sku_seasonality_index(weekly: pd.DataFrame, sim_start: pd.Timestamp) -> dict[tuple[int, int, int], float]:
    """
    Compute SKU-level seasonality (per chain × product × month).

    Returns:
        (Cadena, Producto, MonthNum) -> seasonality_index
    """
    notify("[SZN] Computing SKU-level seasonality...")

    hist = weekly[weekly["Date"] < sim_start].copy()
    if hist.empty:
        return {}

    start = sim_start - pd.DateOffset(months=SKU_SZN_LOOKBACK_MONTHS)
    hist = hist[hist["Date"] >= start].copy()
    if hist.empty:
        return {}

    hist["MonthNum"] = hist["Date"].dt.month

    sku_month = (
        hist.groupby(["Cadena", "Producto", "MonthNum"])["WeeklyTotal"]
            .sum()
            .reset_index()
            .rename(columns={"WeeklyTotal": "sku_month"})
    )

    sku_total = (
        hist.groupby(["Cadena", "Producto"])["WeeklyTotal"]
            .sum()
            .reset_index()
            .rename(columns={"WeeklyTotal": "sku_total"})
    )

    merged = sku_month.merge(sku_total, on=["Cadena", "Producto"], how="left")

    merged["season_share"] = merged["sku_month"] / merged["sku_total"].replace(0.0, np.nan)

    merged["season_index"] = merged.groupby(["Cadena", "Producto"])["season_share"].transform(
        lambda s: s / (s.mean() if s.mean() > 0 else 1.0)
    )

    # shrink + clip to reduce SKU volatility
    idx = {}
    for _, r in merged.iterrows():
        c = int(r["Cadena"])
        p = int(r["Producto"])
        m = int(r["MonthNum"])
        val = _shrink_and_clip(r["season_index"], shrink_alpha=0.4, clip_min=0.8, clip_max=1.25)
        idx[(c, p, m)] = val

    notify(f"[SZN] SKU seasonality generated for {len(idx)} sku-month triples.")
    return idx


# ------------------------------------------------------------------------------
# Product-level 4-season index
# ------------------------------------------------------------------------------
def compute_product_season_index(weekly: pd.DataFrame, sim_start: pd.Timestamp) -> dict[tuple[int, int], float]:
    """
    Compute seasonality by product across the four seasons:
        1 = Winter (Dec–Feb)
        2 = Spring (Mar–May)
        3 = Summer (Jun–Aug)
        4 = Fall   (Sep–Nov)

    Returns:
        (Producto, Season) -> season_index
    """
    notify("[SZN] Computing product-level 4-season index...")

    hist = weekly[weekly["Date"] < sim_start].copy()
    if hist.empty:
        return {}

    start = sim_start - pd.DateOffset(months=PRODUCT_SZN_LOOKBACK_MONTHS)
    hist = hist[hist["Date"] >= start].copy()
    if hist.empty:
        return {}

    hist["MonthNum"] = hist["Date"].dt.month
    hist["Season"] = hist["MonthNum"].apply(month_to_season)

    prod_season = (
        hist.groupby(["Producto", "Season"])["WeeklyTotal"]
            .sum()
            .reset_index()
            .rename(columns={"WeeklyTotal": "prod_season"})
    )

    prod_total = (
        hist.groupby("Producto")["WeeklyTotal"]
            .sum()
            .reset_index()
            .rename(columns={"WeeklyTotal": "prod_total"})
    )

    merged = prod_season.merge(prod_total, on="Producto", how="left")

    merged["season_share"] = merged["prod_season"] / merged["prod_total"].replace(0.0, np.nan)

    merged["season_index"] = merged.groupby("Producto")["season_share"].transform(
        lambda s: s / (s.mean() if s.mean() > 0 else 1.0)
    )

    idx = {}
    for _, r in merged.iterrows():
        p = int(r["Producto"])
        s = int(r["Season"])
        val = _shrink_and_clip(r["season_index"], shrink_alpha=0.4, clip_min=0.85, clip_max=1.15)
        idx[(p, s)] = val

    notify(f"[SZN] Product 4-season indices generated for {len(idx)} entries.")
    return idx
