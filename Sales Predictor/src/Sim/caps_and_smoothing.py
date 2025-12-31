"""
caps_and_smoothing.py

Implements:
    - Chain-level YoY weekly caps
    - SKU-level YoY caps
    - Chain-month caps
    - Improved within-month smoothing (Option C+)

All caps are upper-bound only and never push predictions upward.
"""

import numpy as np
import pandas as pd

from .config import (
    CHAIN_YOY_MAX_RATIO,
    SKU_CAP_TIERS,
    MONTH_MAX_RATIO_DEFAULT,
    MONTH_MAX_RATIO_HIGHCHAIN,
    SMOOTH_ALPHA,
)
from .utils import notify, month_to_season


# ==============================================================================
# CHAIN-LEVEL YoY WEEKLY CAPS
# ==============================================================================
def apply_chain_week_caps(out: pd.DataFrame,
                          weekly_hist: pd.DataFrame,
                          chain_use_yoy_caps: dict[int, bool]) -> pd.DataFrame:
    """
    Applies upper caps based on last year's same-week chain total.
    """
    notify("[CAPS] Applying chain-level YoY weekly caps...")

    out = out.copy()
    out["Week"] = out["Date"].dt.to_period("W-SUN")
    weekly_hist = weekly_hist.copy()
    weekly_hist["Week"] = weekly_hist["Date"].dt.to_period("W-SUN")

    chain_week_actual = weekly_hist.groupby(["Cadena", "Week"])["WeeklyTotal"].sum()

    forecast_chain_week = (
        out.groupby(["Cadena", "Week"])["Forecast"]
           .sum()
           .reset_index()
           .rename(columns={"Forecast": "Forecast_sum"})
    )
    forecast_chain_week["Week_ref"] = forecast_chain_week["Week"] - 52

    lastyear = (
        chain_week_actual.rename("LastYear_chain_sum")
                         .reset_index()
                         .rename(columns={"Week": "Week_ref"})
    )

    forecast_chain_week = forecast_chain_week.merge(
        lastyear, on=["Cadena", "Week_ref"], how="left"
    )

    scale_map = {}

    for _, r in forecast_chain_week.iterrows():
        c = int(r["Cadena"])
        wk = r["Week"]

        if not chain_use_yoy_caps.get(c, False):
            scale_map[(c, wk)] = 1.0
            continue

        fsum = float(r["Forecast_sum"])
        last_year_sum = r["LastYear_chain_sum"]

        if pd.isna(last_year_sum) or last_year_sum <= 0 or fsum <= 0:
            scale_map[(c, wk)] = 1.0
            continue

        upper = CHAIN_YOY_MAX_RATIO * last_year_sum
        target = min(fsum, upper)
        scale_map[(c, wk)] = target / fsum

    # Apply scaling per row
    out["Forecast"] = out.apply(
        lambda r: r["Forecast"] * scale_map.get((int(r["Cadena"]), r["Week"]), 1.0),
        axis=1,
    )

    out.drop(columns=["Week"], inplace=True, errors="ignore")
    return out


# ==============================================================================
# SKU-LEVEL YoY CAPS
# ==============================================================================
def _sku_max_ratio(last_year_sku: float) -> float:
    """
    Use adaptive caps based on volume tier.
    SKU_CAP_TIERS = {
        "<20": 2.5,
        "<100": 2.0,
        "<1000": 1.5,
        "default": 1.2
    }
    """
    if last_year_sku < 20:
        return SKU_CAP_TIERS["<20"]
    if last_year_sku < 100:
        return SKU_CAP_TIERS["<100"]
    if last_year_sku < 1000:
        return SKU_CAP_TIERS["<1000"]
    return SKU_CAP_TIERS["default"]


def apply_sku_caps(out: pd.DataFrame,
                   weekly_hist: pd.DataFrame,
                   chain_use_yoy_caps: dict[int, bool]) -> pd.DataFrame:
    """
    Applies SKU-level caps vs last-year same-week.
    """
    notify("[CAPS] Applying SKU-level YoY caps...")

    out = out.copy()
    out["Week"] = out["Date"].dt.to_period("W-SUN")
    weekly_hist = weekly_hist.copy()
    weekly_hist["Week"] = weekly_hist["Date"].dt.to_period("W-SUN")

    sku_lastyear = (
        weekly_hist.groupby(["Cadena", "Tienda", "Producto", "Week"])["WeeklyTotal"]
                   .sum()
                   .rename("LastYearSKU")
                   .reset_index()
    )

    out["Week_ref"] = out["Week"] - 52
    out = out.merge(
        sku_lastyear.rename(columns={"Week": "Week_ref"}),
        on=["Cadena", "Tienda", "Producto", "Week_ref"],
        how="left",
    )

    def cap_row(r):
        c = int(r["Cadena"])
        if not chain_use_yoy_caps.get(c, False):
            return r["Forecast"]

        lastyr = r["LastYearSKU"]
        if pd.isna(lastyr) or lastyr <= 0:
            return r["Forecast"]

        max_ratio = _sku_max_ratio(lastyr)
        return min(r["Forecast"], max_ratio * lastyr)

    out["Forecast"] = out.apply(cap_row, axis=1)

    out.drop(columns=["Week", "Week_ref", "LastYearSKU"], inplace=True, errors="ignore")
    return out

def compute_global_fallback_bounds(
    hist_df: pd.DataFrame,
    lookback_months: int = 6,
    clip: tuple = (0.65, 1.35),
):
    """
    Computes fallback (low, high) bounds for YoY ratio
    using recent historical behavior when LY same-month is missing.

    hist_df must contain:
      - Date
      - Actual
      - Forecast
    """
    if hist_df.empty:
        return clip

    hist_df = hist_df.sort_values("Date").tail(lookback_months)

    actual_sum = hist_df["Actual"].sum()
    forecast_sum = hist_df["Forecast"].sum()

    if actual_sum <= 0 or forecast_sum <= 0:
        return clip

    ratio = forecast_sum / actual_sum

    delta = abs(1.0 - ratio)
    low = max(clip[0], 1.0 - delta)
    high = min(clip[1], 1.0 + delta)

    if low >= high:
        return clip

    return (low, high)

# ==============================================================================
# CHAIN-MONTH CAPS (WITH SAFE MULTI-LEVEL FALLBACKS)
# ==============================================================================
def apply_chain_month_caps(
    out: pd.DataFrame,
    weekly_full: pd.DataFrame,
    chain_use_yoy_caps: dict,
    hist_forecasts_df=None,
    season_direction_map=None,
):
    """
    Applies chain-level MONTH caps with the following priority:

    1) Last-year same month (YoY) → volume-aware bounds
    2) Historical ACTUALS (recent months)
    3) Historical FORECASTS (if available)
    4) Global conservative fallback

    Always safe. Never assumes history exists.
    """

    notify("[CAPS] Applying chain-month caps...")

    out = out.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out["Month"] = out["Date"].dt.to_period("M")

    weekly_hist = weekly_full.copy()
    weekly_hist["Date"] = pd.to_datetime(weekly_hist["Date"])
    weekly_hist["Month"] = weekly_hist["Date"].dt.to_period("M")
    # ✅ Use ONLY real actual history for caps/trend logic (prevents leakage)
    if "has_actual" in weekly_hist.columns:
        weekly_hist = weekly_hist[weekly_hist["has_actual"]].copy()
    else:
        weekly_hist = weekly_hist[weekly_hist["WeeklyTotal"].notna()].copy()

    # ------------------------------------------------------------------
    # Aggregate monthly ACTUALS
    # ------------------------------------------------------------------
    hist_month_actuals = (
        weekly_hist
        .groupby(["Cadena", "Month"])["WeeklyTotal"]
        .sum()
        .reset_index()
        .rename(columns={"WeeklyTotal": "ActualSum"})
    )

        # ------------------------------------------------------------------
    # Aggregate monthly FORECASTS (current run)
    # ------------------------------------------------------------------
    forecast_month = (
        out.groupby(["Cadena", "Month"])["Forecast"]
           .sum()
           .reset_index()
           .rename(columns={"Forecast": "ForecastSum"})
    )

    # ------------------------------------------------------------------
    # Attach last-year month reference (YoY)
    # ------------------------------------------------------------------
    forecast_month["Month_LY"] = forecast_month["Month"] - 12

    forecast_month = forecast_month.merge(
        hist_month_actuals.rename(
            columns={"Month": "Month_LY", "ActualSum": "LY_Actual"}
        ),
        on=["Cadena", "Month_LY"],
        how="left",
    )

    # ------------------------------------------------------------------
    # Helper: volume-aware bounds
    # ------------------------------------------------------------------
    def _volume_bounds(vol):
        if vol < 20:
            return 0.60, 1.50
        elif vol < 100:
            return 0.70, 1.35
        elif vol < 500:
            return 0.80, 1.25
        else:
            return 0.90, 1.15

    scale_map = {}

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    for _, r in forecast_month.iterrows():
        c = int(r["Cadena"])
        m = r["Month"]
        fsum = float(r["ForecastSum"])

        if fsum <= 0:
            scale_map[(c, m)] = 1.0
            continue

        # --------------------------------------------------------------
        # 1) YoY SAME MONTH (BEST CASE)
        # --------------------------------------------------------------
        ly = r["LY_Actual"]
        if (
            chain_use_yoy_caps.get(c, False)
            and pd.notna(ly)
            and ly > 0
        ):
            # HARD YoY anchor: never exceed +60% vs same month LY
            upper = 1.60 * ly
            target = min(fsum, upper)
            scale_map[(c, m)] = target / fsum
            continue

        else:
            # ----------------------------------------------------------
            # 2) RECENT ACTUALS FALLBACK
            # ----------------------------------------------------------
            recent_actuals = hist_month_actuals[
                (hist_month_actuals["Cadena"] == c) &
                (hist_month_actuals["Month"] < m)
            ].tail(6)

            if not recent_actuals.empty:
                base = recent_actuals["ActualSum"].mean()
                low, high = _volume_bounds(base)
            # ----------------------------------------------------------
            # 3) HISTORICAL FORECAST FALLBACK (OPTIONAL)
            # ----------------------------------------------------------
            elif hist_forecasts_df is not None and not hist_forecasts_df.empty:
                hist_forecasts_df["Month"] = (
                    pd.to_datetime(hist_forecasts_df["Date"])
                    .dt.to_period("M")
                )

                hf = hist_forecasts_df[
                    (hist_forecasts_df["Cadena"] == c) &
                    (hist_forecasts_df["Month"] < m)
                ]

                if not hf.empty:
                    base = hf.groupby("Month")["Forecast"].sum().mean()
                    low, high = _volume_bounds(base)
                else:
                    base = fsum
                    low, high = 0.65, 1.35

            # ----------------------------------------------------------
            # 4) GLOBAL FALLBACK (LAST RESORT)
            # ----------------------------------------------------------
            else:
                base = fsum
                low, high = 0.65, 1.35
        
        # --------------------------------------------------------------
        # Trend-aware, learned-season bias (soft adjustment)
        # --------------------------------------------------------------
        if season_direction_map is not None:
            season = month_to_season(m.to_timestamp().month)
            season_dir = season_direction_map.get((c, season), 0)

            recent = (
                weekly_hist[
                    (weekly_hist["Cadena"] == c) &
                    (weekly_hist["Date"] < m.to_timestamp())
                ]
                .sort_values("Date")
                .tail(8)["WeeklyTotal"]
            )

            if len(recent) >= 4:
                last_4 = recent.tail(4).sum()
                prev_4 = recent.head(4).sum()
                trend_ratio = (last_4 + 1e-6) / (prev_4 + 1e-6)
            else:
                trend_ratio = 1.0

            trend_ratio = np.clip(trend_ratio, 0.7, 1.3)
            bias_adj = 1.0 + 0.30 * season_dir * (trend_ratio - 1.0)
            bias_adj = np.clip(bias_adj, 0.85, 1.15)
        else:
            bias_adj = 1.0

        # --------------------------------------------------------------
        # Ensure base exists (fallback anchor)
        # --------------------------------------------------------------
        if (base is None) or (not np.isfinite(base)) or (base <= 0):
            base = fsum

        # --------------------------------------------------------------
        # ✅ Correct cap: clip fsum to [base*low, base*high]
        # --------------------------------------------------------------
        lo = base * low * bias_adj
        hi = base * high * bias_adj

        lo = max(lo, 0.0)
        hi = max(hi, lo + 1e-6)

        capped = np.clip(fsum, lo, hi)
        scale_map[(c, m)] = capped / fsum

    # ------------------------------------------------------------------
    # Apply scaling
    # ------------------------------------------------------------------
    out["Forecast"] = out.apply(
        lambda r: r["Forecast"] * scale_map.get((int(r["Cadena"]), r["Month"]), 1.0),
        axis=1,
    )

    out.drop(columns=["Month"], inplace=True, errors="ignore")
    return out

# ==============================================================================
# WITHIN-MONTH SMOOTHING (Option C+)
# ==============================================================================
def apply_within_month_smoothing(out: pd.DataFrame) -> pd.DataFrame:
    """
    Redistributes forecast mass within each chain-month while
    preserving totals and respecting actuals.

    - Weeks with Actual > 0 are locked
    - Only remaining forecast weeks are adjusted
    - Total month forecast is preserved
    """

    notify("[SMOOTH] Applying within-month redistribution...")

    out = out.copy()
    out["Month"] = out["Date"].dt.to_period("M")
    out["Week"] = out["Date"].dt.to_period("W-SUN")

    # How strong the redistribution is (small on purpose)
    ALPHA = 0.15  # 15% pull toward mean

    for (c, m), grp in out.groupby(["Cadena", "Month"]):

        idx = grp.index

        actual = grp["Actual"].to_numpy(float)
        forecast = grp["Forecast"].to_numpy(float)

        # Lock weeks that already have actuals
        locked_mask = actual > 0
        free_mask = ~locked_mask

        if free_mask.sum() <= 1:
            continue

        # Remaining forecast budget
        target_total = forecast.sum()
        locked_total = forecast[locked_mask].sum()
        remaining_budget = target_total - locked_total

        if remaining_budget <= 0:
            continue

        free_forecast = forecast[free_mask]

        mean_free = free_forecast.mean()

        # Pull each week slightly toward the mean
        adjusted = (
            (1 - ALPHA) * free_forecast +
            ALPHA * mean_free
        )

        # Renormalize to preserve remaining budget
        scale = remaining_budget / adjusted.sum()
        adjusted *= scale

        # Write back
        forecast[free_mask] = adjusted
        out.loc[idx, "Forecast"] = forecast

    out.drop(columns=["Month", "Week"], inplace=True, errors="ignore")
    return out

def reconcile_chain_week_to_sku(out: pd.DataFrame) -> pd.DataFrame:
    """
    After SKU-level caps, chain-week totals can shrink unintentionally.
    This function re-scales *only the free (unlocked)* SKUs inside each (Cadena, Week)
    so that the chain-week total is preserved.

    - Locked rows: Actual > 0 (do not change)
    - Free rows: Actual == 0 (scaled proportionally)
    """
    notify("[RECON] Reconciling chain-week totals after SKU caps...")

    out = out.copy()
    out["Week"] = out["Date"].dt.to_period("W-SUN")

    # Process each chain-week
    for (c, wk), grp in out.groupby(["Cadena", "Week"]):
        idx = grp.index

        actual = grp["Actual"].to_numpy(float)
        fcst   = grp["Forecast"].to_numpy(float)

        locked = actual > 0
        free   = ~locked

        if free.sum() == 0:
            continue

        # Target chain-week total = current total (post chain-week caps)
        target_total = fcst.sum()
        locked_total = fcst[locked].sum()
        remaining = target_total - locked_total

        if remaining <= 0:
            continue

        free_fcst = fcst[free]

        # If everything is zero, nothing to distribute
        if free_fcst.sum() <= 0:
            continue

        # Scale free forecasts proportionally to preserve remaining budget
        scale = remaining / free_fcst.sum()
        fcst[free] = free_fcst * scale

        out.loc[idx, "Forecast"] = fcst

    out.drop(columns=["Week"], inplace=True, errors="ignore")
    return out
