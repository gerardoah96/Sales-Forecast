"""
simulator.py

Main orchestrator for the 2025 weekly simulation process.

Steps:
 1. Load raw SQL data
 2. Build weekly dataset and features
 3. Determine valid chains (minimum history)
 4. Seasonality computation
 5. Recency λ tuning
 6. SKU density + chain tier determination
 7. Chain × season calibration factors
 8. Rolling weekly simulation loop
 9. Apply caps & smoothing
10. Produce final DataFrame (Date, Cadena, Tienda, Producto, Actual, Forecast)

All submodules:
    - weekly_builder         → weekly aggregation + feature engineering
    - seasonality            → global/chain/SKU/product seasonal indices
    - modeling               → recency tuning, tier training, predictions
    - caps_and_smoothing     → YoY caps, SKU caps, chain-month caps, smoothing
"""

import numpy as np
import pandas as pd

from .config import WEEKLY_MODEL_DIR, MIN_CHAIN_WEEKS
from .weekly_builder import build_weekly_dataset
from .seasonality import (
    compute_global_seasonality_index,
    compute_chain_seasonality_index,
    compute_sku_seasonality_index,
    compute_product_season_index,
)
from .modeling import (
    tune_recency_lambda,
    compute_sku_density_flags,
    train_models_for_cutoff,
    predict_components_for_df,
)
from .caps_and_smoothing import (
    apply_chain_week_caps,
    apply_sku_caps,
    apply_chain_month_caps,
    apply_within_month_smoothing,
    reconcile_chain_week_to_sku,
)
from .utils import notify, month_to_season
from Testing.DB.T_DB_CSV import SQL_fetcher, fetch_historical_forecasts


# ==============================================================================
# MAIN SIMULATION
# ==============================================================================
def run_weekly_2025_simulation(horizon_weeks: int = 4) -> pd.DataFrame:
    """
    Simulates weekly predictions from Mar–Oct 2025.
    Returns DataFrame:
         Date, Cadena, Tienda, Producto, Actual, Forecast
    """

    notify("[SIM] Starting 2025 simulator...")

    # 1) LOAD RAW DATA ---------------------------------------------------------
    df = SQL_fetcher()
    if df.empty:
        notify("[SIM] ERROR: SQL_fetcher returned empty DataFrame.")
        return _empty_output()

    df["Date"] = pd.to_datetime(df["Date"])
    df["Cantidad"] = pd.to_numeric(df["Cantidad"], errors="coerce").fillna(0.0)

    # ------------------------------------------------------------------
    # DETERMINE LAST ACTUAL DATE & FORECAST HORIZON LIMIT
    # ------------------------------------------------------------------
    last_actual_date = df.loc[df["Cantidad"] > 0, "Date"].max()

    if pd.isna(last_actual_date):
        notify("[SIM][WARN] No positive actual sales found — disabling horizon clamp.")
        forecast_cutoff = None
    else:
        forecast_cutoff = last_actual_date + pd.Timedelta(weeks=horizon_weeks)
        notify(
            f"[SIM] Last actual date: {last_actual_date.date()} "
            f"→ forecasting capped at {forecast_cutoff.date()}"
        )

    # 2) BUILD WEEKLY ----------------------------------------------------------
    weekly_full, feature_cols = build_weekly_dataset(df)
    if weekly_full.empty:
        notify("[SIM] ERROR: weekly dataset is empty.")
        return _empty_output()

    # Simulation window
    sim_start_cal = pd.Timestamp("2025-09-01")
    offset_to_sun = (6 - sim_start_cal.weekday()) % 7
    sim_start = sim_start_cal + pd.Timedelta(days=offset_to_sun)

    sim_end_cal = pd.Timestamp("2025-12-31")

    # Weeks start on Sunday (W-SUN).
    # If sim_end_cal is not Sunday, EXTEND forward to complete the week.
    if sim_end_cal.weekday() != 6:  # not Sunday
        offset_to_next_sun = (6 - sim_end_cal.weekday()) % 7
        sim_end = sim_end_cal + pd.Timedelta(days=offset_to_next_sun)
    else:
        sim_end = sim_end_cal

    notify(f"[SIM] Window: {sim_start.date()} → {sim_end.date()}")

    # 3) FILTER VALID CHAINS ---------------------------------------------------
    hist_pre = weekly_full[weekly_full["Date"] < sim_start]
    chain_week_counts = (
        hist_pre.groupby("Cadena")["Week"]
                .nunique()
                .reset_index()
                .rename(columns={"Week": "Weeks"})
    )

    chain_weeks_map = {
        int(r["Cadena"]): int(r["Weeks"])
        for _, r in chain_week_counts.iterrows()
    }

    valid_chains = [c for c, w in chain_weeks_map.items() if w >= MIN_CHAIN_WEEKS]

    if not valid_chains:
        notify("[SIM] ERROR: No chains with enough history.")
        return _empty_output()

    weekly_full = weekly_full[weekly_full["Cadena"].isin(valid_chains)].copy()

    # ------------------------------------------------------------------
    # ENSURE FUTURE WEEKS EXIST UP TO sim_end
    # ------------------------------------------------------------------
    notify("[SIM] Ensuring weekly rows exist through simulation end...")

    all_weeks = pd.period_range(
        weekly_full["Week"].min(),
        sim_end,
        freq="W-SUN"
    )

    keys = ["Cadena", "Tienda", "Producto"]

    sku_keys = weekly_full[keys].drop_duplicates()
    calendar = pd.DataFrame({"Week": all_weeks})

    full_index = sku_keys.merge(calendar, how="cross")
    full_index["Date"] = full_index["Week"].dt.to_timestamp()

    weekly_full = full_index.merge(
        weekly_full,
        on=keys + ["Week", "Date"],
        how="left",
    )

    # Mark rows that truly have actuals
    weekly_full["has_actual"] = weekly_full["WeeklyTotal"].notna()

    # Recursive target used for lags/rollings during sim
    weekly_full["SimSales"] = weekly_full["WeeklyTotal"].copy()

    # Determine chains eligible for YoY caps
    chain_first_date = weekly_full.groupby("Cadena")["Date"].min().to_dict()
    yoy_threshold = sim_start - pd.Timedelta(days=365)

    chain_use_yoy_caps = {
        int(c): (chain_first_date[c] <= yoy_threshold)
        for c in valid_chains
    }

    # 4) SEASONALITY -----------------------------------------------------------
    global_szn = compute_global_seasonality_index(weekly_full, sim_start)
    chain_szn = compute_chain_seasonality_index(weekly_full, sim_start)
    sku_szn = compute_sku_seasonality_index(weekly_full, sim_start)
    prod_season = compute_product_season_index(weekly_full, sim_start)
    season_direction_map = compute_season_direction(weekly_full)

    # Assign monthly/season indices to dataset
    weekly_full["MonthNum"] = weekly_full["Date"].dt.month
    weekly_full["Season"] = weekly_full["MonthNum"].apply(month_to_season)

    weekly_full["chain_szn_index"] = weekly_full.apply(
        lambda r: _choose_chain_szn(r, chain_szn, global_szn, chain_weeks_map),
        axis=1,
    )

    weekly_full["sku_szn_index"] = weekly_full.apply(
        lambda r: _choose_sku_szn(r, sku_szn, chain_szn, global_szn),
        axis=1,
    )

    weekly_full["prod_season_index"] = weekly_full.apply(
        lambda r: prod_season.get((int(r["Producto"]), int(r["Season"])), 1.0),
        axis=1,
    )

    # ----------------------------------------------------------------------
    # Month-of-year × seasonality interaction features
    # ----------------------------------------------------------------------
    month_num = weekly_full["MonthNum"].astype(float)

    weekly_full["month_chain_szn"] = month_num * weekly_full["chain_szn_index"].astype(float)
    weekly_full["month_sku_szn"] = month_num * weekly_full["sku_szn_index"].astype(float)
    weekly_full["month_prod_season"] = month_num * weekly_full["prod_season_index"].astype(float)

    # Ensure all are included as features
    for col in [
        "chain_szn_index",
        "sku_szn_index",
        "prod_season_index",
        "month_chain_szn",
        "month_sku_szn",
        "month_prod_season",
    ]:
        if col not in feature_cols:
            feature_cols.append(col)

    # 5) RECENCY LAMBDA TUNING -------------------------------------------------
    lam = tune_recency_lambda(weekly_full, feature_cols, sim_start)

    # 6) SKU DENSITY + TIERING -------------------------------------------------
    sku_dense_map = compute_sku_density_flags(weekly_full, sim_start)

    # 8) ROLLING SIMULATION ----------------------------------------------------
    forecast_map_model = {}
    forecast_map_base = {}

    current_week = sim_start
    current_cutoff = None
    current_month_label = None
    global_model = None

    while current_week <= sim_end and (forecast_cutoff is None or current_week <= forecast_cutoff):

        month_start = pd.Timestamp(current_week.year, current_week.month, 1)

        # Re-train first week of month
        if current_cutoff is None or month_start != current_cutoff:
            month_label = month_start.strftime("%Y-%m")

            if current_month_label and month_label != current_month_label:
                notify(f"[MONTH] Finished {current_month_label}")
            notify(f"[MONTH] Starting {month_label}")

            current_month_label = month_label

            train_res = train_models_for_cutoff(
                weekly_full,
                feature_cols,
                month_start,
                lam,
                WEEKLY_MODEL_DIR
            )

            global_model = train_res["model"]
            current_cutoff = month_start

        if global_model is None:
            notify(f"[WARN] No model at cutoff {current_cutoff}, using baseline only.")

        # Build horizon window
        horizon_end = current_week + pd.Timedelta(weeks=horizon_weeks)

        # Recompute lag/rolling features using SimSales (actuals + prior forecasts)
        weekly_full = _recompute_recursive_features(weekly_full)

        test = weekly_full[
            (weekly_full["Date"] >= current_week) &
            (weekly_full["Date"] < horizon_end)
        ].copy()

        # Clamp to simulation window
        test = test[(test["Date"] >= sim_start) & (test["Date"] <= sim_end)].copy()

        if not test.empty:
            pred_model, pred_base, _dense = predict_components_for_df(
                test, feature_cols, global_model, sku_dense_map
            )

            test["Pred_Model"] = pred_model
            test["Pred_Base"] = pred_base

            # Final per-SKU choice (NumPy-safe)
            test["Pred_Final"] = np.clip(
                np.where(_dense, test["Pred_Model"], test["Pred_Base"]),
                0.0,
                None
            )

            # Write recursive forecasts back using key-based update
            upd_cols = ["Date", "Cadena", "Tienda", "Producto", "Pred_Final"]

            weekly_full = weekly_full.merge(
                test[upd_cols],
                on=["Date", "Cadena", "Tienda", "Producto"],
                how="left",
            )

            # SIMULATOR MODE: ALWAYS ALLOW FORWARD OVERWRITE
            mask_update = weekly_full["Pred_Final"].notna() & (weekly_full["Date"] >= current_week)
            weekly_full.loc[mask_update, "SimSales"] = weekly_full.loc[mask_update, "Pred_Final"]
            weekly_full.drop(columns=["Pred_Final"], inplace=True)

            # Save outputs
            for _, r in test.iterrows():
                key = (r["Date"], int(r["Cadena"]), int(r["Tienda"]), int(r["Producto"]))
                forecast_map_model[key] = float(r["Pred_Model"])
                forecast_map_base[key] = float(r["Pred_Base"])

        notify(f"[WEEK] Simulated week start {current_week.date()}")
        current_week += pd.Timedelta(weeks=1)

    if current_month_label:
        notify(f"[MONTH] Finished {current_month_label}")
    notify("[SIM] Raw simulation complete.")

    # Build output DataFrame ---------------------------------------------------
    out_model = _assemble_output(forecast_map_model, weekly_full).rename(columns={"Forecast": "Pred_Model"})
    out_base = _assemble_output(forecast_map_base, weekly_full).rename(columns={"Forecast": "Pred_Base"})

    out = out_model.merge(
        out_base[["Date", "Cadena", "Tienda", "Producto", "Pred_Base"]],
        on=["Date", "Cadena", "Tienda", "Producto"],
        how="left",
    )

    # Initialize Forecast with model prediction
    out["Forecast"] = out["Pred_Model"].copy()

    # ======================================================================
    # ENSEMBLE BLEND WEIGHTS (Model vs Baseline) using last N observed weeks
    # ======================================================================
    lookback_weeks = 8

    hist = out[(out["Date"] <= last_actual_date) & (out["Actual"] > 0)].copy()
    if not hist.empty:
        hist = hist.sort_values("Date")
        hist["abs_err_model"] = np.abs(hist["Pred_Model"] - hist["Actual"])
        hist["abs_err_base"] = np.abs(hist["Pred_Base"] - hist["Actual"])

        # last N weeks per SKU
        hist["rank"] = hist.groupby(["Cadena", "Tienda", "Producto"])["Date"].rank(
            method="first", ascending=False
        )
        histN = hist[hist["rank"] <= lookback_weeks].copy()

        sku_err = (
            histN.groupby(["Cadena", "Tienda", "Producto"], as_index=False)
                 .agg(mae_model=("abs_err_model", "mean"),
                      mae_base=("abs_err_base", "mean"))
        )

        sku_err["w_model"] = sku_err["mae_base"] / (sku_err["mae_model"] + sku_err["mae_base"] + 1e-9)
        sku_err["w_model"] = sku_err["w_model"].clip(0.10, 0.90)

        out = out.merge(
            sku_err[["Cadena", "Tienda", "Producto", "w_model"]],
            on=["Cadena", "Tienda", "Producto"],
            how="left",
        )
        out["w_model"] = out["w_model"].fillna(0.70)  # fallback

        # Apply only to FUTURE weeks
        fut = out["Date"] > last_actual_date
        out.loc[fut, "Forecast"] = (
            out.loc[fut, "w_model"] * out.loc[fut, "Pred_Model"] +
            (1.0 - out.loc[fut, "w_model"]) * out.loc[fut, "Pred_Base"]
        ).clip(lower=0.0)

        out.drop(columns=["w_model"], inplace=True)
        notify(f"[ENS] Ensemble applied (lookback={lookback_weeks} weeks)")
    else:
        notify("[ENS] No history for ensemble weighting — skipping")

    # --------------------------------------------------
    # LAST-ACTUAL ANCHORING (first forecast weeks only)
    # --------------------------------------------------
    anchor_weeks = 2

    last_actual = (
        out[out["Date"] <= last_actual_date]
        .sort_values("Date")
        .groupby(["Cadena", "Tienda", "Producto"])
        .tail(1)[["Cadena", "Tienda", "Producto", "Actual"]]
    )

    out = out.merge(
        last_actual,
        on=["Cadena", "Tienda", "Producto"],
        how="left",
        suffixes=("", "_last")
    )

    fut = out["Date"] > last_actual_date
    week_idx = out.loc[fut].groupby(["Cadena", "Tienda", "Producto"]).cumcount()
    w_anchor = (1 - week_idx / anchor_weeks).clip(0, 1)

    out.loc[fut, "Forecast"] = (
        w_anchor * out.loc[fut, "Actual_last"] +
        (1 - w_anchor) * out.loc[fut, "Forecast"]
    )

    out.drop(columns=["Actual_last"], inplace=True)

    # ======================================================================
    # CONFIDENCE-WEIGHTED RESIDUAL NUDGING (future only)
    # ======================================================================
    hist_mask = out["Date"] <= last_actual_date
    hist = out[hist_mask].copy()

    if len(hist) >= 100:
        hist["residual"] = hist["Actual"] - hist["Forecast"]

        hist["resid_ewma"] = (
            hist.sort_values("Date")
                .groupby(["Cadena", "Tienda", "Producto"])["residual"]
                .transform(lambda x: x.ewm(span=8, min_periods=3).mean())
        )

        nz_12 = (
            hist.groupby(["Cadena", "Tienda", "Producto"])["Actual"]
                .transform(lambda x: x.rolling(12, min_periods=3)
                                    .apply(lambda v: (v > 0).mean()))
        )

        hist_len = hist.groupby(["Cadena", "Tienda", "Producto"])["Actual"].transform("count")

        hist["confidence"] = (
            0.6 * nz_12.clip(0, 1) +
            0.4 * (hist_len / 52).clip(0, 1)
        ).clip(0, 1)

        resid_state = (
            hist.sort_values("Date")
                .groupby(["Cadena", "Tienda", "Producto"])
                .tail(1)[["Cadena", "Tienda", "Producto", "resid_ewma", "confidence"]]
        )

        out = out.merge(resid_state, on=["Cadena", "Tienda", "Producto"], how="left")

        future_mask = out["Date"] > last_actual_date
        alpha = 0.35 * out.loc[future_mask, "confidence"].fillna(0.0)
        adj = alpha * out.loc[future_mask, "resid_ewma"].fillna(0.0)
        adj[out.loc[future_mask, "confidence"].fillna(0.0) < 0.2] = 0.0

        out.loc[future_mask, "Forecast"] = (out.loc[future_mask, "Forecast"] - adj).clip(lower=0.0)

        out.drop(columns=["resid_ewma", "confidence"], inplace=True, errors="ignore")
        notify("[BIAS] Confidence-weighted residual nudging applied")
    else:
        notify("[BIAS] Not enough history for residual nudging")

    # ======================================================================
    # FETCH HISTORICAL FORECASTS (FOR MONTHLY FALLBACK CAPS)
    # ======================================================================
    try:
        current_run_date = pd.Timestamp.today().normalize()
        hist_forecasts_df = fetch_historical_forecasts(cutoff_run_date=current_run_date)
        notify(f"[DB] Loaded {len(hist_forecasts_df)} historical forecasts for caps fallback")
    except Exception as e:
        notify(f"[DB][WARN] Could not load historical forecasts: {e}")
        hist_forecasts_df = None

    # ======================================================================
    # APPLY RECONCILIATION + CAPS + SMOOTHING (FUTURE ONLY)
    # ======================================================================
    out["Date"] = pd.to_datetime(out["Date"])

    out_hist = out[out["Date"] <= last_actual_date].copy()
    out_fut = out[out["Date"] > last_actual_date].copy()

    # 1) Reconcile hierarchy (future only)
    out_fut = reconcile_chain_week_to_sku(out_fut)

    # 2) Caps (future only)
    out_fut = apply_chain_week_caps(out_fut, weekly_full, chain_use_yoy_caps)
    out_fut = apply_sku_caps(out_fut, weekly_full, chain_use_yoy_caps)

    out_fut = apply_chain_month_caps(
        out_fut,
        weekly_full,
        chain_use_yoy_caps,
        hist_forecasts_df=hist_forecasts_df,
        season_direction_map=season_direction_map,
    )

    # 3) Within-month smoothing (future only)
    out_fut = apply_within_month_smoothing(out_fut)

    # Recombine (history untouched)
    out = pd.concat([out_hist, out_fut], ignore_index=True)

    # --------------------------------------------------
    # FINAL FORMATTING + RETURN (CRITICAL)
    # --------------------------------------------------
    out["Forecast"] = np.round(out["Forecast"]).fillna(0).astype(int)
    out["Actual"] = np.maximum(np.round(out["Actual"]).fillna(0), 0).astype(int)

    out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
    out = out.sort_values(["Date", "Cadena", "Tienda", "Producto"]).reset_index(drop=True)

    notify("[SIM] Finished end-to-end simulation.")
    return out


# ==============================================================================
# HELPERS (kept once)
# ==============================================================================
def compute_season_direction(weekly_full: pd.DataFrame) -> dict[tuple[int, int], int]:
    """
    Learns season direction (+1 / 0 / -1) from historical YoY behavior.
    Works for any dataset.
    """
    df = weekly_full.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Season"] = df["Date"].dt.month.apply(month_to_season)

    season_sales = (
        df.groupby(["Cadena", "Season", "Year"], as_index=False)
          .agg(season_sales=("WeeklyTotal", "sum"))
    )

    season_sales["prev_year_sales"] = (
        season_sales.groupby(["Cadena", "Season"])["season_sales"].shift(1)
    )

    season_sales["yoy_change"] = season_sales["season_sales"] / season_sales["prev_year_sales"] - 1.0

    avg_yoy = (
        season_sales.dropna(subset=["yoy_change"])
                   .groupby(["Cadena", "Season"])["yoy_change"]
                   .mean()
    )

    return avg_yoy.apply(lambda x: 1 if x > 0.05 else (-1 if x < -0.05 else 0)).to_dict()


def _choose_chain_szn(row, chain_szn, global_szn, chain_weeks_map):
    c = int(row["Cadena"])
    m = int(row["MonthNum"])
    if chain_weeks_map.get(c, 0) >= 26:
        val = chain_szn.get((c, m))
        if val is not None:
            return float(val)
    return float(global_szn.get(m, 1.0))


def _choose_sku_szn(row, sku_szn, chain_szn, global_szn):
    c = int(row["Cadena"])
    p = int(row["Producto"])
    m = int(row["MonthNum"])

    val = sku_szn.get((c, p, m))
    if val is not None:
        return float(val)

    return float(chain_szn.get((c, m), global_szn.get(m, 1.0)))


def _assemble_output(forecast_map, weekly_full):
    rows = []
    for (date_ts, cadena, tienda, producto), forecast_val in forecast_map.items():
        rows.append({
            "Date": date_ts,
            "Cadena": cadena,
            "Tienda": tienda,
            "Producto": producto,
            "Forecast": forecast_val,
        })

    out = pd.DataFrame(rows)
    out["Date"] = pd.to_datetime(out["Date"])

    actual = (
        weekly_full[["Date", "Cadena", "Tienda", "Producto", "WeeklyTotal"]]
        .rename(columns={"WeeklyTotal": "Actual"})
    )

    out = out.merge(actual, on=["Date", "Cadena", "Tienda", "Producto"], how="left")
    out["Actual"] = out["Actual"].fillna(0.0)

    return out


def _empty_output() -> pd.DataFrame:
    return pd.DataFrame(columns=["Date", "Cadena", "Tienda", "Producto", "Actual", "Forecast"])


def _recompute_recursive_features(weekly_full: pd.DataFrame) -> pd.DataFrame:
    keys = ["Cadena", "Tienda", "Producto"]

    df = weekly_full.sort_values(keys + ["Date"]).copy()
    s = df["SimSales"].fillna(0.0)

    # LAGS
    df["lag_1w"] = df.groupby(keys)[s.name].shift(1)
    df["lag_2w"] = df.groupby(keys)[s.name].shift(2)
    df["lag_3w"] = df.groupby(keys)[s.name].shift(3)
    df["lag_52w"] = df.groupby(keys)[s.name].shift(52)

    # ROLLING MEANS
    def _roll_mean(v, window):
        return v.shift(1).rolling(window, min_periods=1).mean()

    df["roll3w"] = (
        df.groupby(keys)[s.name]
          .apply(lambda v: _roll_mean(v, 3))
          .reset_index(level=keys, drop=True)
    )

    df["roll12w"] = (
        df.groupby(keys)[s.name]
          .apply(lambda v: _roll_mean(v, 12))
          .reset_index(level=keys, drop=True)
    )

    # AVG NONZERO 12W
    def _avg_nonzero(v):
        v = v.shift(1)
        return (
            v.rolling(12, min_periods=1)
             .apply(lambda x: x[x > 0].mean() if (x > 0).any() else 0.0, raw=False)
        )

    df["avg_nonzero_12w"] = (
        df.groupby(keys)[s.name]
          .apply(_avg_nonzero)
          .reset_index(level=keys, drop=True)
    )

    return df
