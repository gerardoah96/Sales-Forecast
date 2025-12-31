"""
online_simulator.py

"Online" version of simulator.py designed for production runs (PHP-triggered).

Goal:
- Match simulator.py behavior as closely as possible.
- Given a current_date, run the weekly rolling simulation starting at the
  Sunday-aligned week containing current_date, and produce forecasts for:
    current week + horizon_weeks ahead (inclusive of current week).
  Example: horizon_weeks=4 -> 5 week-starts total.

Key differences vs simulator.py:
- Uses DB SQL_fetcher() directly (no CSV).
- Outputs are formatted for PHP (Date as int YYYYMMDD).
- Forecast window is "this week + 4 weeks ahead" instead of hard-coded 2025 window.

Important safety:
- Caps/smoothing/reconciliation apply FUTURE-ONLY relative to last_actual_date.
- Anchoring uses LAST NONZERO actual per SKU to avoid "first forecast week cliff".
- Optional DB delete+insert is supported via caller-defined SQL.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Sequence

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

from Database.database import SQL_fetcher  # your generic source DB extractor
from Database.database_upload import push_forecast_to_db  # caller-defined SQL uploader


# ==============================================================================
# PUBLIC ENTRYPOINT (RETURN DF ONLY)
# ==============================================================================
def get_weekly_forecast_for_date(
    current_date: str | datetime,
    db_host: str,
    db_name: str,
    db_user: str,
    db_password: str,
    source_query: str,
    horizon_weeks: int = 4,
    return_date_as_int: bool = True,
) -> pd.DataFrame:
    """
    PHP entrypoint.

    Runs the simulator starting from the Sunday-aligned week that contains
    current_date, generating forecasts for:
        [week_start, week_start + horizon_weeks weeks]  (inclusive)

    Returns columns:
        Date, Cadena, Tienda, Producto, Actual, Forecast, Pred_Model, Pred_Base
    """

    # ------------------------------------------------------------------
    # 1) Parse current_date and compute week window (Sunday aligned)
    # ------------------------------------------------------------------
    if isinstance(current_date, str):
        as_of = pd.to_datetime(current_date)
    elif isinstance(current_date, datetime):
        as_of = pd.Timestamp(current_date)
    else:
        raise TypeError("current_date must be str or datetime")

    as_of = as_of.normalize()

    # Most recent Sunday (or same day if Sunday).
    # weekday: Mon=0 ... Sun=6
    week_start = as_of - pd.Timedelta(days=(as_of.weekday() + 1) % 7)
    week_start = week_start.normalize()

    sim_start = week_start
    sim_end = week_start + pd.Timedelta(weeks=horizon_weeks)
    notify(f"[ONLINE] Window: {sim_start.date()} → {sim_end.date()} (horizon={horizon_weeks}w)")

    # ------------------------------------------------------------------
    # 2) Load raw data from DB
    # ------------------------------------------------------------------
    notify("[ONLINE] Fetching raw sales from DB...")
    df = SQL_fetcher(
        db_host=db_host,
        db_name=db_name,
        db_user=db_user,
        db_password=db_password,
        query=source_query,
    )

    if df.empty:
        notify("[ONLINE] ERROR: SQL_fetcher returned empty df.")
        return _empty_output(return_date_as_int)

    # Match simulator expectations
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Cantidad"] = (
        pd.to_numeric(df["Cantidad"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )

    # ------------------------------------------------------------------
    # 3) Determine last actual date (used for future-only safety)
    # ------------------------------------------------------------------
    last_actual_date = df.loc[df["Cantidad"] > 0, "Date"].max()
    if pd.isna(last_actual_date):
        notify("[ONLINE][WARN] No positive actual sales found — treating all rows as future for caps/smoothing.")
        last_actual_date = pd.Timestamp.min.normalize()
    else:
        last_actual_date = pd.to_datetime(last_actual_date).normalize()
        notify(f"[ONLINE] Last actual date: {last_actual_date.date()}")

    # ------------------------------------------------------------------
    # 4) Build weekly dataset + features
    # ------------------------------------------------------------------
    weekly_full, feature_cols = build_weekly_dataset(df)
    if weekly_full.empty:
        notify("[ONLINE] ERROR: weekly dataset is empty.")
        return _empty_output(return_date_as_int)

    # ------------------------------------------------------------------
    # 5) Filter valid chains (minimum pre-history before sim_start)
    # ------------------------------------------------------------------
    hist_pre = weekly_full[weekly_full["Date"] < sim_start]
    if hist_pre.empty:
        notify("[ONLINE] ERROR: No history before sim_start.")
        return _empty_output(return_date_as_int)

    chain_week_counts = (
        hist_pre.groupby("Cadena")["Week"]
        .nunique()
        .reset_index()
        .rename(columns={"Week": "Weeks"})
    )
    chain_weeks_map = {int(r["Cadena"]): int(r["Weeks"]) for _, r in chain_week_counts.iterrows()}

    valid_chains = [c for c, w in chain_weeks_map.items() if w >= MIN_CHAIN_WEEKS]
    if not valid_chains:
        notify("[ONLINE] ERROR: no chains with enough history.")
        return _empty_output(return_date_as_int)

    weekly_full = weekly_full[weekly_full["Cadena"].isin(valid_chains)].copy()
    notify(f"[ONLINE] Valid chains: {sorted(valid_chains)}")

    # Determine chains eligible for YoY caps
    chain_first_date = weekly_full.groupby("Cadena")["Date"].min().to_dict()
    yoy_threshold = sim_start - pd.Timedelta(days=365)
    chain_use_yoy_caps = {int(c): (chain_first_date[c] <= yoy_threshold) for c in valid_chains}

    # ------------------------------------------------------------------
    # 6) Ensure future weeks exist through sim_end (same style as simulator.py)
    # ------------------------------------------------------------------
    notify("[ONLINE] Ensuring weekly rows exist through sim_end...")

    all_weeks = pd.period_range(weekly_full["Week"].min(), sim_end, freq="W-SUN")

    keys = ["Cadena", "Tienda", "Producto"]
    sku_keys = weekly_full[keys].drop_duplicates()
    calendar = pd.DataFrame({"Week": all_weeks})
    full_index = sku_keys.merge(calendar, how="cross")
    full_index["Date"] = full_index["Week"].dt.to_timestamp()

    weekly_full = full_index.merge(weekly_full, on=keys + ["Week", "Date"], how="left")

    weekly_full["has_actual"] = weekly_full["WeeklyTotal"].notna()
    weekly_full["SimSales"] = weekly_full["WeeklyTotal"].copy()

    # ------------------------------------------------------------------
    # 7) Seasonality (same as simulator.py)
    # ------------------------------------------------------------------
    notify("[ONLINE] Computing seasonality indices...")

    global_szn = compute_global_seasonality_index(weekly_full, sim_start)
    chain_szn = compute_chain_seasonality_index(weekly_full, sim_start)
    sku_szn = compute_sku_seasonality_index(weekly_full, sim_start)
    prod_season = compute_product_season_index(weekly_full, sim_start)
    season_direction_map = compute_season_direction(weekly_full)

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

    month_num = weekly_full["MonthNum"].astype(float)
    weekly_full["month_chain_szn"] = month_num * weekly_full["chain_szn_index"].astype(float)
    weekly_full["month_sku_szn"] = month_num * weekly_full["sku_szn_index"].astype(float)
    weekly_full["month_prod_season"] = month_num * weekly_full["prod_season_index"].astype(float)

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

    # ------------------------------------------------------------------
    # 8) Recency λ tuning + density flags
    # ------------------------------------------------------------------
    notify("[ONLINE] Tuning recency lambda...")
    lam = tune_recency_lambda(weekly_full, feature_cols, sim_start)
    sku_dense_map = compute_sku_density_flags(weekly_full, sim_start)

    # ------------------------------------------------------------------
    # 9) Rolling weekly simulation loop (SIMULATOR-LIKE)
    # ------------------------------------------------------------------
    forecast_map_model: dict[tuple[pd.Timestamp, int, int, int], float] = {}
    forecast_map_base: dict[tuple[pd.Timestamp, int, int, int], float] = {}

    current_week = sim_start
    current_cutoff = None
    current_month_label = None
    global_model = None

    while current_week <= sim_end:
        month_start = pd.Timestamp(current_week.year, current_week.month, 1)

        # retrain first week of month if month changes (same as simulator)
        if current_cutoff is None or month_start != current_cutoff:
            month_label = month_start.strftime("%Y-%m")
            if current_month_label and month_label != current_month_label:
                notify(f"[MONTH] Finished {current_month_label}")
            notify(f"[MONTH] Starting {month_label}")
            current_month_label = month_label

            train_res = train_models_for_cutoff(
                weekly_full, feature_cols, month_start, lam, WEEKLY_MODEL_DIR
            )
            global_model = train_res["model"]
            current_cutoff = month_start

        if global_model is None:
            notify(f"[WARN] No model at cutoff {current_cutoff}, using baseline only for this step.")

        horizon_end = current_week + pd.Timedelta(weeks=horizon_weeks)

        weekly_full = _recompute_recursive_features(weekly_full)

        test = weekly_full[
            (weekly_full["Date"] >= current_week) &
            (weekly_full["Date"] < horizon_end)
        ].copy()

        test = test[(test["Date"] >= sim_start) & (test["Date"] <= sim_end)].copy()

        if not test.empty:
            pred_model, pred_base, _dense = predict_components_for_df(
                test, feature_cols, global_model, sku_dense_map
            )

            test["Pred_Model"] = pred_model
            test["Pred_Base"] = pred_base
            test["Pred_Final"] = np.clip(
                np.where(_dense, test["Pred_Model"], test["Pred_Base"]),
                0.0,
                None
            )

            upd_cols = ["Date", "Cadena", "Tienda", "Producto", "Pred_Final"]
            weekly_full = weekly_full.merge(test[upd_cols], on=["Date", "Cadena", "Tienda", "Producto"], how="left")

            mask_update = weekly_full["Pred_Final"].notna() & (weekly_full["Date"] >= current_week)
            weekly_full.loc[mask_update, "SimSales"] = weekly_full.loc[mask_update, "Pred_Final"]
            weekly_full.drop(columns=["Pred_Final"], inplace=True)

            for _, r in test.iterrows():
                key = (r["Date"], int(r["Cadena"]), int(r["Tienda"]), int(r["Producto"]))
                forecast_map_model[key] = float(r["Pred_Model"])
                forecast_map_base[key] = float(r["Pred_Base"])

        notify(f"[WEEK] Simulated week start {current_week.date()}")
        current_week += pd.Timedelta(weeks=1)

    if current_month_label:
        notify(f"[MONTH] Finished {current_month_label}")
    notify("[ONLINE] Raw simulation complete.")

    # ------------------------------------------------------------------
    # 10) Assemble output (Pred_Model / Pred_Base / Actual)
    # ------------------------------------------------------------------
    out_model = _assemble_output(forecast_map_model, weekly_full).rename(columns={"Forecast": "Pred_Model"})
    out_base = _assemble_output(forecast_map_base, weekly_full).rename(columns={"Forecast": "Pred_Base"})

    out = out_model.merge(
        out_base[["Date", "Cadena", "Tienda", "Producto", "Pred_Base"]],
        on=["Date", "Cadena", "Tienda", "Producto"],
        how="left",
    )

    out["Forecast"] = out["Pred_Model"].copy()

    # ------------------------------------------------------------------
    # 11) Ensemble blend weights (future only) — same as simulator
    # ------------------------------------------------------------------
    lookback_weeks = 8
    hist = out[(out["Date"] <= last_actual_date) & (out["Actual"] > 0)].copy()

    if not hist.empty:
        hist = hist.sort_values("Date")
        hist["abs_err_model"] = np.abs(hist["Pred_Model"] - hist["Actual"])
        hist["abs_err_base"] = np.abs(hist["Pred_Base"] - hist["Actual"])

        hist["rank"] = hist.groupby(["Cadena", "Tienda", "Producto"])["Date"].rank(method="first", ascending=False)
        histN = hist[hist["rank"] <= lookback_weeks].copy()

        sku_err = (
            histN.groupby(["Cadena", "Tienda", "Producto"], as_index=False)
                 .agg(mae_model=("abs_err_model", "mean"),
                      mae_base=("abs_err_base", "mean"))
        )
        sku_err["w_model"] = sku_err["mae_base"] / (sku_err["mae_model"] + sku_err["mae_base"] + 1e-9)
        sku_err["w_model"] = sku_err["w_model"].clip(0.10, 0.90)

        out = out.merge(sku_err[["Cadena", "Tienda", "Producto", "w_model"]],
                        on=["Cadena", "Tienda", "Producto"], how="left")
        out["w_model"] = out["w_model"].fillna(0.70)

        fut = out["Date"] > last_actual_date
        out.loc[fut, "Forecast"] = (
            out.loc[fut, "w_model"] * out.loc[fut, "Pred_Model"] +
            (1.0 - out.loc[fut, "w_model"]) * out.loc[fut, "Pred_Base"]
        ).clip(lower=0.0)

        out.drop(columns=["w_model"], inplace=True)
        notify(f"[ENS] Ensemble applied (lookback={lookback_weeks} weeks)")
    else:
        notify("[ENS] No history for ensemble weighting — skipping")

    # ------------------------------------------------------------------
    # 12) Last-actual anchoring (future only) — last NONZERO actual per SKU
    # ------------------------------------------------------------------
    anchor_weeks = 2
    hist_for_anchor = out[(out["Date"] <= last_actual_date) & (out["Actual"] > 0)].copy()

    if not hist_for_anchor.empty:
        last_nonzero_actual = (
            hist_for_anchor.sort_values("Date")
                           .groupby(["Cadena", "Tienda", "Producto"])
                           .tail(1)[["Cadena", "Tienda", "Producto", "Actual"]]
                           .rename(columns={"Actual": "Actual_last_nz"})
        )

        out = out.merge(last_nonzero_actual, on=["Cadena", "Tienda", "Producto"], how="left")

        fut = out["Date"] > last_actual_date
        week_idx = out.loc[fut].groupby(["Cadena", "Tienda", "Producto"]).cumcount()
        w_anchor = (1 - week_idx / anchor_weeks).clip(0, 1)

        anchor_ok = fut & out["Actual_last_nz"].notna() & (out["Actual_last_nz"] > 0)

        out.loc[anchor_ok, "Forecast"] = (
            w_anchor[anchor_ok] * out.loc[anchor_ok, "Actual_last_nz"] +
            (1 - w_anchor[anchor_ok]) * out.loc[anchor_ok, "Forecast"]
        )

        out.drop(columns=["Actual_last_nz"], inplace=True, errors="ignore")
        notify("[ANCHOR] Applied last-nonzero anchoring")
    else:
        notify("[ANCHOR] No nonzero history for anchoring — skipping")

    # ------------------------------------------------------------------
    # 13) Confidence-weighted residual nudging (future only)
    # ------------------------------------------------------------------
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
                .transform(lambda x: x.rolling(12, min_periods=3).apply(lambda v: (v > 0).mean()))
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

    # ------------------------------------------------------------------
    # 14) Optional: historical forecasts for caps fallback
    #     (leave None unless you implement it in your DB layer)
    # ------------------------------------------------------------------
    hist_forecasts_df = None

    # ------------------------------------------------------------------
    # 15) Reconciliation + caps + smoothing (FUTURE ONLY)
    # ------------------------------------------------------------------
    out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()

    out_hist = out[out["Date"] <= last_actual_date].copy()
    out_fut = out[out["Date"] > last_actual_date].copy()

    out_fut = reconcile_chain_week_to_sku(out_fut)
    out_fut = apply_chain_week_caps(out_fut, weekly_full, chain_use_yoy_caps)
    out_fut = apply_sku_caps(out_fut, weekly_full, chain_use_yoy_caps)

    out_fut = apply_chain_month_caps(
        out_fut,
        weekly_full,
        chain_use_yoy_caps,
        hist_forecasts_df=hist_forecasts_df,
        season_direction_map=season_direction_map,
    )
    out_fut = apply_within_month_smoothing(out_fut)

    out = pd.concat([out_hist, out_fut], ignore_index=True)

    # ------------------------------------------------------------------
    # 16) Final formatting (PHP-friendly)
    # ------------------------------------------------------------------
    out["Forecast"] = np.maximum(np.round(out["Forecast"]), 0).astype(int)
    out["Actual"] = np.maximum(np.round(out["Actual"]), 0).astype(int)

    # strict window
    out = out[(out["Date"] >= sim_start) & (out["Date"] <= sim_end)].copy()
    out = out.sort_values(["Date", "Cadena", "Tienda", "Producto"]).reset_index(drop=True)

    if return_date_as_int:
        out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y%m%d").astype(int)

    notify("[ONLINE] Finished.")
    return out


# ==============================================================================
# OPTIONAL: RUN + DELETE/INSERT FORECASTS (DB WRITE WRAPPER)
# ==============================================================================
def run_and_upload_weekly_forecast(
    *,
    current_date: str | datetime,
    db_host: str,
    db_name: str,
    db_user: str,
    db_password: str,
    source_query: str,
    horizon_weeks: int,
    insert_sql: str,
    delete_sql: Optional[str] = None,
    delete_params: Optional[Sequence] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper:
    1) generates forecast df (Date int YYYYMMDD)
    2) deletes any existing rows for the predicted window (if delete_sql provided)
    3) inserts the new rows (insert_sql required)

    IMPORTANT: delete_sql/delete_params are fully caller-defined to match your schema.
    """
    out = get_weekly_forecast_for_date(
        current_date=current_date,
        db_host=db_host,
        db_name=db_name,
        db_user=db_user,
        db_password=db_password,
        source_query=source_query,
        horizon_weeks=horizon_weeks,
        return_date_as_int=True,
    )

    if out.empty:
        notify("[ONLINE][DB] No rows produced; skipping upload.")
        return out

    # push only required cols
    df_up = out[["Date", "Cadena", "Tienda", "Producto", "Forecast"]].copy()

    push_forecast_to_db(
        df=df_up,
        insert_sql=insert_sql,
        delete_sql=delete_sql,
        delete_params=list(delete_params) if delete_params is not None else None,
        db_host=db_host,
        db_user=db_user,
        db_password=db_password,
        db_name=db_name,
    )

    return out


# ==============================================================================
# HELPERS (copied from simulator.py)
# ==============================================================================
def compute_season_direction(weekly_full: pd.DataFrame) -> dict[tuple[int, int], int]:
    """
    Learns season direction (+1 / 0 / -1) from historical YoY behavior.
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

    val = chain_szn.get((c, m), global_szn.get(m, 1.0))
    return float(val)


def _assemble_output(
    forecast_map: dict[tuple[pd.Timestamp, int, int, int], float],
    weekly_full: pd.DataFrame,
) -> pd.DataFrame:
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
    out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()

    actual = (
        weekly_full[["Date", "Cadena", "Tienda", "Producto", "WeeklyTotal"]]
        .rename(columns={"WeeklyTotal": "Actual"})
    )
    actual["Date"] = pd.to_datetime(actual["Date"]).dt.normalize()

    out = out.merge(actual, on=["Date", "Cadena", "Tienda", "Producto"], how="left")
    out["Actual"] = out["Actual"].fillna(0.0)

    return out


def _empty_output(return_date_as_int: bool) -> pd.DataFrame:
    cols = ["Date", "Cadena", "Tienda", "Producto", "Actual", "Forecast", "Pred_Model", "Pred_Base"]
    out = pd.DataFrame(columns=cols)
    if return_date_as_int:
        out["Date"] = out["Date"].astype("Int64")
    return out


def _recompute_recursive_features(weekly_full: pd.DataFrame) -> pd.DataFrame:
    keys = ["Cadena", "Tienda", "Producto"]

    df = weekly_full.sort_values(keys + ["Date"]).copy()
    s = df["SimSales"].fillna(0.0)

    df["lag_1w"] = df.groupby(keys)[s.name].shift(1)
    df["lag_2w"] = df.groupby(keys)[s.name].shift(2)
    df["lag_3w"] = df.groupby(keys)[s.name].shift(3)
    df["lag_52w"] = df.groupby(keys)[s.name].shift(52)

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
