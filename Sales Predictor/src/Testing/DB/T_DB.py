from dotenv import load_dotenv
import os
import pandas as pd
import mysql.connector

# ------------------------------------------------------------------
# Safe notify
# ------------------------------------------------------------------
try:
    from Sim.utils import notify
except Exception:
    def notify(msg: str):
        print(msg, flush=True)

load_dotenv()

# ------------------------------------------------------------------
# DB CONNECTION
# ------------------------------------------------------------------
def _connect():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "192.168.0.40"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", "3306")),
        connection_timeout=15,
        auth_plugin="mysql_native_password",
        use_pure=True,
    )

# ------------------------------------------------------------------
# ACTUAL SALES FETCHER (REQUIRED – FROM DB)
# ------------------------------------------------------------------
def SQL_fetcher() -> pd.DataFrame:
    """
    Fetch historical ACTUAL sales data.
    REQUIRED for simulator.
    Must return column: Cantidad
    """

    notify("[DB] Fetching historical actuals...")

    query = """
        SELECT
            DateKey        AS Date,
            RIDIM_29107KEY AS Cadena,
            RIDIM_29038KEY AS Tienda,
            RIDIM_29039KEY AS Producto,
            IND_29018      AS Cantidad
        FROM rifact_29003
    """

    try:
        conn = _connect()
        df = pd.read_sql(query, conn)
        conn.close()
    except Exception as e:
        notify(f"[DB][ERROR] Failed to fetch actuals: {e}")
        return pd.DataFrame()

    if df.empty:
        notify("[DB][ERROR] SQL_fetcher returned EMPTY actuals")
        return df

    # Normalize
    df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d")
    df["Cantidad"] = pd.to_numeric(df["Cantidad"], errors="coerce").fillna(0.0)

    notify(f"[DB] Loaded {len(df)} actual rows")
    return df

# ------------------------------------------------------------------
# HISTORICAL FORECAST FETCHER (CSV – TESTING MODE)
# ------------------------------------------------------------------
def fetch_historical_forecasts(cutoff_run_date: pd.Timestamp) -> pd.DataFrame:
    """
    Load historical FORECASTS from CSV (testing mode).
    Used ONLY for caps fallback.
    Safe if empty or missing.
    """

    path = "forecast_history.csv"

    if not os.path.exists(path):
        notify("[DB][INFO] forecast_history.csv not found — using safety net caps")
        return _empty_df()

    try:
        df = pd.read_csv(path)
    except Exception as e:
        notify(f"[DB][WARN] Failed to read forecast history CSV: {e}")
        return _empty_df()

    if df.empty:
        notify("[DB][INFO] Forecast history CSV empty — using safety net caps")
        return _empty_df()

    # Normalize dates (DATE ONLY)
    df["Date"] = pd.to_datetime(
        df["Date"], format="mixed", errors="coerce"
    ).dt.date

    df["forecast_run_date"] = pd.to_datetime(
        df["forecast_run_date"], format="mixed", errors="coerce"
    ).dt.date

    cutoff = pd.to_datetime(cutoff_run_date).date()
    df = df[df["forecast_run_date"] < cutoff]

    if df.empty:
        notify("[DB][INFO] No historical forecasts before cutoff — safety net caps")
        return _empty_df()

    df["Forecast"] = pd.to_numeric(df["Forecast"], errors="coerce").fillna(0.0)
    df["Actual"]   = pd.to_numeric(df["Actual"], errors="coerce").fillna(0.0)

    notify(f"[DB] Loaded {len(df)} historical forecast rows (CSV)")
    return df

# ------------------------------------------------------------------
# EMPTY STRUCTURE (SAFETY NET)
# ------------------------------------------------------------------
def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Date",
            "Cadena",
            "Tienda",
            "Producto",
            "Actual",
            "Forecast",
        ]
    )
