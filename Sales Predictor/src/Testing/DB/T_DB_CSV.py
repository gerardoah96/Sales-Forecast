from __future__ import annotations

from pathlib import Path
import pandas as pd


def SQL_fetcher(
    csv_path: str | None = None,
    query: str | None = None,   # kept for compatibility; ignored
) -> pd.DataFrame:
    """
    CSV-backed replacement for SQL_fetcher.

    Expected minimal columns:
      - Date
      - Cadena
      - Tienda
      - Producto
      - One of: Cantidad | Quantity | Qty | Actual

    If 'Actual' exists (like your Weekly_2025_Forecast.csv), it will be used as Cantidad.
    """

    # ----------------------------
    # Resolve CSV path robustly
    # ----------------------------
    if csv_path is None:
        # Default to sales_data.csv placed next to this file
        here = Path(__file__).resolve().parent
        csv_path = str(here / "sales_data.csv")

    p = Path(csv_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")

    df = pd.read_csv(p)
    if df.empty:
        return df

    # Normalize column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    # ----------------------------
    # Validate required ID columns
    # ----------------------------
    required = ["Date", "Cadena", "Tienda", "Producto"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. Found: {list(df.columns)}")

    # ----------------------------
    # Parse Date safely
    # ----------------------------
    # Works for 'YYYY-MM-DD' and 'YYYYMMDD' and many others.
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    if df.empty:
        return df

    # ----------------------------
    # Quantity mapping
    # ----------------------------
    # Your file has 'Actual' (weekly actuals). Treat it as Cantidad.
    qty_candidates = ["Cantidad", "Quantity", "Qty", "Actual"]
    qty_col = next((c for c in qty_candidates if c in df.columns), None)

    if qty_col is None:
        raise ValueError(
            "CSV must contain a quantity column: "
            "'Cantidad' (or 'Quantity'/'Qty'/'Actual'). "
            f"Found columns: {list(df.columns)}"
        )

    df["Cantidad"] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0.0)

    # Enforce ID dtypes (optional but helps stability)
    df["Cadena"] = pd.to_numeric(df["Cadena"], errors="coerce").astype("Int64")
    df["Tienda"] = pd.to_numeric(df["Tienda"], errors="coerce").astype("Int64")
    df["Producto"] = pd.to_numeric(df["Producto"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["Cadena", "Tienda", "Producto"])
    if df.empty:
        return df

    # Convert back to int (after dropna)
    df["Cadena"] = df["Cadena"].astype(int)
    df["Tienda"] = df["Tienda"].astype(int)
    df["Producto"] = df["Producto"].astype(int)

    return df[["Date", "Cadena", "Tienda", "Producto", "Cantidad"]].copy()


def fetch_historical_forecasts(*args, **kwargs) -> pd.DataFrame:
    """
    Optional: simulator calls this for caps fallback.
    For CSV mode, return empty unless you want to load another CSV.
    """
    return pd.DataFrame(columns=["Date", "Cadena", "Tienda", "Producto", "Forecast"])
