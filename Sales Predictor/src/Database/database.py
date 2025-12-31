# src/Database/database.py

import pandas as pd
import mysql.connector

def SQL_fetcher(
    db_host: str,
    db_name: str,
    db_user: str,
    db_password: str,
    query: str,
) -> pd.DataFrame:
    """
    Fetch historical ACTUAL sales data from DB.
    This is the canonical entry for the simulator.
    """

    if not all([db_host, db_user, db_password, db_name, query]):
        raise ValueError("Missing required parameters for SQL_fetcher().")

    conn = mysql.connector.connect(
        host=db_host,
        user=db_user,
        password=db_password,
        database=db_name,
    )

    df = pd.read_sql(query, conn)
    conn.close()

    # --- REQUIRED NORMALIZATION ---
    # Simulator expects these names
    rename_map = {
        "DateKey": "Date",
        "IND_29018": "Cantidad",   # <-- critical fix
    }
    for k, v in rename_map.items():
        if k in df.columns:
            df.rename(columns={k: v}, inplace=True)

    required = {"Date", "Cadena", "Tienda", "Producto", "Cantidad"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"SQL_fetcher missing required columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d", errors="coerce")
    df["Cantidad"] = pd.to_numeric(df["Cantidad"], errors="coerce").fillna(0.0)

    return df
