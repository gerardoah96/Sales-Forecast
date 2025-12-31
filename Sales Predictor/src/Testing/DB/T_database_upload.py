import os
import pandas as pd
import mysql.connector
from typing import Optional

from dotenv import load_dotenv  # pip install python-dotenv

# Load .env once at module import
load_dotenv()
import os
import sys

# Add the src/ directory to sys.path so "Sim" and "Testing" can be imported
CURRENT_DIR = os.path.dirname(__file__)
SRC_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

# Import your simulation function
from Sim.simulator import run_weekly_2025_simulation

def push_forecast_to_db(
    df: pd.DataFrame,
    algo_id: int,
) -> None:
    """
    Uploads forecast rows to the DB.

    DB connection parameters are taken from environment variables:
        DB_HOST, DB_USER, DB_PASSWORD, DB_NAME

    Expects columns: Date (int yyyymmdd), Cadena, Tienda, Producto, Forecast.
    """
    if df.empty:
        print("No rows to insert.")
        return

    # Ensure basic columns exist
    required_cols = {"Date", "Cadena", "Tienda", "Producto", "Forecast"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"push_forecast_to_db: missing columns in df: {missing}")

    # Ensure plain Python ints/floats
    df = df.copy()
    df["Date"]     = pd.to_datetime(df["Date"]).dt.strftime("%Y%m%d").astype(int)
    df["Cadena"]   = df["Cadena"].astype(int)
    df["Tienda"]   = df["Tienda"].astype(int)
    df["Producto"] = df["Producto"].astype(int)

    # ---- Read DB config from environment ----
    db_host = os.getenv("TARGET_DB_HOST")
    db_user = os.getenv("TARGET_DB_USER")
    db_password = os.getenv("TARGET_DB_PASSWORD")
    db_name = os.getenv("TARGET_DB_NAME")
    table_name = os.getenv("TARGET_TABLE_NAME")

    if not all([db_host, db_user, db_password, db_name]):
        raise RuntimeError(
            "Missing one or more DB env vars: DB_HOST, DB_USER, DB_PASSWORD, DB_NAME"
        )

    conn = mysql.connector.connect(
        host=db_host,
        user=db_user,
        password=db_password,
        database=db_name,
    )
    cursor = conn.cursor()

    # Delete existing rows for those dates + algo_id
    unique_dates = sorted(df["Date"].unique().tolist())
    if unique_dates:
        placeholders = ", ".join(["%s"] * len(unique_dates))
        delete_sql = f"""
            DELETE FROM {table_name}
            WHERE DateKey IN ({placeholders})
              AND RIDIM_29187KEY = %s
        """
        delete_params = unique_dates + [int(algo_id)]
        cursor.execute(delete_sql, delete_params)
        conn.commit()
        print(
            f"Deleted existing rows in {table_name} for dates {unique_dates} "
            f"and algo_id={algo_id}."
        )

    insert_sql = f"""
        INSERT INTO {table_name} (
            DateKey,
            RIDIM_29107KEY,
            RIDIM_29135KEY,
            RIDIM_29038KEY,
            RIDIM_29039KEY,
            RIDIM_29187KEY,
            IND_29153
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    rows = []
    for _, row in df.iterrows():
        date_key_int = int(row["Date"])
        cadena_key   = int(row["Cadena"])
        tienda_key   = int(row["Tienda"])
        producto_key = int(row["Producto"])
        forecast_val = float(row.get("Forecast", 0.0))

        rows.append(
            (
                date_key_int,     # DateKey
                cadena_key,       # RIDIM_29107KEY
                None,             # RIDIM_29135KEY (Proveedor) -> NULL
                tienda_key,       # RIDIM_29038KEY
                producto_key,     # RIDIM_29039KEY
                int(algo_id),     # RIDIM_29187KEY
                forecast_val,     # IND_29153
            )
        )

    batch_size = 500
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        cursor.executemany(insert_sql, chunk)
        conn.commit()

    cursor.close()
    conn.close()
    print(f"Inserted {len(rows)} rows into {table_name} with algo_id={algo_id}.")


def run_sim_and_push(
    algo_id: int,
) -> Optional[pd.DataFrame]:
    """
    Runs the weekly 2025 simulation, then uploads ONLY the forecasts to the DB.

    DB connection values are read from the environment (.env).
    Returns the full simulation DataFrame (with Actual + Forecast) for inspection.
    """
    sim_df = run_weekly_2025_simulation()

    if sim_df.empty:
        print("Simulation returned no rows. Nothing to upload.")
        return sim_df

    # We keep Actual locally, but do NOT send it to DB.
    upload_df = sim_df[["Date", "Cadena", "Tienda", "Producto", "Forecast"]].copy()

    push_forecast_to_db(
        df=upload_df,
        algo_id=algo_id,
    )

    return sim_df
