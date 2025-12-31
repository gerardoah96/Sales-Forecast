import pandas as pd
import mysql.connector


def push_forecast_to_db(
    df: pd.DataFrame,
    insert_sql: str,
    delete_sql: str | None,
    delete_params: list | None,
    db_host: str,
    db_user: str,
    db_password: str,
    db_name: str,
):
    """
    Push forecast rows to DB using fully caller-defined SQL.

    REQUIRED df columns:
        Date        (YYYYMMDD int or datetime)
        Cadena
        Tienda
        Producto
        Forecast

    SQL CONTRACT:
        - insert_sql: parameterized SQL with %s placeholders
        - delete_sql: optional parameterized SQL
        - delete_params: list of params for delete_sql (caller-defined)
    """

    if df is None or df.empty:
        print("[DB] No rows to insert.")
        return

    required = {"Date", "Cadena", "Tienda", "Producto", "Forecast"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"push_forecast_to_db: missing columns {sorted(missing)}")

    df = df.copy()

    # --------------------------------------------------
    # Normalize types
    # --------------------------------------------------
    if not pd.api.types.is_integer_dtype(df["Date"]):
        df["Date"] = (
            pd.to_datetime(df["Date"], errors="coerce")
              .dt.strftime("%Y%m%d")
              .astype(int)
        )

    df["Cadena"]   = df["Cadena"].astype(int)
    df["Tienda"]   = df["Tienda"].astype(int)
    df["Producto"] = df["Producto"].astype(int)
    df["Forecast"] = pd.to_numeric(df["Forecast"], errors="coerce").fillna(0.0)

    conn = mysql.connector.connect(
        host=db_host,
        user=db_user,
        password=db_password,
        database=db_name,
    )
    cursor = conn.cursor()

    # --------------------------------------------------
    # DELETE (fully caller-controlled)
    # --------------------------------------------------
    if delete_sql and delete_params:
        cursor.execute(delete_sql, delete_params)
        conn.commit()
        print("[DB] Existing forecast rows deleted.")

    # --------------------------------------------------
    # INSERT
    # --------------------------------------------------
    rows = [
        (
            int(r.Date),
            int(r.Cadena),
            int(r.Tienda),
            int(r.Producto),
            float(r.Forecast),
        )
        for r in df.itertuples(index=False)
    ]

    batch_size = 500
    for i in range(0, len(rows), batch_size):
        cursor.executemany(insert_sql, rows[i:i + batch_size])
        conn.commit()

    cursor.close()
    conn.close()

    print(f"[DB] Inserted {len(rows)} forecast rows.")
