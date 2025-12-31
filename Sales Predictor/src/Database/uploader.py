import os
import sys
import argparse

# Make sure src/ is on path if this file is in src/Trainer or similar
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from database_upload import push_forecast_to_db
from Algo.Algo import get_weekly_forecast as forecast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run weekly forecast and upload results to DB."
    )

    parser.add_argument("--db_host", required=True, help="Database host/IP")
    parser.add_argument("--db_name", required=True, help="Database name")
    parser.add_argument("--db_user", required=True, help="Database user")
    parser.add_argument("--db_pass", required=True, help="Database password")
    parser.add_argument("--table",   required=True, help="Target table for forecast")
    parser.add_argument("--algo_id", type=int, required=True, help="Algorithm ID (int)")

    # Query string to be passed down to the weekly model / SQL_fetcher
    parser.add_argument("--query", required=True, help="SQL query to fetch sales data")

    return parser.parse_args()


def main():
    args = parse_args()

    # Get forecast DataFrame from weekly model
    df = forecast(
        db_host=args.db_host,
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_pass,
        query=args.query,
    )

    if df is None or df.empty:
        print("No forecast generated; nothing to upload.")
        return

    # Push forecast to DB
    push_forecast_to_db(
        df=df,
        db_host=args.db_host,
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_pass,
        table_name=args.table,
        algo_id=args.algo_id,
    )

    print("Forecast uploaded successfully.")


if __name__ == "__main__":
    main()
