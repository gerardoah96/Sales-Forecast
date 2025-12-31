# run_weekly_forecast.py

import os
from dotenv import load_dotenv

from Testing.DB.T_database_upload import run_sim_and_push

def main() -> None:
    # Load environment variables from .env
    load_dotenv()

    # If you also want algo_id in .env, you can do:
    # algo_id = int(os.getenv("ALGO_ID", "3"))
    # For now we just hardcode it here:
    algo_id = 2

    # Run simulation and push to DB
    sim_df = run_sim_and_push(
        algo_id=algo_id,
    )

    # Optional: print a small sample for sanity check
    if sim_df is not None and not sim_df.empty:
        print("Sample of simulated data:")
        print(sim_df.head())
    else:
        print("No simulation rows returned.")

if __name__ == "__main__":
    main()
