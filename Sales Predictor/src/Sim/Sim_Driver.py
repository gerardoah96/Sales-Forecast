import pandas as pd
import os
from datetime import datetime
from Sim.simulator import run_weekly_2025_simulation

# --------------------------------------------------
# Run simulation
# --------------------------------------------------
df = run_weekly_2025_simulation()

# --------------------------------------------------
# Save weekly output (current behavior)
# --------------------------------------------------
weekly_path = "Weekly_2025_Forecast.csv"
df.to_csv(weekly_path, index=False)
print(f"Saved {weekly_path}")

# --------------------------------------------------
# Append to forecast history (NEW – fallback source)
# --------------------------------------------------
history_path = "forecast_history.csv"

df_hist = df.copy()
df_hist["forecast_run_date"] = pd.Timestamp.today().normalize()

if os.path.exists(history_path):
    prev = pd.read_csv(history_path)
    prev["forecast_run_date"] = pd.to_datetime(prev["forecast_run_date"])
    df_hist = pd.concat([prev, df_hist], ignore_index=True)

df_hist.to_csv(history_path, index=False)
print(f"[LOCAL] Forecast history updated → {history_path}")
