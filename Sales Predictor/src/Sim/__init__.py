"""
Sim — Weekly Forecasting Simulator Package

This package contains a modular, refactored, and optimized version of the
weekly 2025 forecasting simulator.

Modules included:

    config.py          – Global constants and simulation parameters
    utils.py           – Utility helpers (notify, date/season helpers)
    data_prep.py       – Weekly aggregation, feature engineering
    seasonality.py     – Global/chain/SKU/product seasonality indices
    modeling.py        – Recency tuning, density classification, model training
                         + NEW: Automatic tier-based XGBoost tuning
    calibration.py     – Chain × season calibration (smoothed + shrunk)
    caps.py            – Chain-level, SKU-level, and month-level caps
    smoothing.py       – Monthly-aware smoothing + within-month weekly smoothing
    run_simulation.py  – Main entry point to run the end-to-end simulation

To run the simulator, use:

    from Sim.run_simulation import run_weekly_forecast
    df = run_weekly_forecast()

The final output is a DataFrame containing:

    Date, Cadena, Tienda, Producto, Actual, Forecast

This package is designed to be clean, maintainable, and extendable.
"""
