# Sales Forecasting System

This repository contains an end-to-end **weekly sales forecasting pipeline** built in Python, with optional database and PHP integration for production use.

The system supports **two execution modes**:

1. **Batch simulation mode** for controlled historical or future simulations  
2. **Online (production) mode** for automated weekly forecasting and database upload

---

## Repository Structure

Sales-Forecast/
└── src/
├── Sim/
│ ├── simulator.py
│ ├── Sim_Driver.py
│ ├── online_simulator.py
│ ├── modeling.py
│ ├── weekly_builder.py
│ ├── seasonality.py
│ ├── caps_and_smoothing.py
│ ├── utils.py
│ └── init.py
├── Database/
│ └── (database connectors and helpers)
├── Testing/
│ └── (testing utilities and local validation code)
├── PHP/
│ └── (PHP runner scripts for web / cron execution)
├── Deprecated/
│ └── (legacy modules)
└── REQUIREMENTS


Trained model artifacts (`Sim/Models/`) and environment files are intentionally excluded from version control.

---

## High-Level Overview

This forecasting system predicts **weekly sales at the SKU level**  
(`Cadena × Tienda × Producto`) using a combination of:

- Feature-engineered historical signals  
- Machine learning regression models  
- Rule-based reconciliation, caps, and smoothing  
- Confidence-weighted ensemble logic  

The pipeline is designed to be **dataset-agnostic** and works across multiple chains, stores, and products without hard-coded assumptions.

---

## Execution Modes

### Batch Simulator (`simulator.py`)

`simulator.py` is the **full batch simulation orchestrator**.

It is used to:
- Run forecasts over a **manually defined date range**
- Perform historical backtesting or future simulations
- Inspect intermediate outputs and model behavior

This simulator is **not called directly**.  
It is executed via:

Sim_Driver.py


#### Responsibilities

- Load historical data  
- Build weekly aggregates and features  
- Compute seasonality (global, chain, SKU, product)  
- Tune recency weighting  
- Train models on rolling monthly cutoffs  
- Recursively simulate weekly forecasts  
- Apply:
  - Model vs baseline ensemble blending
  - Last-actual anchoring
  - Confidence-weighted residual correction
  - Hierarchical reconciliation
  - YoY, SKU, and chain caps
  - Within-month smoothing  

This mode is intended for **analysis, experimentation, and validation**.

---

###  Online Simulator (`online_simulator.py`)

`online_simulator.py` is the **production-ready forecasting engine**.

It is designed to:
- Run using a single `current_date`
- Automatically align to the current week (Sunday start)
- Forecast the current week plus the next *N* weeks
- Produce clean, final forecasts ready for storage

This simulator is **not standalone**.  
It is executed by:

Uploader.py


#### Responsibilities

- Fetch raw sales data from a database  
- Run the online simulator  
- Delete existing forecasts for the target window  
- Insert newly generated forecasts into the database  

This mode is intended for **scheduled execution** (cron jobs, PHP triggers, or job runners).

---

## Database Integration

Database access is handled via the `Database/` module.

- Raw sales data is fetched using a configurable SQL query  
- Forecasts are written back to a target table  
- The uploader script accepts:
  - Database host, name, user, and password
  - Forecast table name
  - Algorithm ID
  - Source SQL query  

This design allows seamless integration with existing transactional databases or data warehouses.

---

## Testing

The `Testing/` directory contains:
- Local test scripts  
- Dataset validation helpers  
- Development-time sanity checks  

It is used to validate changes before running large simulations or production uploads.

---

## Requirements

Install dependencies with:

```bash
pip install numpy pandas scikit-learn joblib xgboost mysql-connector-python python-dotenv

These dependencies provide:

    Numerical computation (numpy)

    Data manipulation (pandas)

    Machine learning models (scikit-learn, xgboost)

    Model persistence (joblib)

    Database connectivity (mysql-connector-python)

    Environment configuration (python-dotenv)

Example Usage
Batch Simulation

python src/Sim/Sim_Driver.py

Runs the full simulator using dates defined inside the driver script.
Online Forecast + Upload

python src/Sim/Uploader.py \
  --current_date 2025-12-01 \
  --db_host 127.0.0.1 \
  --db_name SALES_DB \
  --db_user sales_user \
  --db_pass sales_pass \
  --forecast_table forecast_weekly \
  --algo_id 42 \
  --source_query "SELECT Date, Cadena, Tienda, Producto, Cantidad FROM sales_table"

This process:

    Runs the online simulator

    Removes existing forecasts for the affected weeks

    Inserts updated forecast rows into the database

PHP Integration

A PHP runner is included to support:

    Web-based execution

    Cron-based scheduling

    Operational monitoring

The PHP script:

    Computes the as-of date

    Calls the Python uploader

    Displays standard output and error logs for debugging
