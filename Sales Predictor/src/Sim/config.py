"""
config.py — Minimal recommended configuration for the forecasting simulator.

This file contains only the core hyperparameters and paths that should be
adjusted by the user. All modules (seasonality, modeling, caps, simulator)
depend only on the constants defined here.

Clean, production-friendly version (Option A).
"""

import os


# ==============================================================================
# PATHS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEEKLY_MODEL_DIR = os.path.join(BASE_DIR, "Models")
os.makedirs(WEEKLY_MODEL_DIR, exist_ok=True)


# ==============================================================================
# SIMULATION PARAMETERS
# ==============================================================================
MIN_CHAIN_WEEKS = 8      # minimum weekly history required to include a chain
HORIZON_WEEKS = 4        # forecast window forward per simulated week


# ==============================================================================
# RECENCY-WEIGHTING (λ-TUNING)
# ==============================================================================
RECENCY_LAM_DEFAULT = 0.03
RECENCY_LAM_CANDIDATES = [0.01, 0.02, 0.03, 0.05]


# ==============================================================================
# SEASONALITY LOOKBACK WINDOWS
# ==============================================================================
GLOBAL_SZN_LOOKBACK_MONTHS = 24
CHAIN_SZN_LOOKBACK_MONTHS  = 18
SKU_SZN_LOOKBACK_MONTHS    = 18
PRODUCT_SZN_LOOKBACK_MONTHS = 24


# ==============================================================================
# TRANSITION-SEASON FEATURE
# ==============================================================================
# Months near season boundaries where behavior tends to change more abruptly.
# Grouped as (2,3), (5,6), (8,9), (11,12).
TRANSITION_MONTHS = [2, 3, 5, 6, 8, 9, 11, 12]


# ==============================================================================
# SKU DENSITY (dense vs sparse SKU rules)
# ==============================================================================
SKUDENSE_AVG_THRESHOLD = 3.0       # minimum average weekly sales
SKUDENSE_ZERO_RATE_THRESHOLD = 0.70  # max fraction of zero weeks to still be "dense"


# ==============================================================================
# HIGH-VOLUME GUARDRAILS (for dense SKUs in transition months)
# ==============================================================================
# If a SKU has avg / rolling weekly volume above this, it is considered "high volume".
HIGH_VOLUME_THRESHOLD = 200.0  # tweak if needed

# In transition months, high-volume dense SKUs are constrained so that
# model predictions cannot move too far away from the baseline.
GUARD_MAX_RATIO = 1.4  # at most 1.4x baseline
GUARD_MIN_RATIO = 0.6  # at least 0.6x baseline (if baseline > 0)


# ==============================================================================
# EXTRA RECENCY BOOST FOR HIGH-VOLUME SKUs
# ==============================================================================
# For high-volume SKUs, we increase the effective recency lambda so that
# older data decays faster. Example: 0.5 → 50% stronger recency.
HIGH_VOLUME_LAM_BOOST = 0.5


# ==============================================================================
# RECENCY LAMBDA POST-BIAS
# ==============================================================================
# After tuning lambda, we slightly bias it upwards (stronger recency)
# but cap it so it doesn't get crazy.
RECENCY_LAM_POST_BIAS = 1.2   # 20% stronger than tuned value
RECENCY_LAM_MAX = 0.08        # hard cap on effective lambda


# ==============================================================================
# CHAIN TIERING THRESHOLDS (low / mid / high)
# ==============================================================================
TIER_LOW_THRESHOLD  = 50_000.0
TIER_MID_THRESHOLD  = 100_000.0
# high-tier = ≥ mid_threshold


# ==============================================================================
# YOY CHAIN & SKU CAPS
# ==============================================================================
CHAIN_YOY_MAX_RATIO = 2.00   # chain total cannot exceed 2× last year for same week

# Adaptive SKU cap tiers (upper bound multiplier)
SKU_CAP_TIERS = {
    "<20": 2.5,
    "<100": 2.0,
    "<1000": 1.5,
    "default": 1.2,
}

# ==============================================================================
# CHAIN-MONTH CAPS
# ==============================================================================
MONTH_MAX_RATIO_DEFAULT   = 1.15   # regular chains
MONTH_MAX_RATIO_HIGHCHAIN = 1.05   # high-volume chains (tighter control)


# ==============================================================================
# WITHIN-MONTH SMOOTHING (Option C+)
# ==============================================================================
SMOOTH_ALPHA = 0.60  # 0=no smoothing, 1=no smoothing; 0.6 = recommended mild smoothing

# Whether to keep all historical monthly models on disk
KEEP_ALL_MODELS = False

# If not keeping all, how many months of models to retain
MODEL_RETENTION_MONTHS = 6