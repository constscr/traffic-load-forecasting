from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Raw dataset configuration
RAW_DATA_FILENAME = "metro_interstate_traffic_volume.csv"
RAW_DATA_PATH = RAW_DATA_DIR / RAW_DATA_FILENAME

# Dataset schema
DATETIME_COLUMN = "date_time"
TARGET_COLUMN = "traffic_volume"

# Forecasting setup
TIME_FREQUENCY = "h"
FORECAST_HORIZON = 1

# Lag values selected for short-term, daily, and weekly traffic dependencies
LAG_HOURS = (1, 2, 3, 24, 168)

# Rolling window sizes for moving average features to capture short-term and weekly trends
ROLLING_WINDOWS = (3, 6)

# Default proportions for chronological train, validation, and test splits
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.10
TEST_SIZE = 0.20

# Baseline evaluation outputs
BASELINE_METRICS_PATH = METRICS_DIR / "baseline_model_metrics.csv"
ENSEMBLE_METRICS_PATH = METRICS_DIR / "core_ensemble_model_metrics.csv"
MODEL_COMPARISON_PATH = METRICS_DIR / "baseline_ensemble_comparison.csv"

# Time-series tuning configuration and outputs
TIME_SERIES_CV_SPLITS = 3
TUNING_N_ITER = 3
TUNING_BEST_PARAMS_PATH = METRICS_DIR / "tuning_best_parameters.csv"
TUNING_RESULTS_PATH = METRICS_DIR / "time_series_tuning_results.csv"
TUNING_COMPARISON_PATH = METRICS_DIR / "default_tuned_model_comparison.csv"

# Reproducibility
RANDOM_STATE = 42
