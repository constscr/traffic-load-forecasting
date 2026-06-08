from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
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

# Reproducibility
RANDOM_STATE = 42
