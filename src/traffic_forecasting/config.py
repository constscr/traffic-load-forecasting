from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RAW_DATA_FILENAME = "metro_interstate_traffic_volume.csv"
RAW_DATA_PATH = RAW_DATA_DIR / RAW_DATA_FILENAME

DATETIME_COLUMN = "date_time"
TARGET_COLUMN = "traffic_volume"

RANDOM_STATE = 42
