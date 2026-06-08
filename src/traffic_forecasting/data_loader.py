from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from traffic_forecasting.config import (
    DATETIME_COLUMN,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TIME_FREQUENCY,
)
from traffic_forecasting.logging_utils import setup_logger

logger = setup_logger(__name__)

REQUIRED_COLUMNS = {DATETIME_COLUMN, TARGET_COLUMN}
MAX_AGGREGATION_COLUMNS = {"rain_1h", "snow_1h"}


def load_raw_data(file_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw traffic volume dataset and prepare it for feature engineering."""
    if not file_path.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {file_path}")

    data = pd.read_csv(file_path)

    # The timestamp and target columns define the forecasting task.
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Raw dataset is missing required columns: {missing}")

    data[DATETIME_COLUMN] = pd.to_datetime(data[DATETIME_COLUMN], errors="raise")
    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="raise")

    initial_rows = len(data)
    data = data.drop_duplicates()
    removed_full_duplicates = initial_rows - len(data)

    data = data.sort_values(DATETIME_COLUMN).reset_index(drop=True)

    # Resolve timestamp collisions before checking the regular hourly grid.
    duplicate_timestamp_rows = count_duplicate_timestamps(data)
    if duplicate_timestamp_rows:
        data = resolve_duplicate_timestamps(data)

    # Keep missing values explicit; imputation is handled in later pipeline stages.
    missing_timestamps = find_missing_timestamps(data)
    if len(missing_timestamps) > 0:
        data = make_hourly_time_grid(data)

    data = data.reset_index(drop=True)

    logger.info("Loaded raw dataset from: %s", file_path)
    logger.info("Removed full duplicate rows: %s", removed_full_duplicates)
    logger.info("Resolved duplicate timestamp rows: %s", duplicate_timestamp_rows)
    logger.info("Added missing hourly timestamps: %s", len(missing_timestamps))
    logger.info("Dataset dimensions after preparation: %s rows, %s columns", *data.shape)

    return data


def count_duplicate_timestamps(data: pd.DataFrame) -> int:
    """Count rows that repeat an already observed timestamp."""
    return int(data.duplicated(subset=[DATETIME_COLUMN]).sum())


def resolve_duplicate_timestamps(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate rows sharing the same timestamp using deterministic rules."""
    # Multiple weather descriptions may refer to the same traffic observation hour.
    return (
        data.groupby(DATETIME_COLUMN, as_index=False, sort=True)
        .agg(_build_timestamp_aggregation_rules(data))
        .sort_values(DATETIME_COLUMN)
        .reset_index(drop=True)
    )


def find_missing_timestamps(data: pd.DataFrame) -> pd.DatetimeIndex:
    """Find missing timestamps in the expected hourly time grid."""
    if data.empty:
        return pd.DatetimeIndex([], name=DATETIME_COLUMN)

    timestamps = pd.DatetimeIndex(data[DATETIME_COLUMN])
    expected_timestamps = pd.date_range(
        start=timestamps.min(),
        end=timestamps.max(),
        freq=TIME_FREQUENCY,
        name=DATETIME_COLUMN,
    )

    return expected_timestamps.difference(timestamps)


def make_hourly_time_grid(data: pd.DataFrame) -> pd.DataFrame:
    """Reindex observations to a regular hourly grid without filling missing values."""
    if data.empty:
        return data.copy()

    data = data.sort_values(DATETIME_COLUMN).set_index(DATETIME_COLUMN)

    full_time_grid = pd.date_range(
        start=data.index.min(),
        end=data.index.max(),
        freq=TIME_FREQUENCY,
        name=DATETIME_COLUMN,
    )

    # Missing hours remain empty so later stages can choose an explicit strategy.
    return data.reindex(full_time_grid).rename_axis(DATETIME_COLUMN).reset_index()


def _build_timestamp_aggregation_rules(data: pd.DataFrame) -> dict[str, object]:
    """Build aggregation rules for duplicate timestamp resolution."""
    rules: dict[str, object] = {}

    for column in data.columns:
        if column == DATETIME_COLUMN:
            continue

        if column in MAX_AGGREGATION_COLUMNS:
            # Retain the strongest precipitation report for each observation hour.
            rules[column] = "max"
        elif is_numeric_dtype(data[column]):
            rules[column] = "median"
        else:
            rules[column] = _most_frequent_value

    return rules


def _most_frequent_value(values: pd.Series) -> object:
    """Return the first most frequent non-empty value."""
    non_missing_values = values.dropna()

    if non_missing_values.empty:
        return pd.NA

    non_empty_values = non_missing_values[non_missing_values.astype("string").str.strip().ne("")]

    if non_empty_values.empty:
        return pd.NA

    modes = non_empty_values.mode()

    if modes.empty:
        return pd.NA

    return modes.iloc[0]
