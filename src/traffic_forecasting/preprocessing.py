import math

import pandas as pd

from traffic_forecasting.config import (
    DATETIME_COLUMN,
    TARGET_COLUMN,
    TEST_SIZE,
    TRAIN_SIZE,
    VALIDATION_SIZE,
)
from traffic_forecasting.features import get_feature_groups

EXCLUDED_MODEL_COLUMNS = (DATETIME_COLUMN, TARGET_COLUMN, "holiday")


def get_model_feature_groups() -> dict[str, tuple[str, ...]]:
    """Return feature groups routed through model preprocessing."""
    feature_groups = get_feature_groups()
    return {
        "continuous_numeric": (
            *feature_groups["weather_numeric"],
            *feature_groups["lag"],
            *feature_groups["rolling"],
        ),
        "categorical": feature_groups["weather_categorical"],
        "binary": feature_groups["calendar"],
        "cyclical": feature_groups["cyclical"],
    }


def get_model_feature_columns() -> tuple[str, ...]:
    """Return model feature columns in deterministic transformer order."""
    feature_groups = get_model_feature_groups()
    feature_columns = tuple(column for columns in feature_groups.values() for column in columns)

    duplicated_columns = sorted(
        {column for column in feature_columns if feature_columns.count(column) > 1}
    )
    if duplicated_columns:
        raise ValueError(f"Duplicated model feature columns: {duplicated_columns}")

    excluded_columns = sorted(set(feature_columns) & set(EXCLUDED_MODEL_COLUMNS))
    if excluded_columns:
        raise ValueError(f"Service or target columns cannot be model features: {excluded_columns}")

    return feature_columns


def select_model_features(data: pd.DataFrame) -> pd.DataFrame:
    """Select engineered predictors while excluding service and target columns."""
    feature_columns = get_model_feature_columns()
    missing_columns = [column for column in feature_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing model feature columns: {missing_columns}")

    return data.loc[:, feature_columns].copy()


def split_chronologically(
    data: pd.DataFrame,
    train_size: float = TRAIN_SIZE,
    validation_size: float = VALIDATION_SIZE,
    test_size: float = TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into ordered train, validation, and test subsets."""
    split_sizes = (train_size, validation_size, test_size)

    if any(size <= 0 or size >= 1 for size in split_sizes):
        raise ValueError("Split ratios must be greater than 0 and less than 1.")

    if not math.isclose(sum(split_sizes), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Split ratios must sum to 1.")

    if DATETIME_COLUMN not in data.columns:
        raise ValueError(f"Missing datetime column: {DATETIME_COLUMN}")

    if data[DATETIME_COLUMN].isna().any():
        raise ValueError(f"Datetime column contains missing values: {DATETIME_COLUMN}")

    if not pd.api.types.is_datetime64_any_dtype(data[DATETIME_COLUMN]):
        raise TypeError(f"Datetime column must have datetime dtype: {DATETIME_COLUMN}")

    ordered_data = data.sort_values(DATETIME_COLUMN).reset_index(drop=True).copy()

    train_end = int(len(ordered_data) * train_size)
    validation_end = train_end + int(len(ordered_data) * validation_size)

    train = ordered_data.iloc[:train_end].reset_index(drop=True)
    validation = ordered_data.iloc[train_end:validation_end].reset_index(drop=True)
    test = ordered_data.iloc[validation_end:].reset_index(drop=True)

    if any(split.empty for split in (train, validation, test)):
        raise ValueError("Chronological splitting produced an empty subset.")

    if train[DATETIME_COLUMN].max() >= validation[DATETIME_COLUMN].min():
        raise ValueError("Train and validation timestamps must be strictly ordered.")

    if validation[DATETIME_COLUMN].max() >= test[DATETIME_COLUMN].min():
        raise ValueError("Validation and test timestamps must be strictly ordered.")

    return train, validation, test


def _prepare_single_split(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    required_columns = (DATETIME_COLUMN, TARGET_COLUMN)
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if not data[DATETIME_COLUMN].is_monotonic_increasing:
        raise ValueError("Input split must be sorted chronologically.")

    features = select_model_features(data)

    forbidden_columns = sorted(set(features.columns) & set(EXCLUDED_MODEL_COLUMNS))
    if forbidden_columns:
        raise ValueError(f"Forbidden columns found in model features: {forbidden_columns}")

    target = data[TARGET_COLUMN].copy()
    timestamps = data[DATETIME_COLUMN].copy()
    return features, target, timestamps


def prepare_model_inputs(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Separate model features, targets, and timestamps for each split."""
    X_train, y_train, timestamps_train = _prepare_single_split(train)
    X_validation, y_validation, timestamps_validation = _prepare_single_split(validation)
    X_test, y_test, timestamps_test = _prepare_single_split(test)

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        timestamps_train,
        timestamps_validation,
        timestamps_test,
    )
