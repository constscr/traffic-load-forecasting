import pandas as pd

from traffic_forecasting.config import DATETIME_COLUMN, TARGET_COLUMN
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
