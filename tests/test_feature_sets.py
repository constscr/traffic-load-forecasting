import pandas as pd
import pytest

from traffic_forecasting.feature_sets import (
    get_feature_set_columns,
    get_feature_set_scenarios,
    select_feature_set_columns,
    validate_feature_set_columns,
)
from traffic_forecasting.preprocessing import get_model_feature_columns


def test_feature_set_registry_contains_stage_eleven_scenarios() -> None:
    scenarios = get_feature_set_scenarios()

    assert tuple(scenarios) == (
        "temporal_calendar",
        "temporal_calendar_weather",
        "temporal_calendar_lag",
        "temporal_calendar_lag_rolling",
        "full",
    )
    assert scenarios["temporal_calendar"].feature_groups == ("cyclical", "calendar")
    assert "weather_categorical" not in scenarios["temporal_calendar_lag"].feature_groups


def test_full_feature_set_matches_existing_model_features() -> None:
    assert set(get_feature_set_columns("full")) == set(get_model_feature_columns())


def test_reduced_feature_set_excludes_weather_columns() -> None:
    columns = get_feature_set_columns("temporal_calendar_lag_rolling")

    assert "weather_main" not in columns
    assert "weather_description" not in columns
    assert "traffic_volume_lag_1" in columns
    assert "traffic_volume_rolling_mean_3" in columns


def test_feature_set_validation_reports_missing_columns() -> None:
    data = pd.DataFrame({"hour_sin": [0.0]})

    with pytest.raises(ValueError, match="Missing columns for feature set"):
        validate_feature_set_columns(data, "temporal_calendar")


def test_feature_set_selection_returns_independent_frame() -> None:
    columns = get_feature_set_columns("temporal_calendar")
    data = pd.DataFrame({column: [0.0, 1.0] for column in columns})

    selected = select_feature_set_columns(data, "temporal_calendar")
    selected.iloc[0, 0] = 99.0

    assert tuple(selected.columns) == columns
    assert data.iloc[0, 0] == 0.0


def test_unknown_feature_set_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown feature set scenario"):
        get_feature_set_columns("unknown")
