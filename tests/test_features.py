import numpy as np
import pandas as pd
import pytest

from traffic_forecasting.config import (
    DATETIME_COLUMN,
    LAG_HOURS,
    ROLLING_WINDOWS,
    TARGET_COLUMN,
)
from traffic_forecasting.features import (
    HISTORICAL_FEATURE_COLUMNS,
    LAG_FEATURE_COLUMNS,
    ROLLING_FEATURE_COLUMNS,
    add_calendar_features,
    add_cyclical_features,
    add_lag_features,
    add_rolling_features,
    add_time_features,
    add_weather_features,
    build_feature_dataset,
    drop_incomplete_feature_rows,
)


def _make_hourly_data(periods: int = 200) -> pd.DataFrame:
    return pd.DataFrame(
        {
            DATETIME_COLUMN: pd.date_range(
                "2026-01-01",
                periods=periods,
                freq="h",
            ),
            "holiday": [pd.NA] * periods,
            "temp": np.arange(periods, dtype=float) + 273.15,
            "rain_1h": np.zeros(periods),
            "snow_1h": np.zeros(periods),
            "clouds_all": np.full(periods, 25.0),
            "weather_main": ["Clear"] * periods,
            "weather_description": ["clear sky"] * periods,
            TARGET_COLUMN: np.arange(1, periods + 1, dtype=float),
        }
    )


def test_add_time_and_calendar_features() -> None:
    data = pd.DataFrame(
        {
            DATETIME_COLUMN: pd.to_datetime(
                [
                    "2026-01-02 23:00:00",
                    "2026-01-03 00:00:00",
                    "2026-01-04 01:00:00",
                ]
            ),
            "holiday": [pd.NA, "New Years Day", "  "],
        }
    )

    result = add_calendar_features(add_time_features(data))

    assert result["hour"].tolist() == [23, 0, 1]
    assert result["day_of_week"].tolist() == [4, 5, 6]
    assert result["month"].tolist() == [1, 1, 1]
    assert result["is_weekend"].tolist() == [0, 1, 1]
    assert result["is_holiday"].tolist() == [0, 1, 0]
    assert "hour" not in data.columns


def test_add_cyclical_features_uses_expected_periods_and_names() -> None:
    data = pd.DataFrame(
        {
            "hour": [0, 6],
            "day_of_week": [0, 1],
            "month": [1, 4],
        }
    )

    result = add_cyclical_features(data)

    expected_columns = {
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
    }

    assert expected_columns.issubset(result.columns)
    assert result.loc[0, "hour_sin"] == pytest.approx(0.0)
    assert result.loc[0, "hour_cos"] == pytest.approx(1.0)
    assert result.loc[1, "hour_sin"] == pytest.approx(1.0)
    assert result.loc[1, "day_of_week_sin"] == pytest.approx(np.sin(2 * np.pi / 7))
    assert result.loc[0, "month_sin"] == pytest.approx(0.0)
    assert result.loc[1, "month_sin"] == pytest.approx(1.0)


def test_add_weather_features_preserves_weather_columns() -> None:
    data = _make_hourly_data(periods=2)

    result = add_weather_features(data)

    assert result["temp_celsius"].tolist() == pytest.approx([0.0, 1.0])

    for column in (
        "temp",
        "rain_1h",
        "snow_1h",
        "clouds_all",
        "weather_main",
        "weather_description",
    ):
        pd.testing.assert_series_equal(result[column], data[column])


def test_add_lag_features_shifts_target_by_configured_hours() -> None:
    data = _make_hourly_data()

    result = add_lag_features(data)

    assert len(LAG_FEATURE_COLUMNS) == len(LAG_HOURS)

    for lag_hours, column in zip(
        LAG_HOURS,
        LAG_FEATURE_COLUMNS,
        strict=True,
    ):
        pd.testing.assert_series_equal(
            result[column],
            data[TARGET_COLUMN].shift(lag_hours),
            check_names=False,
        )


def test_add_rolling_features_uses_past_values_only() -> None:
    data = _make_hourly_data(periods=10)

    result = add_rolling_features(data)

    for window in ROLLING_WINDOWS:
        expected_mean = data[TARGET_COLUMN].shift(1).rolling(window).mean()
        expected_std = data[TARGET_COLUMN].shift(1).rolling(window).std(ddof=0)

        pd.testing.assert_series_equal(
            result[f"{TARGET_COLUMN}_rolling_mean_{window}"],
            expected_mean,
            check_names=False,
        )
        pd.testing.assert_series_equal(
            result[f"{TARGET_COLUMN}_rolling_std_{window}"],
            expected_std,
            check_names=False,
        )

    assert result.loc[3, f"{TARGET_COLUMN}_rolling_mean_3"] == pytest.approx(2.0)


def test_current_target_does_not_leak_into_historical_features() -> None:
    original = _make_hourly_data(periods=200)
    changed = original.copy()
    changed.loc[180, TARGET_COLUMN] = 1_000_000

    original_features = add_rolling_features(add_lag_features(original))
    changed_features = add_rolling_features(add_lag_features(changed))

    pd.testing.assert_series_equal(
        original_features.loc[180, list(HISTORICAL_FEATURE_COLUMNS)],
        changed_features.loc[180, list(HISTORICAL_FEATURE_COLUMNS)],
    )
    assert (
        original_features.loc[181, f"{TARGET_COLUMN}_lag_1"]
        != changed_features.loc[181, f"{TARGET_COLUMN}_lag_1"]
    )


def test_missing_hour_propagates_without_using_future_values() -> None:
    data = _make_hourly_data(periods=10)
    data.loc[4, TARGET_COLUMN] = np.nan

    result = add_rolling_features(add_lag_features(data))

    assert pd.isna(result.loc[5, f"{TARGET_COLUMN}_lag_1"])
    assert pd.isna(result.loc[5, f"{TARGET_COLUMN}_rolling_mean_3"])
    assert result.loc[4, f"{TARGET_COLUMN}_lag_1"] == 4.0


def test_cleanup_runs_after_feature_creation_and_preserves_text_missing_values() -> None:
    data = _make_hourly_data()
    data.loc[175, "weather_description"] = pd.NA

    enriched = add_rolling_features(add_lag_features(data))
    cleaned = drop_incomplete_feature_rows(enriched)

    assert enriched[list(HISTORICAL_FEATURE_COLUMNS)].isna().any(axis=None)
    assert not cleaned[[TARGET_COLUMN, *HISTORICAL_FEATURE_COLUMNS]].isna().any(axis=None)
    assert cleaned[DATETIME_COLUMN].is_monotonic_increasing
    assert cleaned.index.equals(pd.RangeIndex(len(cleaned)))
    assert cleaned["weather_description"].isna().sum() == 1


def test_historical_feature_columns_include_lags_and_rolling_features() -> None:
    assert set(LAG_FEATURE_COLUMNS).issubset(HISTORICAL_FEATURE_COLUMNS)
    assert set(ROLLING_FEATURE_COLUMNS).issubset(HISTORICAL_FEATURE_COLUMNS)


def test_build_feature_dataset_keeps_target_and_expected_feature_groups() -> None:
    data = _make_hourly_data()

    result = build_feature_dataset(data)

    assert TARGET_COLUMN in result.columns
    assert set(LAG_FEATURE_COLUMNS).issubset(result.columns)
    assert set(ROLLING_FEATURE_COLUMNS).issubset(result.columns)
    assert {"weather_main", "weather_description"}.issubset(result.columns)
    assert not any(column.startswith("weather_main_") for column in result.columns)
    assert not any(column.startswith("weather_description_") for column in result.columns)
    assert not result[[TARGET_COLUMN, *HISTORICAL_FEATURE_COLUMNS]].isna().any(axis=None)
    assert result[DATETIME_COLUMN].is_monotonic_increasing
    assert result.index.equals(pd.RangeIndex(len(result)))


def test_build_feature_dataset_sorts_data_before_historical_features() -> None:
    data = _make_hourly_data()
    shuffled = data.sample(frac=1.0, random_state=42).reset_index(drop=True)

    result = build_feature_dataset(shuffled)

    assert result[DATETIME_COLUMN].is_monotonic_increasing
