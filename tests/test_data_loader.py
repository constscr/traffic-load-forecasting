from pathlib import Path

import pandas as pd
import pytest
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from traffic_forecasting.config import DATETIME_COLUMN, TARGET_COLUMN, TIME_FREQUENCY
from traffic_forecasting.data_loader import (
    count_duplicate_timestamps,
    find_missing_timestamps,
    load_raw_data,
)


def _write_csv(tmp_path: Path, data: pd.DataFrame, filename: str = "traffic.csv") -> Path:
    file_path = tmp_path / filename
    data.to_csv(file_path, index=False)
    return file_path


def test_load_raw_data_raises_for_missing_file(tmp_path: Path) -> None:
    missing_file = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Raw dataset not found"):
        load_raw_data(missing_file)


@pytest.mark.parametrize("missing_column", [DATETIME_COLUMN, TARGET_COLUMN])
def test_load_raw_data_raises_when_required_column_is_missing(
    tmp_path: Path,
    missing_column: str,
) -> None:
    data = pd.DataFrame(
        {
            DATETIME_COLUMN: ["2026-01-01 00:00:00"],
            TARGET_COLUMN: [100],
            "temp": [270.0],
        }
    ).drop(columns=missing_column)
    file_path = _write_csv(tmp_path, data)

    with pytest.raises(ValueError, match=missing_column):
        load_raw_data(file_path)


def test_load_raw_data_converts_types_sorts_and_removes_full_duplicates(
    tmp_path: Path,
) -> None:
    data = pd.DataFrame(
        {
            DATETIME_COLUMN: [
                "2026-01-01 01:00:00",
                "2026-01-01 00:00:00",
                "2026-01-01 00:00:00",
            ],
            TARGET_COLUMN: ["200", "100", "100"],
            "temp": [271.0, 270.0, 270.0],
        }
    )
    file_path = _write_csv(tmp_path, data)

    result = load_raw_data(file_path)

    assert result.shape == (2, 3)
    assert is_datetime64_any_dtype(result[DATETIME_COLUMN])
    assert is_numeric_dtype(result[TARGET_COLUMN])
    assert result[DATETIME_COLUMN].is_monotonic_increasing
    assert result.duplicated().sum() == 0
    assert result.index.equals(pd.RangeIndex(len(result)))


def test_load_raw_data_resolves_duplicate_timestamps_by_column_type(
    tmp_path: Path,
) -> None:
    duplicate_time = "2026-01-01 00:00:00"
    data = pd.DataFrame(
        {
            DATETIME_COLUMN: [duplicate_time, duplicate_time, duplicate_time],
            TARGET_COLUMN: [100, 300, 500],
            "temp": [270.0, 274.0, 278.0],
            "clouds_all": [10, 30, 50],
            "rain_1h": [0.1, 0.7, 0.4],
            "snow_1h": [0.0, 0.2, 0.1],
            "weather_main": ["Rain", "Clouds", "Clouds"],
            "weather_description": ["", None, ""],
        }
    )
    file_path = _write_csv(tmp_path, data)

    assert (
        count_duplicate_timestamps(
            data.assign(**{DATETIME_COLUMN: pd.to_datetime(data[DATETIME_COLUMN])})
        )
        == 2
    )

    result = load_raw_data(file_path)
    row = result.iloc[0]

    assert len(result) == 1
    assert row[TARGET_COLUMN] == 300
    assert row["temp"] == 274
    assert row["clouds_all"] == 30
    assert row["rain_1h"] == pytest.approx(0.7)
    assert row["snow_1h"] == pytest.approx(0.2)
    assert row["weather_main"] == "Clouds"
    assert pd.isna(row["weather_description"])
    assert count_duplicate_timestamps(result) == 0


def test_categorical_tie_uses_first_pandas_mode_value(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            DATETIME_COLUMN: ["2026-01-01 00:00:00"] * 3,
            TARGET_COLUMN: [100, 200, 300],
            "weather_main": ["Rain", "Clouds", ""],
        }
    )
    file_path = _write_csv(tmp_path, data)

    result = load_raw_data(file_path)

    assert result.loc[0, "weather_main"] == "Clouds"


def test_load_raw_data_adds_missing_hours_without_filling_values(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            DATETIME_COLUMN: ["2026-01-01 00:00:00", "2026-01-01 02:00:00"],
            TARGET_COLUMN: [100, 300],
            "temp": [270.0, 274.0],
            "weather_main": ["Clear", "Clouds"],
        }
    )
    file_path = _write_csv(tmp_path, data)
    parsed_data = data.assign(**{DATETIME_COLUMN: pd.to_datetime(data[DATETIME_COLUMN])})

    missing_timestamps = find_missing_timestamps(parsed_data)
    result = load_raw_data(file_path)

    expected_timestamps = pd.date_range(
        "2026-01-01 00:00:00",
        "2026-01-01 02:00:00",
        freq=TIME_FREQUENCY,
        name=DATETIME_COLUMN,
    )
    expected_datetime_series = pd.Series(expected_timestamps, name=DATETIME_COLUMN)
    added_row = result.loc[result[DATETIME_COLUMN] == pd.Timestamp("2026-01-01 01:00:00")]

    assert missing_timestamps.equals(
        pd.DatetimeIndex([pd.Timestamp("2026-01-01 01:00:00")], name=DATETIME_COLUMN)
    )
    pd.testing.assert_series_equal(result[DATETIME_COLUMN], expected_datetime_series)
    assert len(added_row) == 1
    assert added_row.drop(columns=DATETIME_COLUMN).isna().all(axis=None)
    assert pd.isna(added_row.iloc[0][TARGET_COLUMN])
    assert (
        result.loc[
            result[DATETIME_COLUMN] == pd.Timestamp("2026-01-01 00:00:00"), TARGET_COLUMN
        ].iat[0]
        == 100
    )
    assert (
        result.loc[
            result[DATETIME_COLUMN] == pd.Timestamp("2026-01-01 02:00:00"), TARGET_COLUMN
        ].iat[0]
        == 300
    )
