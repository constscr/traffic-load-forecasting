import numpy as np
import pandas as pd
import pytest
from sklearn.utils.validation import check_is_fitted

from traffic_forecasting.config import DATETIME_COLUMN, TARGET_COLUMN
from traffic_forecasting.preprocessing import (
    EXCLUDED_MODEL_COLUMNS,
    build_preprocessor,
    get_model_feature_columns,
    get_model_feature_groups,
    prepare_model_inputs,
    select_model_features,
    split_chronologically,
    transform_model_inputs,
)


def _make_synthetic_feature_data(periods: int = 100) -> pd.DataFrame:
    feature_groups = get_model_feature_groups()
    data = pd.DataFrame(
        {
            DATETIME_COLUMN: pd.date_range("2026-01-01", periods=periods, freq="h"),
            TARGET_COLUMN: np.arange(periods, dtype=float) + 1_000,
            "holiday": [pd.NA] * periods,
        }
    )

    for offset, column in enumerate(feature_groups["continuous_numeric"]):
        data[column] = np.arange(periods, dtype=float) + offset

    data["weather_main"] = ["Clear"] * periods
    data["weather_description"] = ["clear sky"] * periods
    data["is_weekend"] = (data[DATETIME_COLUMN].dt.dayofweek >= 5).astype("int8")
    data["is_holiday"] = np.zeros(periods, dtype="int8")

    for offset, column in enumerate(feature_groups["cyclical"]):
        data[column] = np.sin(np.arange(periods) + offset)

    return data


def _as_dense(matrix: object) -> np.ndarray:
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def test_split_chronologically_uses_default_ratios_and_preserves_order() -> None:
    data = _make_synthetic_feature_data().sample(frac=1.0, random_state=42)

    train, validation, test = split_chronologically(data)

    assert [len(train), len(validation), len(test)] == [70, 10, 20]
    assert all(
        split[DATETIME_COLUMN].is_monotonic_increasing for split in (train, validation, test)
    )
    assert train[DATETIME_COLUMN].max() < validation[DATETIME_COLUMN].min()
    assert validation[DATETIME_COLUMN].max() < test[DATETIME_COLUMN].min()

    timestamp_sets = [set(split[DATETIME_COLUMN]) for split in (train, validation, test)]
    assert timestamp_sets[0].isdisjoint(timestamp_sets[1])
    assert timestamp_sets[0].isdisjoint(timestamp_sets[2])
    assert timestamp_sets[1].isdisjoint(timestamp_sets[2])


@pytest.mark.parametrize(
    ("train_size", "validation_size", "test_size"),
    [
        (0.7, 0.2, 0.2),
        (0.0, 0.5, 0.5),
    ],
)
def test_split_chronologically_rejects_invalid_ratios(
    train_size: float,
    validation_size: float,
    test_size: float,
) -> None:
    with pytest.raises(ValueError):
        split_chronologically(
            _make_synthetic_feature_data(),
            train_size=train_size,
            validation_size=validation_size,
            test_size=test_size,
        )


def test_split_chronologically_rejects_empty_subsets() -> None:
    with pytest.raises(ValueError, match="empty subset"):
        split_chronologically(_make_synthetic_feature_data(periods=5))


def test_model_feature_selection_uses_engineered_groups_and_excludes_service_columns() -> None:
    data = _make_synthetic_feature_data()

    result = select_model_features(data)
    feature_groups = get_model_feature_groups()

    assert tuple(result.columns) == get_model_feature_columns()
    assert not set(EXCLUDED_MODEL_COLUMNS) & set(result.columns)
    assert feature_groups["categorical"] == ("weather_main", "weather_description")
    assert set(feature_groups["categorical"]).issubset(result.columns)
    assert {"hour", "day_of_week", "month"}.isdisjoint(result.columns)


def test_prepare_model_inputs_separates_features_targets_and_timestamps() -> None:
    splits = split_chronologically(_make_synthetic_feature_data())

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        timestamps_train,
        timestamps_validation,
        timestamps_test,
    ) = prepare_model_inputs(*splits)

    for features in (X_train, X_validation, X_test):
        assert TARGET_COLUMN not in features
        assert DATETIME_COLUMN not in features
        assert "holiday" not in features

    for source, target, timestamps in zip(
        splits,
        (y_train, y_validation, y_test),
        (timestamps_train, timestamps_validation, timestamps_test),
        strict=True,
    ):
        pd.testing.assert_series_equal(target, source[TARGET_COLUMN])
        pd.testing.assert_series_equal(timestamps, source[DATETIME_COLUMN])


def test_preprocessor_handles_missing_values_and_unseen_categories() -> None:
    data = _make_synthetic_feature_data()
    data.loc[5, "temp"] = np.nan
    data.loc[6, "weather_description"] = np.nan
    data.loc[75, "weather_main"] = "Rain"
    data.loc[90, "weather_main"] = "Thunderstorm"

    splits = split_chronologically(data)
    X_train, X_validation, X_test, *_ = prepare_model_inputs(*splits)
    preprocessor = build_preprocessor()

    train_transformed = preprocessor.fit_transform(X_train)
    validation_transformed = preprocessor.transform(X_validation)
    test_transformed = preprocessor.transform(X_test)

    assert train_transformed.shape[0] == len(X_train)
    assert validation_transformed.shape[0] == len(X_validation)
    assert test_transformed.shape[0] == len(X_test)
    assert validation_transformed.shape[1] == train_transformed.shape[1]
    assert test_transformed.shape[1] == train_transformed.shape[1]
    for transformed in (train_transformed, validation_transformed, test_transformed):
        assert not np.isnan(_as_dense(transformed).astype(float)).any()


def test_preprocessor_statistics_and_categories_are_learned_from_train_only() -> None:
    data = _make_synthetic_feature_data()
    data.loc[70:, "temp"] = 100_000
    data.loc[70:, "weather_main"] = "Unseen weather"

    splits = split_chronologically(data)
    X_train, X_validation, X_test, *_ = prepare_model_inputs(*splits)
    (
        _,
        _,
        _,
        preprocessor,
    ) = transform_model_inputs(X_train, X_validation, X_test)

    continuous_columns = get_model_feature_groups()["continuous_numeric"]
    temp_index = continuous_columns.index("temp")
    scaler = preprocessor.named_transformers_["continuous"].named_steps["scaler"]
    encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]

    assert scaler.mean_[temp_index] == pytest.approx(X_train["temp"].mean())
    assert scaler.mean_[temp_index] != pytest.approx(data["temp"].mean())
    assert "Unseen weather" not in encoder.categories_[0]
    check_is_fitted(preprocessor)


def test_binary_and_cyclical_features_are_passed_through_without_scaling() -> None:
    data = _make_synthetic_feature_data()
    splits = split_chronologically(data)
    X_train, X_validation, X_test, *_ = prepare_model_inputs(*splits)

    train_transformed, _, _, preprocessor = transform_model_inputs(
        X_train,
        X_validation,
        X_test,
    )
    dense_train = _as_dense(train_transformed)

    binary_slice = preprocessor.output_indices_["binary"]
    cyclical_slice = preprocessor.output_indices_["cyclical"]
    np.testing.assert_allclose(
        dense_train[:, binary_slice],
        X_train.loc[:, get_model_feature_groups()["binary"]].to_numpy(),
    )
    np.testing.assert_allclose(
        dense_train[:, cyclical_slice],
        X_train.loc[:, get_model_feature_groups()["cyclical"]].to_numpy(),
    )


def test_select_model_features_rejects_missing_model_columns() -> None:
    data = _make_synthetic_feature_data().drop(columns=["temp"])

    with pytest.raises(ValueError, match="Missing model feature columns"):
        select_model_features(data)


def test_lag_and_rolling_features_are_routed_to_continuous_pipeline() -> None:
    feature_groups = get_model_feature_groups()
    model_columns = get_model_feature_columns()

    continuous_columns = feature_groups["continuous_numeric"]

    lag_columns = [column for column in model_columns if "_lag_" in column]
    rolling_columns = [column for column in model_columns if "_rolling_" in column]

    assert lag_columns
    assert rolling_columns
    assert set(lag_columns).issubset(continuous_columns)
    assert set(rolling_columns).issubset(continuous_columns)
