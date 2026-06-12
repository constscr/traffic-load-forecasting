from dataclasses import dataclass

import pandas as pd

from traffic_forecasting.features import get_feature_groups


@dataclass(frozen=True)
class FeatureSetScenario:
    """Describe one reproducible feature group experiment."""

    name: str
    label: str
    feature_groups: tuple[str, ...]


def get_feature_set_scenarios() -> dict[str, FeatureSetScenario]:
    """Return Stage 11 feature scenarios in increasing feature-space order."""
    return {
        "temporal_calendar": FeatureSetScenario(
            name="temporal_calendar",
            label="Temporal + calendar",
            feature_groups=("cyclical", "calendar"),
        ),
        "temporal_calendar_weather": FeatureSetScenario(
            name="temporal_calendar_weather",
            label="Temporal + calendar + weather",
            feature_groups=(
                "cyclical",
                "calendar",
                "weather_numeric",
                "weather_categorical",
            ),
        ),
        "temporal_calendar_lag": FeatureSetScenario(
            name="temporal_calendar_lag",
            label="Temporal + calendar + lag",
            feature_groups=("cyclical", "calendar", "lag"),
        ),
        "temporal_calendar_lag_rolling": FeatureSetScenario(
            name="temporal_calendar_lag_rolling",
            label="Temporal + calendar + lag + rolling",
            feature_groups=("cyclical", "calendar", "lag", "rolling"),
        ),
        "full": FeatureSetScenario(
            name="full",
            label="Full feature set",
            feature_groups=(
                "cyclical",
                "calendar",
                "weather_numeric",
                "weather_categorical",
                "lag",
                "rolling",
            ),
        ),
    }


def get_feature_set_columns(scenario_name: str) -> tuple[str, ...]:
    """Build the ordered feature columns for one scenario."""
    scenarios = get_feature_set_scenarios()
    if scenario_name not in scenarios:
        raise ValueError(f"Unknown feature set scenario: {scenario_name}")

    feature_groups = get_feature_groups()
    scenario = scenarios[scenario_name]
    return tuple(
        column for group_name in scenario.feature_groups for column in feature_groups[group_name]
    )


def validate_feature_set_columns(
    data: pd.DataFrame,
    scenario_name: str,
) -> tuple[str, ...]:
    """Validate and return the columns required by one feature scenario."""
    feature_columns = get_feature_set_columns(scenario_name)
    missing_columns = [column for column in feature_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns for feature set '{scenario_name}': {missing_columns}")
    return feature_columns


def select_feature_set_columns(
    data: pd.DataFrame,
    scenario_name: str,
) -> pd.DataFrame:
    """Select a validated feature scenario without modifying the input."""
    feature_columns = validate_feature_set_columns(data, scenario_name)
    return data.loc[:, feature_columns].copy()
