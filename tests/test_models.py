from sklearn.base import RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from traffic_forecasting.config import RANDOM_STATE
from traffic_forecasting.models import (
    build_decision_tree_regressor,
    build_dummy_regressor,
    build_knn_regressor,
    build_linear_regression,
    build_ridge_regressor,
    build_svr_regressor,
    get_baseline_model_registry,
)


def test_baseline_model_constructors_return_expected_estimators() -> None:
    assert isinstance(build_dummy_regressor(), DummyRegressor)
    assert isinstance(build_linear_regression(), LinearRegression)
    assert isinstance(build_ridge_regressor(), Ridge)
    assert isinstance(build_decision_tree_regressor(), DecisionTreeRegressor)
    assert isinstance(build_knn_regressor(), KNeighborsRegressor)
    assert isinstance(build_svr_regressor(), SVR)
    assert build_decision_tree_regressor().random_state == RANDOM_STATE


def test_baseline_model_registry_contains_only_stage_six_models() -> None:
    registry = get_baseline_model_registry()

    assert tuple(registry) == (
        "dummy_mean",
        "linear_regression",
        "ridge",
        "decision_tree",
        "knn",
        "svr",
    )
    assert all(isinstance(model, RegressorMixin) for model in registry.values())


def test_baseline_model_registry_returns_fresh_estimators() -> None:
    first_registry = get_baseline_model_registry()
    second_registry = get_baseline_model_registry()

    assert all(first_registry[name] is not second_registry[name] for name in first_registry)
