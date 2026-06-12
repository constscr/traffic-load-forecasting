from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import RegressorMixin, is_regressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from traffic_forecasting.config import RANDOM_STATE
from traffic_forecasting.models import (
    build_catboost_regressor,
    build_decision_tree_regressor,
    build_dummy_regressor,
    build_gradient_boosting_regressor,
    build_knn_regressor,
    build_lgbm_regressor,
    build_linear_regression,
    build_random_forest_regressor,
    build_ridge_regressor,
    build_svr_regressor,
    build_voting_regressor,
    build_xgb_regressor,
    get_baseline_model_registry,
    get_core_ensemble_model_registry,
    get_experiment_model_registry,
    get_extended_ensemble_base_model_registry,
    get_feature_set_model_registry,
    get_hyperparameter_search_spaces,
    get_tuning_model_registry,
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


def test_core_ensemble_constructors_return_reproducible_estimators() -> None:
    random_forest = build_random_forest_regressor()
    gradient_boosting = build_gradient_boosting_regressor()
    xgboost = build_xgb_regressor()
    lightgbm = build_lgbm_regressor()
    catboost = build_catboost_regressor()

    assert isinstance(random_forest, RandomForestRegressor)
    assert isinstance(gradient_boosting, GradientBoostingRegressor)
    assert isinstance(xgboost, XGBRegressor)
    assert isinstance(lightgbm, LGBMRegressor)
    assert isinstance(catboost, CatBoostRegressor)

    assert random_forest.random_state == RANDOM_STATE
    assert gradient_boosting.random_state == RANDOM_STATE
    assert xgboost.random_state == RANDOM_STATE
    assert lightgbm.random_state == RANDOM_STATE
    assert catboost.get_param("random_seed") == RANDOM_STATE
    assert catboost.get_param("allow_writing_files") is False


def test_core_ensemble_registry_is_separate_and_returns_fresh_estimators() -> None:
    baseline_registry = get_baseline_model_registry()
    first_registry = get_core_ensemble_model_registry()
    second_registry = get_core_ensemble_model_registry()

    assert tuple(first_registry) == (
        "random_forest",
        "gradient_boosting",
        "xgboost",
        "lightgbm",
        "catboost",
    )
    assert set(baseline_registry).isdisjoint(first_registry)
    assert all(is_regressor(model) for model in first_registry.values())
    assert all(first_registry[name] is not second_registry[name] for name in first_registry)


def test_experiment_registry_combines_completed_model_groups() -> None:
    registry = get_experiment_model_registry()

    assert tuple(registry) == (
        "dummy_mean",
        "linear_regression",
        "ridge",
        "decision_tree",
        "knn",
        "svr",
        "random_forest",
        "gradient_boosting",
        "xgboost",
        "lightgbm",
        "catboost",
    )
    assert all(is_regressor(model) for model in registry.values())


def test_feature_set_registry_contains_only_selected_robustness_models() -> None:
    registry = get_feature_set_model_registry()

    assert tuple(registry) == ("catboost", "xgboost", "random_forest")
    assert all(is_regressor(model) for model in registry.values())


def test_voting_constructor_and_extended_registry() -> None:
    registry = get_extended_ensemble_base_model_registry()
    voting = build_voting_regressor([(name, model) for name, model in registry.items()])

    assert tuple(registry) == ("catboost", "xgboost", "random_forest")
    assert isinstance(voting, VotingRegressor)
    assert voting.weights is None


def test_tuning_registry_and_search_spaces_cover_primary_models() -> None:
    registry = get_tuning_model_registry()
    search_spaces = get_hyperparameter_search_spaces()

    assert tuple(registry) == (
        "ridge",
        "decision_tree",
        "random_forest",
        "gradient_boosting",
        "xgboost",
        "lightgbm",
        "catboost",
    )
    assert set(search_spaces) == set(registry)
    assert all(search_space for search_space in search_spaces.values())
    assert all(
        parameter.startswith("model__")
        for search_space in search_spaces.values()
        for parameter in search_space
    )
