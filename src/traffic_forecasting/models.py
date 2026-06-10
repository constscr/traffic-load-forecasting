from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from traffic_forecasting.config import RANDOM_STATE


def build_dummy_regressor() -> RegressorMixin:
    """Build the mean-prediction naive baseline."""
    return DummyRegressor(strategy="mean")


def build_linear_regression() -> RegressorMixin:
    """Build an ordinary least-squares baseline."""
    return LinearRegression()


def build_ridge_regressor() -> RegressorMixin:
    """Build an L2-regularized linear baseline."""
    return Ridge()


def build_decision_tree_regressor() -> RegressorMixin:
    """Build a reproducible decision-tree baseline."""
    return DecisionTreeRegressor(random_state=RANDOM_STATE)


def build_knn_regressor() -> RegressorMixin:
    """Build a k-nearest-neighbors regression baseline."""
    return KNeighborsRegressor()


def build_svr_regressor() -> RegressorMixin:
    """Build a support-vector regression baseline."""
    return SVR()


def build_random_forest_regressor() -> RegressorMixin:
    """Build a reproducible random-forest ensemble."""
    return RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_gradient_boosting_regressor() -> RegressorMixin:
    """Build a reproducible gradient-boosting ensemble."""
    return GradientBoostingRegressor(random_state=RANDOM_STATE)


def build_xgb_regressor() -> RegressorMixin:
    """Build a reproducible XGBoost regressor."""
    return XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )


def build_lgbm_regressor() -> RegressorMixin:
    """Build a reproducible LightGBM regressor."""
    return LGBMRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )


def build_catboost_regressor() -> RegressorMixin:
    """Build a reproducible CatBoost regressor without local training files."""
    return CatBoostRegressor(
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )


def get_baseline_model_registry() -> dict[str, RegressorMixin]:
    """Return fresh baseline model instances in comparison order."""
    return {
        "dummy_mean": build_dummy_regressor(),
        "linear_regression": build_linear_regression(),
        "ridge": build_ridge_regressor(),
        "decision_tree": build_decision_tree_regressor(),
        "knn": build_knn_regressor(),
        "svr": build_svr_regressor(),
    }


def get_core_ensemble_model_registry() -> dict[str, RegressorMixin]:
    """Return fresh core ensemble model instances in comparison order."""
    return {
        "random_forest": build_random_forest_regressor(),
        "gradient_boosting": build_gradient_boosting_regressor(),
        "xgboost": build_xgb_regressor(),
        "lightgbm": build_lgbm_regressor(),
        "catboost": build_catboost_regressor(),
    }


def get_tuning_model_registry() -> dict[str, RegressorMixin]:
    """Return the primary models selected for time-series tuning."""
    return {
        "ridge": build_ridge_regressor(),
        "decision_tree": build_decision_tree_regressor(),
        "random_forest": build_random_forest_regressor(),
        "gradient_boosting": build_gradient_boosting_regressor(),
        "xgboost": build_xgb_regressor(),
        "lightgbm": build_lgbm_regressor(),
        "catboost": build_catboost_regressor(),
    }


def get_hyperparameter_search_spaces() -> dict[str, dict[str, list[object]]]:
    """Return compact search spaces using sklearn Pipeline parameter names."""
    return {
        "ridge": {
            "model__alpha": [0.1, 1.0, 10.0, 100.0],
        },
        "decision_tree": {
            "model__max_depth": [8, 12, 16, None],
            "model__min_samples_split": [2, 10, 20],
            "model__min_samples_leaf": [1, 5, 10],
        },
        "random_forest": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [12, 20, None],
            "model__min_samples_leaf": [1, 3, 5],
            "model__max_features": ["sqrt", 0.8, 1.0],
        },
        "gradient_boosting": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
            "model__subsample": [0.8, 1.0],
        },
        "xgboost": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [3, 6, 9],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
        },
        "lightgbm": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__num_leaves": [15, 31, 63],
            "model__max_depth": [-1, 10, 20],
        },
        "catboost": {
            "model__iterations": [100, 200, 300],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__depth": [4, 6, 8],
            "model__l2_leaf_reg": [3.0, 5.0, 10.0],
        },
    }
