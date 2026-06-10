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
