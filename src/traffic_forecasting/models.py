from sklearn.base import RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

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
