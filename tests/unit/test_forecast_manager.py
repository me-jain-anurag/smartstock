import numpy as np
import pandas as pd
import pytest

from smartstock.forecasting.forecast_manager import ForecastManager
from smartstock.models.base import BaseForecaster
from smartstock.models.naive import NaiveForecaster


class MockForecaster(BaseForecaster):
    """Mock forecaster for testing."""

    def __init__(self, constant_val: float = 10.0):
        self.constant_val = constant_val

    def fit(self, df: pd.DataFrame) -> None:
        self.fitted = True

    def predict(self, periods: int, include_history: bool = True) -> pd.DataFrame:
        dates = pd.date_range("2026-01-01", periods=periods, freq="D")
        return pd.DataFrame({"forecast": [self.constant_val] * periods}, index=dates)


def test_manager_initialization() -> None:
    """Test standard initialization."""
    manager = ForecastManager()
    assert len(manager) == 0
    assert manager.list_models() == []
    assert repr(manager) == "ForecastManager(models=0, status=untrained)\n"


def test_add_and_get_model() -> None:
    """Test model registration and retrieval."""
    manager = ForecastManager()
    model = NaiveForecaster()
    manager.add_model("naive", model)

    assert len(manager) == 1
    assert "naive" in manager
    assert manager.get_model("naive") is model
    assert manager.list_models() == ["naive"]


def test_add_duplicate_model_raises_value_error() -> None:
    """Test that adding a duplicate model names raises ValueError."""
    manager = ForecastManager()
    manager.add_model("naive", NaiveForecaster())

    with pytest.raises(ValueError, match="already exists"):
        manager.add_model("naive", NaiveForecaster())


def test_add_invalid_model_raises_type_error() -> None:
    """Test that adding non-BaseForecaster raises TypeError."""
    manager = ForecastManager()

    with pytest.raises(TypeError, match="must be an instance of BaseForecaster"):
        manager.add_model("invalid", "not_a_model")  # type: ignore


def test_remove_model() -> None:
    """Test model removal."""
    manager = ForecastManager()
    manager.add_model("naive", NaiveForecaster())
    assert "naive" in manager

    manager.remove_model("naive")
    assert "naive" not in manager
    assert len(manager) == 0


def test_remove_nonexistent_model_raises_key_error() -> None:
    """Test removing non-existent model raises KeyError."""
    manager = ForecastManager()
    with pytest.raises(KeyError, match="not found"):
        manager.remove_model("missing")


def test_get_nonexistent_model_raises_key_error() -> None:
    """Test getting a non-existent model raises KeyError."""
    manager = ForecastManager()
    with pytest.raises(KeyError, match="not found"):
        manager.get_model("missing")


def test_calculate_metrics() -> None:
    """Test purely mathematical metrics calculation."""
    manager = ForecastManager()
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([12.0, 18.0, 28.0])

    metrics = manager._calculate_metrics(actual, predicted)

    assert metrics["mae"] == 2.0  # (2 + 2 + 2) / 3
    assert abs(metrics["rmse"] - 2.0) < 1e-6  # sqrt((4 + 4 + 4) / 3) = 2.0
    assert "mape" in metrics
    assert "r2" in metrics
    assert metrics["n_samples"] == 3


def test_calculate_metrics_empty() -> None:
    """Test calculating metrics with empty arrays."""
    manager = ForecastManager()
    metrics = manager._calculate_metrics(np.array([]), np.array([]))
    assert np.isnan(metrics["mae"])
    assert metrics["n_samples"] == 0


def test_calculate_metrics_mismatched_length() -> None:
    """Test that mismatched lengths raise ValueError."""
    manager = ForecastManager()
    with pytest.raises(ValueError, match="same length"):
        manager._calculate_metrics(np.array([1]), np.array([1, 2]))


def test_clear_methods() -> None:
    """Test clear_metrics and clear_models."""
    manager = ForecastManager()
    manager.add_model("m1", MockForecaster())

    manager.metrics["m1"] = {"rmse": 1.0}
    manager.training_data = pd.DataFrame()

    manager.clear_metrics()
    assert len(manager.metrics) == 0
    assert len(manager.models) == 1
    assert manager.training_data is not None

    manager.clear_models()
    assert len(manager.models) == 0
    assert manager.training_data is None


def test_get_best_model() -> None:
    """Test returning best model across metrics."""
    manager = ForecastManager()
    manager.metrics = {"m1": {"rmse": 10.0, "r2": 0.8}, "m2": {"rmse": 5.0, "r2": 0.9}}

    best_rmse_name, _ = manager.get_best_model("rmse")
    assert best_rmse_name == "m2"

    best_r2_name, _ = manager.get_best_model("r2")
    assert best_r2_name == "m2"
