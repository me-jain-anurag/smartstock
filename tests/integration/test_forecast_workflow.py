import numpy as np
import pandas as pd
import pytest

from smartstock.forecasting.forecast_manager import ForecastManager
from smartstock.models.naive import NaiveForecaster


def test_manager_end_to_end_workflow() -> None:
    """Test the complete life cycle of ForecastManager models."""
    manager = ForecastManager()

    manager.add_model("naive_1", NaiveForecaster())
    manager.add_model("naive_2", NaiveForecaster())

    # Generate mock training and test data
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    base_sales = np.linspace(10, 100, 10)
    train_df = pd.DataFrame({"sales": base_sales[:7]}, index=dates[:7])
    test_df = pd.DataFrame({"sales": base_sales[7:]}, index=dates[7:])

    # 1. Train all
    manager.train_all(train_df)
    assert manager.training_data is not None
    assert len(manager.training_data) == 7

    predictions = manager.predict_all(3, include_history=False)
    assert len(predictions) == 2
    assert "naive_1" in predictions
    assert "naive_2" in predictions

    # Naive model predicts the last observed value continuously
    assert (predictions["naive_1"]["forecast"] == 70.0).all()

    # 3. Compare models
    comparison = manager.compare_models(test_df, forecast_horizon=3)

    # The comparison should return a DataFrame indexed by model name
    assert isinstance(comparison, pd.DataFrame)
    assert set(comparison.index) == {"naive_1", "naive_2"}
    assert "rmse" in comparison.columns
    assert "mae" in comparison.columns
    assert "mape" in comparison.columns
    assert "r2" in comparison.columns

    # 4. Get best model
    best_name, best_metrics = manager.get_best_model("rmse")
    assert best_name in {"naive_1", "naive_2"}
    assert "rmse" in best_metrics

    # 5. Clear state
    manager.clear_models()
    assert len(manager) == 0
    assert manager.training_data is None


def test_train_all_empty_data_raises_error() -> None:
    """Test train_all with empty DataFrame raises ValueError."""
    manager = ForecastManager()
    manager.add_model("naive", NaiveForecaster())

    with pytest.raises(ValueError, match="cannot be empty"):
        manager.train_all(pd.DataFrame())


def test_train_all_missing_sales_column_raises_error() -> None:
    """Test train_all without 'sales' column raises ValueError."""
    manager = ForecastManager()
    manager.add_model("naive", NaiveForecaster())

    df = pd.DataFrame(
        {"other": [1, 2, 3]}, index=pd.date_range("2026-01-01", periods=3)
    )
    with pytest.raises(ValueError, match="must contain 'sales' column"):
        manager.train_all(df)


def test_predict_all_before_train_raises_error() -> None:
    """Test predict_all before train_all raises RuntimeError."""
    manager = ForecastManager()
    manager.add_model("naive", NaiveForecaster())

    with pytest.raises(RuntimeError, match="must be trained before prediction"):
        manager.predict_all(5)


def test_compare_models_before_train_raises_error() -> None:
    """Test compare_models before train_all raises RuntimeError."""
    manager = ForecastManager()
    manager.add_model("naive", NaiveForecaster())

    test_df = pd.DataFrame(
        {"sales": [1, 2, 3]}, index=pd.date_range("2026-01-01", periods=3)
    )

    with pytest.raises(RuntimeError, match="must be trained before comparison"):
        manager.compare_models(test_df)
