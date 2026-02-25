"""
Unit tests for ProphetForecaster class.
Tests core functionality, edge cases, and error handling.
"""

import pandas as pd
import pytest
from prophet import Prophet

from smartstock.forecasting import ProphetForecaster


def test_prophet_forecaster_initialization() -> None:
    """Test that ProphetForecaster initializes with default parameters."""
    model = ProphetForecaster()

    # Check default parameters
    assert model.growth == "linear"
    assert model.seasonality_mode == "additive"
    assert model.holidays is None
    assert model.changepoint_prior_scale == 0.05
    assert model.seasonality_prior_scale == 10.0
    assert model.holidays_prior_scale == 10.0

    # Check model state
    assert not model.fitted
    assert model.training_data is None
    assert isinstance(model.model, Prophet)


def test_prophet_forecaster_custom_initialization() -> None:
    """Test initialization with custom parameters."""
    holidays_df = pd.DataFrame(
        {"holiday": ["test_holiday"], "ds": pd.to_datetime(["2026-01-01"])}
    )

    model = ProphetForecaster(
        growth="logistic",
        seasonality_mode="multiplicative",
        holidays=holidays_df,
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=5.0,
        holidays_prior_scale=5.0,
    )

    # Check custom parameters
    assert model.growth == "logistic"
    assert model.seasonality_mode == "multiplicative"
    assert model.holidays is not None
    assert len(model.holidays) == 1
    assert model.changepoint_prior_scale == 0.1
    assert model.seasonality_prior_scale == 5.0
    assert model.holidays_prior_scale == 5.0


def test_prophet_forecaster_fit() -> None:
    """Test that fit method trains the model correctly."""
    model = ProphetForecaster()

    # Create synthetic time series data
    dates = pd.date_range("2026-01-01", periods=100, freq="D")
    sales = [10 + i * 0.1 + 5 * (i % 7 == 0) for i in range(100)]
    df = pd.DataFrame({"sales": sales}, index=dates)

    # Fit the model
    model.fit(df)

    # Check model state after fitting
    assert model.fitted
    assert model.training_data is not None
    assert len(model.training_data) == 100
    assert model.training_data.index.equals(df.index)


def test_prophet_forecaster_fit_empty_data() -> None:
    """Test that fit raises error with empty DataFrame."""
    model = ProphetForecaster()
    df = pd.DataFrame({"sales": []}, index=pd.DatetimeIndex([]))

    with pytest.raises(ValueError, match="Training data cannot be empty"):
        model.fit(df)


def test_prophet_forecaster_fit_missing_sales_column() -> None:
    """Test that fit raises error when 'sales' column is missing."""
    model = ProphetForecaster()
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    df = pd.DataFrame({"not_sales": [1, 2, 3]}, index=dates[:3])

    with pytest.raises(ValueError, match="DataFrame must contain 'sales' column"):
        model.fit(df)


def test_prophet_forecaster_predict() -> None:
    """Test that predict generates forecasts with correct shape."""
    model = ProphetForecaster()

    # Create and fit model
    dates = pd.date_range("2026-01-01", periods=50, freq="D")
    sales = [20 + i * 0.2 for i in range(50)]
    df = pd.DataFrame({"sales": sales}, index=dates)
    model.fit(df)

    # Generate forecast
    forecast = model.predict(periods=30)

    # Check forecast shape and columns
    assert len(forecast) == 80  # 50 history + 30 future (include_history=True)
    assert "forecast" in forecast.columns
    assert "ci_lower" in forecast.columns
    assert "ci_upper" in forecast.columns
    assert isinstance(forecast.index, pd.DatetimeIndex)

    # Check that forecast dates are sequential
    assert forecast.index.is_monotonic_increasing


def test_prophet_forecaster_predict_future_only() -> None:
    """Test predict with include_history=False returns only future periods."""
    model = ProphetForecaster()

    dates = pd.date_range("2026-01-01", periods=50, freq="D")
    df = pd.DataFrame({"sales": range(50)}, index=dates)
    model.fit(df)

    forecast = model.predict(periods=30, include_history=False)

    # Should only have future periods
    assert len(forecast) == 30
    # All dates should be after last training date
    assert all(forecast.index > dates[-1])


def test_prophet_forecaster_predict_without_fit() -> None:
    """Test that predict raises error when called before fit."""
    model = ProphetForecaster()

    with pytest.raises(RuntimeError, match="Model must be fitted before prediction"):
        model.predict(periods=10)


def test_prophet_forecaster_predict_no_confidence() -> None:
    """Test predict without confidence intervals."""
    model = ProphetForecaster()

    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    df = pd.DataFrame({"sales": range(30)}, index=dates)
    model.fit(df)

    forecast = model.predict(periods=10, include_confidence=False)

    # Should have forecast column but no confidence intervals
    assert "forecast" in forecast.columns
    assert "ci_lower" not in forecast.columns
    assert "ci_upper" not in forecast.columns


def test_prophet_forecaster_add_regressor_before_fit() -> None:
    """Test adding regressor before fitting."""
    model = ProphetForecaster()

    # Should work before fitting
    model.add_regressor("promotions", prior_scale=5.0)

    # Now fit with additional regressor data
    dates = pd.date_range("2026-01-01", periods=50, freq="D")
    df = pd.DataFrame(
        {"sales": range(50), "promotions": [1 if i % 7 == 0 else 0 for i in range(50)]},
        index=dates,
    )

    model.fit(df)
    assert model.fitted


def test_prophet_forecaster_add_regressor_after_fit() -> None:
    """Test that adding regressor after fit raises error."""
    model = ProphetForecaster()

    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    df = pd.DataFrame({"sales": range(30)}, index=dates)
    model.fit(df)

    with pytest.raises(RuntimeError, match="Regressors must be added before fitting"):
        model.add_regressor("promotions")


def test_prophet_forecaster_add_seasonality() -> None:
    """Test adding custom seasonality."""
    model = ProphetForecaster()

    # Add monthly seasonality (period=30.5 days)
    model.add_seasonality(
        name="monthly", period=30.5, fourier_order=5, prior_scale=5.0, mode="additive"
    )

    # Fit model
    dates = pd.date_range("2026-01-01", periods=100, freq="D")
    df = pd.DataFrame({"sales": range(100)}, index=dates)
    model.fit(df)

    assert model.fitted


def test_prophet_forecaster_implements_base_interface() -> None:
    """Test that ProphetForecaster properly implements BaseForecaster interface."""
    model = ProphetForecaster()

    # Check inheritance
    from smartstock.models.base import BaseForecaster

    assert isinstance(model, BaseForecaster)

    # Check required methods exist
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")

    # Check method signatures
    import inspect

    fit_sig = inspect.signature(model.fit)
    predict_sig = inspect.signature(model.predict)

    assert "df" in fit_sig.parameters
    assert "periods" in predict_sig.parameters


def test_prophet_forecaster_with_holidays() -> None:
    """Test ProphetForecaster with holiday effects."""
    # Create holidays dataframe
    holidays = pd.DataFrame(
        {
            "holiday": ["New Year", "Christmas"],
            "ds": pd.to_datetime(["2026-01-01", "2026-12-25"]),
        }
    )

    model = ProphetForecaster(holidays=holidays)

    # Create training data
    dates = pd.date_range("2026-01-01", periods=365, freq="D")
    df = pd.DataFrame({"sales": [50] * 365}, index=dates)

    model.fit(df)
    forecast = model.predict(periods=30)

    assert model.fitted
    assert len(forecast) == 395  # 365 + 30
    assert "ci_lower" in forecast.columns
    assert "ci_upper" in forecast.columns


def test_prophet_forecaster_get_model_components() -> None:
    """Test extracting model components."""
    model = ProphetForecaster()

    dates = pd.date_range("2026-01-01", periods=100, freq="D")
    df = pd.DataFrame({"sales": range(100)}, index=dates)
    model.fit(df)

    components = model.get_model_components()

    # Check component columns
    assert "trend" in components.columns
    assert "yearly" in components.columns
    assert "weekly" in components.columns
    assert "daily" in components.columns
    assert len(components) == 100


def test_prophet_forecaster_get_model_components_without_fit() -> None:
    """Test that get_model_components raises error before fitting."""
    model = ProphetForecaster()

    with pytest.raises(
        RuntimeError, match="Model must be fitted to extract components"
    ):
        model.get_model_components()


def test_prophet_forecaster_cross_validate() -> None:
    """Test cross-validation functionality."""
    model = ProphetForecaster()

    dates = pd.date_range("2026-01-01", periods=500, freq="D")
    df = pd.DataFrame({"sales": range(500)}, index=dates)
    model.fit(df)

    # Perform cross-validation
    metrics = model.cross_validate(
        horizon="30 days", period="15 days", initial="365 days"
    )

    # Check metrics dataframe
    assert isinstance(metrics, pd.DataFrame)
    assert "mae" in metrics.columns
    assert "rmse" in metrics.columns
    assert "mape" in metrics.columns


def test_prophet_forecaster_cross_validate_without_fit() -> None:
    """Test that cross_validate raises error before fitting."""
    model = ProphetForecaster()

    with pytest.raises(
        RuntimeError, match="Model must be fitted before cross-validation"
    ):
        model.cross_validate()
