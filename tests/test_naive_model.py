import pandas as pd
import pytest

from smartstock.models.base import BaseForecaster
from smartstock.models.naive import NaiveForecaster


def test_naive_forecaster_initialization() -> None:
    """Test that NaiveForecaster initializes with None values."""
    model = NaiveForecaster()
    assert model.last_value is None
    assert model.last_date is None


def test_naive_forecaster_fit() -> None:
    """Test that fit() stores the last value and date correctly."""
    model = NaiveForecaster()

    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    df = pd.DataFrame({"sales": [10, 20, 30, 40, 50]}, index=dates)

    model.fit(df)

    assert model.last_value == 50
    assert model.last_date == pd.Timestamp("2026-01-05")


def test_naive_forecaster_fill_empty_data() -> None:
    """Test that fit() raises ValueError with empty data."""
    model = NaiveForecaster()
    df = pd.DataFrame({"sales": []}, index=pd.DatetimeIndex([]))

    with pytest.raises(ValueError, match="Training data is empty"):
        model.fit(df)


def test_naive_forecaster_predict() -> None:
    """Test that predict() generates correct flat forecast."""
    model = NaiveForecaster()

    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    df = pd.DataFrame({"sales": [10, 20, 30]}, index=dates)
    model.fit(df)

    forecast = model.predict(5)

    assert len(forecast) == 5
    assert forecast.index[0] == pd.Timestamp("2026-01-04")
    assert forecast.index[-1] == pd.Timestamp("2026-01-08")
    assert all(forecast)

    assert "forecast" in forecast.columns
    assert isinstance(forecast.index, pd.DatetimeIndex)


def test_naive_forecaster_predict_without_fit() -> None:
    """Test that predict() raises RuntimeError if called before fit()"""
    model = NaiveForecaster()

    with pytest.raises(
        RuntimeError, match="Model must be fitted before calling predict"
    ):
        model.predict(5)


def test_naive_forecaster_implements_base_interface() -> None:
    """Test that NaiveForecaster properly implements BaseForecaster interface."""
    model = NaiveForecaster()

    assert issubclass(NaiveForecaster, BaseForecaster)

    assert hasattr(model, "fit")
    assert hasattr(model, "predict")

    import inspect

    fit_sig = inspect.signature(model.fit)
    predict_sig = inspect.signature(model.predict)

    assert "df" in fit_sig.parameters
    assert "periods" in predict_sig.parameters


def test_naive_forecaster_with_single_data_point() -> None:
    """Test with only one data point."""
    model = NaiveForecaster()

    dates = pd.date_range("2026-01-01", periods=1, freq="D")
    df = pd.DataFrame({"sales": [33]}, index=dates)

    model.fit(df)
    forecast = model.predict(3)

    assert len(forecast) == 3
    assert all(forecast["forecast"] == 33)
    assert forecast.index[0] == pd.Timestamp("2026-01-02")


def test_naive_forecaster_with_irregular_data() -> None:
    """Test that model works with irregular date spacing."""
    model = NaiveForecaster()

    dates = pd.to_datetime(["2026-01-01", "2026-01-05", "2026-01-10"])
    df = pd.DataFrame({"sales": [100, 200, 300]}, index=dates)

    model.fit(df)
    forecast = model.predict(2)

    assert forecast.index[0] == pd.Timestamp("2026-01-11")
    assert forecast.index[1] == pd.Timestamp("2026-01-12")
    assert all(forecast["forecast"] == 300)
