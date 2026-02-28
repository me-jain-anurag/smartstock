import pandas as pd
import pytest

from smartstock.optimization.eoq_calculator import EOQCalculator


def test_calculate_dynamic_eoq() -> None:
    """Test standard dynamic EOQ calculation over a forecast period."""
    calculator = EOQCalculator()
    # Forecast for Q1: [1000, 5000, 0]
    forecast_series = pd.Series([1000, 5000, 0], index=["Jan", "Feb", "Mar"])

    # We already know from prior tests:
    # 1000 demand, 20 order, 5 hold -> 90
    # 5000 demand, 20 order, 5 hold -> sqrt(200000/5) -> 200
    # 0 demand -> 0
    results = calculator.calculate(forecast_series, 20, 5)

    assert len(results) == 3
    # 1000 demand, 20 order, 5 hold -> 90 base EOQ
    assert results.iloc[0]["eoq"] == 90
    assert results.iloc[0]["safety_stock"] == 0
    assert results.iloc[0]["reorder_point"] == 0
    assert results.iloc[0]["total_order_quantity"] == 90

    # 5000 demand -> 200 base EOQ
    assert results.iloc[1]["eoq"] == 200
    assert results.iloc[1]["total_order_quantity"] == 200

    # 0 demand -> 0 base EOQ
    assert results.iloc[2]["eoq"] == 0
    assert results.iloc[2]["total_order_quantity"] == 0


def test_calculate_dynamic_eoq_empty_series_raises_error() -> None:
    """Test empty forecast series raises ValueError."""
    calculator = EOQCalculator()
    with pytest.raises(ValueError, match="cannot be empty"):
        calculator.calculate(pd.Series(dtype=float), 20, 5)


def test_calculate_dynamic_eoq_negative_costs_raises_error() -> None:
    """Test negative costs in dynamic EOQ raise ValueError."""
    calculator = EOQCalculator()
    series = pd.Series([100])
    with pytest.raises(ValueError, match="non-negative"):
        calculator.calculate(series, -10, 5)
    with pytest.raises(ValueError, match="must be positive"):
        calculator.calculate(series, 10, 0)


def test_calculate_with_batching_and_safety_stock() -> None:
    """Test dynamic EOQ with supplier batching and safety stock ML buffers."""
    calculator = EOQCalculator()

    forecast_series = pd.Series([1000])  # 1000 demand, base EOQ = 90
    uncertain_series = pd.Series([50])  # yhat_upper - yhat = 50

    # Lead time = 4 periods. Safety Stock = 50 * sqrt(4) = 100
    # Reorder Point = (1000 * 4) + 100 = 4100
    # Total Order = 90 (EOQ) + 100 (SS) = 190.
    # Batch size = 50. math.ceil(190 / 50) * 50 = 200.

    # with Z score of 0.84 (around probability .80)
    # 0.84 * 50 * 2 = 84.  Using exact Z ~0.8416 -> 85
    results = calculator.calculate(
        forecast_series=forecast_series,
        ordering_cost=20,
        holding_cost_per_period=5,
        lead_time_periods=4,
        uncertainty_series=uncertain_series,
        batch_size=50,
        service_level=0.80,  # Roughly Z=0.84
    )

    assert results.iloc[0]["eoq"] == 90
    assert results.iloc[0]["safety_stock"] > 0
    assert results.iloc[0]["reorder_point"] > 4000
    assert results.iloc[0]["total_order_quantity"] % 50 == 0


def test_invalid_uncertainty_or_batch_raises_error() -> None:
    calculator = EOQCalculator()
    forecast = pd.Series([100, 200])
    uncertain = pd.Series([10])  # mismatched length

    with pytest.raises(ValueError, match="must match forecast series length"):
        calculator.calculate(forecast, 20, 5, uncertainty_series=uncertain)

    with pytest.raises(ValueError, match="must be at least 1"):
        calculator.calculate(forecast, 20, 5, batch_size=0)

    with pytest.raises(ValueError, match="must be non-negative"):
        calculator.calculate(forecast, 20, 5, lead_time_periods=-1)

    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        calculator.calculate(forecast, 20, 5, service_level=0.0)
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        calculator.calculate(forecast, 20, 5, service_level=1.5)
