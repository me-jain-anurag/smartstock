"""
Integration (smoke) tests for smartstock.dashboard.service.

Tests run_eoq() and run_abc() end-to-end via the service layer to verify that
the service wrappers pass arguments correctly and return the expected output
shapes. Underlying EOQCalculator and ABCAnalyzer logic is covered separately
in tests/unit/.
"""

import pandas as pd

from smartstock.dashboard.service import run_abc, run_eoq

# ── run_eoq ───────────────────────────────────────────────────────────────────


def test_run_eoq_returns_correct_columns() -> None:
    """run_eoq returns a DataFrame with the 5 expected EOQ output columns."""
    forecast = pd.Series([100.0, 150.0, 120.0], index=range(3))
    result = run_eoq(
        forecast_series=forecast,
        ordering_cost=50.0,
        holding_cost_per_period=0.5,
    )

    expected_cols = {
        "expected_demand",
        "eoq",
        "safety_stock",
        "reorder_point",
        "total_order_quantity",
    }
    assert expected_cols.issubset(result.columns)


def test_run_eoq_row_count_matches_forecast_length() -> None:
    """Output has one row per forecast period."""
    n = 7
    result = run_eoq(
        forecast_series=pd.Series([50.0] * n),
        ordering_cost=30.0,
        holding_cost_per_period=1.0,
    )
    assert len(result) == n


def test_run_eoq_batch_size_respected() -> None:
    """total_order_quantity is always a multiple of batch_size."""
    result = run_eoq(
        forecast_series=pd.Series([200.0, 180.0, 220.0]),
        ordering_cost=40.0,
        holding_cost_per_period=0.8,
        lead_time_periods=5,
        uncertainty_series=pd.Series([10.0, 8.0, 12.0]),
        batch_size=50,
        service_level=0.95,
    )
    assert (result["total_order_quantity"] % 50 == 0).all()


# ── run_abc ────────────────────────────────────────────────────────────────────


def test_run_abc_returns_abc_column() -> None:
    """run_abc returns a DataFrame containing an abc_category column."""
    df = pd.DataFrame(
        {
            "item_id": [f"SKU{i:03d}" for i in range(10)],
            "unit_cost": [float(i * 10 + 1) for i in range(10)],
            "annual_demand": [int(i * 100 + 50) for i in range(10)],
        }
    )
    result = run_abc(df)

    assert "abc_category" in result.columns
    assert set(result["abc_category"]).issubset({"A", "B", "C"})


def test_run_abc_row_count_preserved() -> None:
    """Output has the same number of rows as the input."""
    df = pd.DataFrame(
        {
            "item_id": ["A", "B", "C", "D", "E"],
            "unit_cost": [10.0, 5.0, 20.0, 1.0, 50.0],
            "annual_demand": [100, 200, 50, 1000, 30],
        }
    )
    result = run_abc(df)
    assert len(result) == len(df)


def test_run_abc_a_category_present_with_skewed_values() -> None:
    """High-value items are assigned category A."""
    df = pd.DataFrame(
        {
            "item_id": [f"SKU{i:03d}" for i in range(20)],
            "unit_cost": [500.0] * 4 + [10.0] * 16,
            "annual_demand": [1000] * 4 + [50] * 16,
        }
    )
    result = run_abc(df)
    assert "A" in result["abc_category"].values
