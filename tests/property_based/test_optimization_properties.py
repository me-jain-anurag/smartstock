import hypothesis.strategies as st
import numpy as np
import pandas as pd
from hypothesis import given, settings

from smartstock.optimization.abc_analyzer import ABCAnalyzer
from smartstock.optimization.eoq_calculator import EOQCalculator


@settings(max_examples=100)  # type: ignore[misc]
@given(  # type: ignore[misc]
    st.lists(
        st.floats(min_value=0.1, max_value=1e6), min_size=1, max_size=100
    ),  # forecast_series
    st.floats(min_value=0.0, max_value=1e6),  # ordering_cost
    st.floats(min_value=0.1, max_value=1e6),  # holding_cost
)
def test_eoq_always_positive_or_zero(
    forecast_list: list[float],
    ordering_cost: float,
    holding_cost: float,
) -> None:
    """
    Property 6: EOQ Calculation Preserves Invariants.
    For positive demand/holding costs and non-negative ordering costs, EOQ >= 0.
    """
    calculator = EOQCalculator()
    forecast_series = pd.Series(forecast_list)

    results = calculator.calculate(forecast_series, ordering_cost, holding_cost)

    for row in results.itertuples():
        eoq = getattr(row, "eoq")
        assert isinstance(eoq, (int, np.integer))
        assert eoq >= 0

        total_order = getattr(row, "total_order_quantity")
        assert isinstance(total_order, (int, np.integer))
        assert total_order >= 0


@settings(max_examples=100)  # type: ignore[misc]
@given(  # type: ignore[misc]
    st.lists(
        st.tuples(
            st.floats(
                min_value=0.1, max_value=1e4, allow_nan=False, allow_infinity=False
            ),  # unit_cost
            st.floats(
                min_value=1.0, max_value=1e4, allow_nan=False, allow_infinity=False
            ),  # annual_demand
        ),
        min_size=1,
        max_size=100,
    )
)
def test_abc_analysis_pareto_principle(items: list[tuple[float, float]]) -> None:
    """
    Property 9: ABC Analysis Preserves Pareto Principle.
    Categories should strictly order by value, and cumulative thresholds
    are correctly enforced.
    """
    analyzer = ABCAnalyzer()
    df = pd.DataFrame(
        {
            "item_id": [f"item_{i}" for i in range(len(items))],
            "unit_cost": [cost for cost, demand in items],
            "annual_demand": [demand for cost, demand in items],
        }
    )

    result = analyzer.analyze(df)

    # Check that values are sorted strictly descending
    values = result["annual_value"].values
    assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))

    # Check category assignments match the rule precisely
    prev_cumulative = result["cumulative_value_pct"].shift(fill_value=0.0)

    for i in range(len(result)):
        cat = result.iloc[i]["abc_category"]
        prev_cum = prev_cumulative.iloc[i]

        if prev_cum < 0.80:
            assert cat == "A"
        elif prev_cum < 0.95:
            assert cat == "B"
        else:
            assert cat == "C"
