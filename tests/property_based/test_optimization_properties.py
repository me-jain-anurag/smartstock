import hypothesis.strategies as st
import numpy as np
import pandas as pd
from hypothesis import given, settings

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
