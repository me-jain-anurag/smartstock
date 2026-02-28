import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings

from smartstock.forecasting.forecast_manager import ForecastManager


@settings(max_examples=100)  # type: ignore[misc]
@given(  # type: ignore[misc]
    st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=1000,
    ),
    st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=1000,
    ),
)
def test_mae_less_than_equal_to_rmse(
    actual_list: list[float], predicted_list: list[float]
) -> None:
    """
    Property 4: Model Comparison Metrics Are Consistent.
    Mathematically, MAE <= RMSE holds for all arrays of equal length.
    """
    # Ensure they have the same size
    min_len = min(len(actual_list), len(predicted_list))
    actual = np.array(actual_list[:min_len])
    predicted = np.array(predicted_list[:min_len])

    manager = ForecastManager()
    metrics = manager._calculate_metrics(actual, predicted)

    mae = metrics["mae"]
    rmse = metrics["rmse"]

    # Assert property holding
    assert mae <= rmse + 1e-6  # small tolerance for floating point errors


@settings(max_examples=100)  # type: ignore[misc]
@given(  # type: ignore[misc]
    st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=1000,
    ),
    st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=1000,
    ),
)
def test_r2_upper_bound(actual_list: list[float], predicted_list: list[float]) -> None:
    """
    Property: R2 bounds. R2 should theoretically be <= 1.0.
    """
    min_len = min(len(actual_list), len(predicted_list))
    actual = np.array(actual_list[:min_len])
    predicted = np.array(predicted_list[:min_len])

    manager = ForecastManager()
    metrics = manager._calculate_metrics(actual, predicted)

    r2 = metrics["r2"]

    if not np.isnan(r2):
        assert r2 <= 1.0 + 1e-6
