"""
Property-based tests for smartstock.dashboard.service.load_and_validate_csv.

Follows the same Hypothesis style as:
tests/property_based/test_optimization_properties.py

Invariants:
  A. Any DataFrame with the 4 canonical columns and >=1 row always succeeds.
  B. Warning count equals the number of renamed columns.
  C. A successful result always contains all 4 required columns.
"""

import io

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from smartstock.dashboard.service import REQUIRED_COLS, load_and_validate_csv


def _df_to_bytes(df: pd.DataFrame) -> io.BytesIO:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


@settings(max_examples=60)  # type: ignore[misc]
@given(  # type: ignore[misc]
    st.lists(
        st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=30,
    )
)
def test_invariant_exact_schema_always_succeeds(sales_values: list[float]) -> None:
    """
    Invariant A: A DataFrame with the 4 canonical column names and >=1 row
    must always be accepted with no error and no warnings.
    """
    n = len(sales_values)
    df = pd.DataFrame(
        {
            "date": [f"2023-01-{(i % 28) + 1:02d}" for i in range(n)],
            "store": list(range(1, n + 1)),
            "item": [10] * n,
            "sales": sales_values,
        }
    )
    result_df, warnings, err = load_and_validate_csv(_df_to_bytes(df))

    assert err is None, f"Unexpected error: {err}"
    assert warnings == [], f"Unexpected warnings: {warnings}"
    assert result_df is not None


@settings(max_examples=30)  # type: ignore[misc]
@given(st.integers(min_value=1, max_value=20))  # type: ignore[misc]
def test_invariant_warning_count_equals_rename_count(n_rows: int) -> None:
    """
    Invariant B: When all 4 columns are fuzzy-matched, exactly 4 warnings
    are produced — one per renamed column.
    """
    df = pd.DataFrame(
        {
            "Date": [f"2023-01-{(i % 28) + 1:02d}" for i in range(n_rows)],
            "Store": list(range(1, n_rows + 1)),
            "item_id": [10] * n_rows,
            "qty": [float(i) for i in range(n_rows)],
        }
    )
    _, warnings, err = load_and_validate_csv(_df_to_bytes(df))

    assert err is None
    assert len(warnings) == 4


@settings(max_examples=60)  # type: ignore[misc]
@given(  # type: ignore[misc]
    st.lists(
        st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=30,
    )
)
def test_invariant_result_contains_required_columns(sales_values: list[float]) -> None:
    """
    Invariant C: Any successfully loaded DataFrame contains all 4 required
    columns as a subset of its columns.
    """
    n = len(sales_values)
    df = pd.DataFrame(
        {
            "date": [f"2023-01-{(i % 28) + 1:02d}" for i in range(n)],
            "store": list(range(1, n + 1)),
            "item": [10] * n,
            "sales": sales_values,
        }
    )
    result_df, _, err = load_and_validate_csv(_df_to_bytes(df))

    if err is None:
        assert result_df is not None
        assert set(REQUIRED_COLS).issubset(result_df.columns)
