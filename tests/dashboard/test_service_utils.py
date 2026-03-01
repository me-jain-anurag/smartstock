"""
Unit tests for smartstock.dashboard.service — utility helpers.

Covers get_stores, get_items, and dataframe_to_csv_bytes.
"""

import io

import pandas as pd

from smartstock.dashboard.service import dataframe_to_csv_bytes, get_items, get_stores


def _sample_df() -> pd.DataFrame:
    """Minimal validated DataFrame with 2 stores × 3 items."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-01"] * 6),
            "store": [1, 1, 1, 2, 2, 2],
            "item": [10, 20, 30, 10, 20, 30],
            "sales": [5.0] * 6,
        }
    )


# ── get_stores ────────────────────────────────────────────────────────────────


def test_get_stores_returns_sorted() -> None:
    """Store IDs are returned in ascending order."""
    result = get_stores(_sample_df())
    assert result == sorted(result)


def test_get_stores_unique_values() -> None:
    """No duplicates — only unique store IDs are returned."""
    result = get_stores(_sample_df())
    assert result == [1, 2]


def test_get_stores_drops_nan() -> None:
    """NaN store IDs are excluded from the result."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-01"] * 3),
            "store": pd.array([1, None, 3], dtype="Int64"),
            "item": [10, 10, 10],
            "sales": [5.0, 5.0, 5.0],
        }
    )
    result = get_stores(df)

    assert None not in result
    assert len(result) == 2


# ── get_items ─────────────────────────────────────────────────────────────────


def test_get_items_all_stores() -> None:
    """Without a store filter, all unique items are returned sorted."""
    result = get_items(_sample_df())
    assert result == [10, 20, 30]


def test_get_items_filtered_by_store() -> None:
    """Items differ per store — filtering returns only that store's items."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-01"] * 4),
            "store": [1, 1, 2, 2],
            "item": [10, 20, 30, 40],
            "sales": [5.0] * 4,
        }
    )
    assert get_items(df, store_id=1) == [10, 20]
    assert get_items(df, store_id=2) == [30, 40]


def test_get_items_sorted() -> None:
    """Items are always returned in ascending order."""
    result = get_items(_sample_df())
    assert result == sorted(result)


# ── dataframe_to_csv_bytes ────────────────────────────────────────────────────


def test_dataframe_to_csv_bytes_returns_bytes() -> None:
    """Return type is bytes."""
    assert isinstance(dataframe_to_csv_bytes(_sample_df()), bytes)


def test_dataframe_to_csv_bytes_is_valid_utf8() -> None:
    """Bytes decode to valid UTF-8 without error."""
    decoded = dataframe_to_csv_bytes(_sample_df()).decode("utf-8")
    assert len(decoded) > 0


def test_dataframe_to_csv_bytes_roundtrip() -> None:
    """Bytes round-trip back to a DataFrame with the same row count."""
    df = _sample_df()
    raw = dataframe_to_csv_bytes(df)
    recovered = pd.read_csv(io.BytesIO(raw), index_col=0)
    assert recovered.shape[0] == df.shape[0]
    assert "sales" in recovered.columns
