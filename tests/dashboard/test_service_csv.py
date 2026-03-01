"""
Unit tests for smartstock.dashboard.service — CSV loading & type casting.

Covers load_and_validate_csv and _cast_types.
"""

import io

import pandas as pd
import pytest

from smartstock.dashboard.service import (
    REQUIRED_COLS,
    _cast_types,
    load_and_validate_csv,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_csv(
    date_col: str = "date",
    store_col: str = "store",
    item_col: str = "item",
    sales_col: str = "sales",
    n_rows: int = 3,
) -> io.BytesIO:
    data = {
        date_col: [f"2023-01-0{i + 1}" for i in range(n_rows)],
        store_col: list(range(1, n_rows + 1)),
        item_col: [10] * n_rows,
        sales_col: [float(i * 5) for i in range(n_rows)],
    }
    buf = io.BytesIO()
    pd.DataFrame(data).to_csv(buf, index=False)
    buf.seek(0)
    return buf


# ── load_and_validate_csv — happy paths ───────────────────────────────────────


def test_exact_columns_accepted() -> None:
    """Canonical column names pass with zero warnings and no error."""
    df, warnings, err = load_and_validate_csv(_make_csv())

    assert err is None
    assert warnings == []
    assert df is not None
    assert set(REQUIRED_COLS).issubset(df.columns)


def test_fuzzy_columns_remapped() -> None:
    """Aliased column names trigger warnings but succeed."""
    buf = _make_csv(
        date_col="Date", store_col="Store", item_col="item_id", sales_col="qty"
    )
    df, warnings, err = load_and_validate_csv(buf)

    assert err is None
    assert df is not None
    assert set(REQUIRED_COLS).issubset(df.columns)
    assert len(warnings) == 4
    for w in warnings:
        assert "→" in w


def test_partial_fuzzy_remap() -> None:
    """Only columns that need renaming produce warnings."""
    buf = _make_csv(item_col="item_id", sales_col="qty")
    df, warnings, err = load_and_validate_csv(buf)

    assert err is None
    assert df is not None
    assert len(warnings) == 2  # item_id → item, qty → sales


def test_extra_columns_preserved() -> None:
    """Extra columns beyond the 4 required pass through unchanged."""
    data = {
        "date": ["2023-01-01", "2023-01-02"],
        "store": [1, 1],
        "item": [10, 10],
        "sales": [5.0, 6.0],
        "price": [1.0, 2.0],
    }
    buf = io.BytesIO()
    pd.DataFrame(data).to_csv(buf, index=False)
    buf.seek(0)

    df, _, err = load_and_validate_csv(buf)

    assert err is None
    assert "price" in df.columns  # type: ignore[union-attr]


# ── load_and_validate_csv — failure paths ─────────────────────────────────────


def test_bad_columns_rejected() -> None:
    """Unrecognisable column names return None + error message."""
    data = {"col_a": [1], "col_b": [2], "col_c": [3], "col_d": [4]}
    buf = io.BytesIO()
    pd.DataFrame(data).to_csv(buf, index=False)
    buf.seek(0)

    df, warnings, err = load_and_validate_csv(buf)

    assert df is None
    assert warnings == []
    assert err is not None
    assert "Could not detect" in err


def test_empty_csv_rejected() -> None:
    """Headers-only CSV returns None + error message."""
    buf = io.BytesIO()
    pd.DataFrame(columns=REQUIRED_COLS).to_csv(buf, index=False)
    buf.seek(0)

    df, warnings, err = load_and_validate_csv(buf)

    assert df is None
    assert err is not None
    assert "empty" in err.lower()


def test_unparseable_bytes_rejected() -> None:
    """Binary data that is not valid CSV returns None + error message."""
    buf = io.BytesIO(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

    df, warnings, err = load_and_validate_csv(buf)

    assert df is None
    assert err is not None


# ── _cast_types ────────────────────────────────────────────────────────────────


def test_cast_types_iso_date_parsed() -> None:
    """ISO 8601 dates are parsed to datetime without NaT."""
    df = pd.DataFrame(
        {
            "date": ["2023-01-01", "2023-06-15", "2023-12-31"],
            "store": [1, 2, 3],
            "item": [10, 10, 10],
            "sales": [5.0, 10.0, 15.0],
        }
    )
    result = _cast_types(df)

    assert pd.api.types.is_datetime64_any_dtype(result["date"])
    assert result["date"].isna().sum() == 0


def test_cast_types_numeric_coercion() -> None:
    """String-encoded numbers are cast to the correct numeric types."""
    df = pd.DataFrame(
        {
            "date": ["2023-01-01"],
            "store": ["7"],
            "item": ["42"],
            "sales": ["99.5"],
        }
    )
    result = _cast_types(df)

    assert result["sales"].iloc[0] == pytest.approx(99.5)
    assert result["store"].iloc[0] == 7
    assert result["item"].iloc[0] == 42


def test_cast_types_unparseable_date_becomes_nat() -> None:
    """Unparseable date strings become NaT without raising."""
    df = pd.DataFrame(
        {
            "date": ["not-a-date", "2023-01-01"],
            "store": [1, 1],
            "item": [1, 1],
            "sales": [1.0, 1.0],
        }
    )
    result = _cast_types(df)

    assert result["date"].isna().sum() == 1


def test_cast_types_preserves_extra_columns() -> None:
    """_cast_types does not drop columns beyond the 4 required."""
    df = pd.DataFrame(
        {
            "date": ["2023-01-01"],
            "store": [1],
            "item": [1],
            "sales": [5.0],
            "stock_level": [100],
        }
    )
    result = _cast_types(df)

    assert "stock_level" in result.columns


def test_cast_types_does_not_mutate_input() -> None:
    """_cast_types returns a copy — the original DataFrame is unchanged."""
    df = pd.DataFrame(
        {
            "date": ["2023-01-01"],
            "store": ["1"],
            "item": ["1"],
            "sales": ["5"],
        }
    )
    original_dtype = df["sales"].dtype
    _cast_types(df)

    assert df["sales"].dtype == original_dtype
