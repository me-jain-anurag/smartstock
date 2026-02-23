import pytest
import pandas as pd
from smartstock.data.loader import load_raw, filter_series

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "date": pd.to_datetime([
            "2013-01-01", "2013-01-01",
            "2013-01-02", "2013-01-02",
            "2013-01-03", "2013-01-03",
        ]),
        "store": [1, 1, 1, 2, 2, 2],
        "item":  [1, 2, 1, 1, 1, 2],
        "sales": [10, 20, 15, 8, 12, 5],
    })

def test_load_raw_returns_dataframe(tmp_path):
    """load_raw() should return a DataFrame with the correct columns"""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("date,store,item,sales\n2013-01-01,1,1,10\n")

    df = load_raw(str(csv_file))

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ['date', 'store', 'item', 'sales']
    assert df['date'].dtype == 'datetime64[ns]'

def test_filter_series_correct_shape(sample_df):
    """filter_series() should return only rows for the requested store+item."""
    result = filter_series(sample_df, store_id = 1, item_id = 1)

    assert len(result) == 2
    assert result.index.name == "date"
    assert "sales" in result.columns

def test_filter_series_sorted(sample_df):
    """Dates in the result must be in ascending order."""
    result = filter_series(sample_df, store_id = 2, item_id = 1)
    assert result.index.is_monotonic_increasing