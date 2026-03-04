import pandas as pd


def load_raw(path: str) -> pd.DataFrame:
    """
    Load raw CSV dataset from the path.

    Args:
        path: Path to the CSV file (e.g. "data/raw/train.csv")

    Returns:
        A DataFrame with columns: date, store, item, sales
    """
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def filter_series(df: pd.DataFrame, store_id: int, item_id: int) -> pd.DataFrame:
    """
    Extract sales data for a single store-item combination.

    Args:
        df:         Full dataset
        store_id:   Store number (1-10)
        item_id:    Item number (1-50)

    Returns:
        A DataFrame with columns: date, sales
    """
    mask = (df["store"] == store_id) & (df["item"] == item_id)
    series = df[mask].sort_values("date").set_index("date")[["sales"]]
    return series
