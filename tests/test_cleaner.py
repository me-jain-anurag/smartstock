import pandas as pd

from smartstock.data.cleaner import clean_series


def test_clean_series_fill_missing_dates() -> None:
    df = pd.DataFrame(
        {"sales": [10.0, 30.0], "date": pd.to_datetime(["2026-01-01", "2026-01-03"])}
    ).set_index("date")

    cleaned = clean_series(df)

    assert len(cleaned) == 3
    assert cleaned.loc["2026-01-02", "sales"] == 20.0


def test_clean_series_caps_outliers() -> None:
    df = pd.DataFrame(
        {
            "sales": [10, 11, 10, 12, 10, 1000],
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-04",
                    "2026-01-05",
                    "2026-01-06",
                ]
            ),
        }
    ).set_index("date")

    cleaned = clean_series(df)

    assert cleaned["sales"].max() < 100


def test_clean_series_negative_sales() -> None:
    df = pd.DataFrame(
        {"sales": [-10.0, 20.0], "date": pd.to_datetime(["2026-01-01", "2026-01-02"])}
    ).set_index("date")

    cleaned = clean_series(df)

    assert cleaned["sales"].min() >= 0
    assert cleaned.loc["2026-01-01", "sales"] == 0.0
