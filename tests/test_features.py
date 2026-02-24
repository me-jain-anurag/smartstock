import pandas as pd

from smartstock.data.features import add_time_features


def test_add_time_features_logic() -> None:
    df = pd.DataFrame(
        {"sales": [10, 20], "date": pd.to_datetime(["2026-01-05", "2026-01-10"])}
    ).set_index("date")

    result = add_time_features(df)

    assert "day_of_week" in result.columns
    assert "month" in result.columns
    assert "is_weekend" in result.columns

    assert result.loc["2026-01-05", "day_of_week"] == 0
    assert result.loc["2026-01-05", "month"] == 1
    assert result.loc["2026-01-05", "is_weekend"] == 0

    assert result.loc["2026-01-10", "day_of_week"] == 5
    assert result.loc["2026-01-10", "month"] == 1
    assert result.loc["2026-01-10", "is_weekend"] == 1
