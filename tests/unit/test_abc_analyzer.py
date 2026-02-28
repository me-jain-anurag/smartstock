import pandas as pd
import pytest

from smartstock.optimization.abc_analyzer import ABCAnalyzer


def test_abc_analyzer_basic_pareto() -> None:
    analyzer = ABCAnalyzer()

    # Setup highly skewed Pareto distribution
    data = pd.DataFrame(
        {
            "item_id": ["item_1", "item_2", "item_3", "item_4"],
            "unit_cost": [100, 10, 5, 1],
            "annual_demand": [10, 10, 10, 10],
        }
    )

    # Values: 1000, 100, 50, 10
    # Total: 1160
    # Cumulative:
    # 1: 1000 (0.86) -> A (0.0 < 0.8)
    # 2: 1100 (0.948) -> B (0.86 >= 0.8 but < 0.95)
    # 3: 1150 (0.991) -> B (0.948 < 0.95)
    # 4: 1160 (1.0) -> C (0.991 >= 0.95)

    result = analyzer.analyze(data)

    assert len(result) == 4
    assert result["item_id"].iloc[0] == "item_1"
    assert result["abc_category"].iloc[0] == "A"
    assert result["abc_category"].iloc[1] == "B"
    assert result["abc_category"].iloc[2] == "B"
    assert result["abc_category"].iloc[3] == "C"


def test_abc_analyzer_empty_dataframe() -> None:
    analyzer = ABCAnalyzer()

    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        analyzer.analyze(pd.DataFrame())


def test_abc_analyzer_missing_columns() -> None:
    analyzer = ABCAnalyzer()
    data = pd.DataFrame({"wrong_id": [1], "unit_cost": [10], "annual_demand": [100]})

    with pytest.raises(ValueError, match="Input must contain columns"):
        analyzer.analyze(data)


def test_abc_analyzer_zero_total_value() -> None:
    analyzer = ABCAnalyzer()
    data = pd.DataFrame(
        {"item_id": [1, 2], "unit_cost": [0, 0], "annual_demand": [100, 50]}
    )

    result = analyzer.analyze(data)

    assert result["cumulative_value_pct"].eq(0.0).all()
    assert result["abc_category"].eq("C").all()
