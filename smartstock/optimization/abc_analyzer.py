import numpy as np
import pandas as pd


class ABCAnalyzer:
    """
    ABC Analysis for inventory prioritization using the Pareto Principle.
    Top 80% of value = Category A
    Next 15% of value = Category B
    Bottom 5% of value = Category C
    """

    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the ABC categories for the given dataset.

        Args:
            data: DataFrame with 'item_id', 'unit_cost', and 'annual_demand' columns.

        Returns:
            DataFrame sorted by annual_value descending, containing the original
            columns plus 'annual_value', 'cumulative_value_pct', and 'abc_category'.

        Raises:
            ValueError: If the DataFrame is empty or missing required columns.
        """
        if data.empty:
            raise ValueError("Input DataFrame is empty")

        required_cols = {"item_id", "unit_cost", "annual_demand"}
        if not required_cols.issubset(data.columns):
            raise ValueError(f"Input must contain columns: {required_cols}")

        df = data.copy()

        # 1. Calculate the financial volume of each line item
        df["annual_value"] = df["unit_cost"] * df["annual_demand"]

        # 2. Sort items descending by their value
        df = df.sort_values(by="annual_value", ascending=False).reset_index(drop=True)

        total_value = df["annual_value"].sum()
        if total_value <= 0:
            # Handle edge case where no items have positive financial value
            df["cumulative_value_pct"] = 0.0
            df["abc_category"] = "C"
            return df

        # 3. Calculate cumulative percentages
        df["cumulative_value_pct"] = df["annual_value"].cumsum() / total_value

        # 4. Enforce Pareto breakpoints (80/15/5)
        # We look at the cumulative sum prior to adding the current item to ensure
        # the item that crosses the boundary is fully included in the higher tier.
        prev_cumulative = df["cumulative_value_pct"].shift(fill_value=0.0)

        conditions = [prev_cumulative < 0.80, prev_cumulative < 0.95]
        choices = ["A", "B"]

        df["abc_category"] = np.select(conditions, choices, default="C")

        return df
