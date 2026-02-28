import math

import numpy as np
import pandas as pd
from scipy import stats


class EOQCalculator:
    """
    Economic Order Quantity calculator.

    Key Concepts:
    - EOQ minimizes total inventory costs (ordering + holding)
    - Assumes constant demand rate and fixed ordering costs
    - Formula: √(2DS/H) where:
        D = Annual demand
        S = Ordering cost per order
        H = Holding cost per unit per year
    """

    def calculate(
        self,
        forecast_series: pd.Series,
        ordering_cost: float,
        holding_cost_per_period: float,
        lead_time_periods: int = 0,
        uncertainty_series: pd.Series | None = None,
        batch_size: int = 1,
        service_level: float = 0.95,
    ) -> pd.DataFrame:
        """
        Calculate Dynamic Economic Order Quantity per forecasted period,
        including Safety Stock buffers and Order Timing (ROP).

        Integrates directly with time-series forecasting (Prophet/SARIMA).
        Calculates precise EOQ, Safety Stock (using prediction uncertainty
        and desired service level), and Reorder Point (ROP).

        Optimized using Vectorized Pandas operations for massive datasets.

        Args:
            forecast_series: A pandas Series containing forecasted demand values (yhat).
            ordering_cost: Cost per order.
            holding_cost_per_period: Holding cost per unit for the forecast period.
            lead_time_periods: Delay between ordering and receiving (in periods).
            uncertainty_series: Expected error (yhat_upper - yhat) or RMSE per period.
            batch_size: The multiple required by suppliers (e.g. pallets of 50).
            service_level: The target probability of NOT stocking out (default 95%).

        Returns:
            A pandas DataFrame containing period-by-period optimizations.

        Raises:
            ValueError: If inputs are negative, invalid, or mismatched.
        """
        if forecast_series.empty:
            raise ValueError("Forecast series cannot be empty")
        if ordering_cost < 0:
            raise ValueError("Ordering cost must be non-negative")
        if holding_cost_per_period <= 0:
            raise ValueError("Holding cost per period must be positive")
        if lead_time_periods < 0:
            raise ValueError("Lead time must be non-negative")
        if batch_size < 1:
            raise ValueError("Batch size must be at least 1")
        if not (0.0 < service_level < 1.0):
            raise ValueError("Service level must be between 0.0 and 1.0")
        if uncertainty_series is not None and len(uncertainty_series) != len(
            forecast_series
        ):
            raise ValueError("Uncertainty series must match forecast series length")

        # Floor negative demand at 0
        safe_demand = np.maximum(0, forecast_series.to_numpy())

        # 1. Base EOQ: sqrt((2 * D * S) / H)
        eoq_raw = np.sqrt((2 * safe_demand * ordering_cost) / holding_cost_per_period)
        eoq = np.ceil(eoq_raw).astype(int)

        # 2. Safety Stock (Buffer against ML Uncertainty)
        # Using Z-score based on requested Service Level
        safety_stock = np.zeros(len(forecast_series), dtype=int)

        if uncertainty_series is not None and lead_time_periods > 0:
            safe_uncertainty = np.maximum(0, uncertainty_series.to_numpy())
            z_score = stats.norm.ppf(service_level)
            ss_raw = z_score * safe_uncertainty * math.sqrt(lead_time_periods)
            safety_stock = np.ceil(ss_raw).astype(int)

        # 3. Reorder Point (ROP) = (Demand * Lead Time) + Safety Stock
        expected_lead_demand = safe_demand * lead_time_periods
        reorder_point = np.ceil(expected_lead_demand).astype(int) + safety_stock

        # 4. Total Order Quantity with Supplier Batch Constraints
        raw_order_quantity = eoq + safety_stock
        batch_order_quantity = np.ceil(raw_order_quantity / batch_size) * batch_size
        batch_order_quantity = batch_order_quantity.astype(int)

        return pd.DataFrame(
            {
                "expected_demand": safe_demand,
                "eoq": eoq,
                "safety_stock": safety_stock,
                "reorder_point": reorder_point,
                "total_order_quantity": batch_order_quantity,
            },
            index=forecast_series.index,
        )
