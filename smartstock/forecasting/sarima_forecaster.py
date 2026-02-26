from typing import Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

from smartstock.models.base import BaseForecaster


class SARIMAForecaster(BaseForecaster):
    """
    Seasonal ARIMA model for time series forecasting.

    Parameters:
    -----------
    order : Tuple[int, int, int]
        (p, d, q) where:
        - p: AR order (number of lag observations)
        - d: I order (degree of differencing)
        - q: MA order (aize of moving average window)
    seasonal order : Tuple[int, int, int, int]
        (P, D, Q, s) where:
        - P: Seasonal AR order
        - D: Seasonal I order
        - Q: Seasonal MA order
        - s: Seasonal period (e.g. 7 for weekly, 12 for monthly)
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7),
    ) -> None:
        """Initialize SARIMA model with specified parameters."""

        self.order = order
        self.seasonal_order = seasonal_order
        self.model: Optional[SARIMAX] = None
        self.result: Optional[SARIMAXResults] = None
        self.fitted: bool = False
        self.training_data: Optional[pd.DataFrame] = None

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit SARIMA model to historical sales data.

        Raises:
        -------
        ValueError: If DataFrame is empty or missing 'sales' column
        """
        if df.empty:
            raise ValueError("Training data cannot be empty")

        if "sales" not in df.columns:
            raise ValueError("DataFrame must contain 'sales' column")

        self.training_data = df.copy()

        self.model = SARIMAX(
            df["sales"],
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_invertibility=False,
            enforce_stationarity=False,
        )

        self.result = self.model.fit(disp=False)
        self.fitted = True

    def predict(
        self,
        periods: int,
        include_history: bool = True,
        include_confidence: bool = True,
    ) -> pd.DataFrame:
        """
        Generate forecast for the specified number of periods.

        Returns:
        --------
        pd.DataFrame
            DataFrame with columns:
            - 'forecast': Point forecast
            - 'ci_lower': Lower confidence bound (95%)
            - 'ci_upper': Upper confidence bound (95%)
            Index: DatetimeIndex
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")

        if self.training_data is None:
            raise RuntimeError("Training data not available")

        if self.result is None:
            raise RuntimeError("Model results not available")

        if periods < 0:
            raise ValueError("Periods must be non-negative")

        # Handle edge case: periods = 0
        if periods == 0:
            if not include_history:
                return pd.DataFrame(
                    columns=["forecast", "ci_lower", "ci_upper"],
                    index=pd.DatetimeIndex([]),
                )

            # Return only historical fitted values
            history_df = pd.DataFrame(
                {"forecast": self.result.fittedvalues}, index=self.training_data.index
            )

            if include_confidence:
                # SARIMA doesn't provide CI for historical fitted values
                history_df["ci_lower"] = np.nan
                history_df["ci_upper"] = np.nan

            return history_df

        # Get forecast for future periods
        forecast_result = self.result.get_forecast(steps=periods)
        last_date = self.training_data.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=periods, freq="D"
        )

        # Create DataFrame for future predictions
        future_df = pd.DataFrame(
            {"forecast": forecast_result.predicted_mean}, index=future_dates
        )

        if include_confidence:
            conf_int = forecast_result.conf_int()
            future_df["ci_lower"] = conf_int.iloc[:, 0]
            future_df["ci_upper"] = conf_int.iloc[:, 1]

        if not include_history:
            return future_df

        # Include historical fitted values if requested
        history_df = pd.DataFrame(
            {"forecast": self.result.fittedvalues}, index=self.training_data.index
        )

        if include_confidence:
            # SARIMA doesn't provide confidence intervals for historical fitted values
            history_df["ci_lower"] = np.nan
            history_df["ci_upper"] = np.nan

        # Combine history and future predictions
        result_df = pd.concat([history_df, future_df])
        return result_df

    def get_model_summary(self) -> str:
        """
        Get statistical summary of fitted model.

        Returns:
        --------
        str
            Model summary including coefficients, AIC, BIC, etc.
        """
        if not self.fitted or self.result is None:
            raise RuntimeError("Model must be fitted to get summary")

        return str(self.result.summary())

    def get_residuals(self) -> pd.Series:
        """
        Get model residuals (errors).

        Returns:
        --------
        pd.Series
            Residuals from fitted model
        """
        if not self.fitted or self.result is None:
            raise RuntimeError("Model must be fitted to get residuals")

        return self.result.resid
