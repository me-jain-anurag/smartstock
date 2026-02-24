import pandas as pd
import numpy as np
from smartstock.models.base import BaseForecaster

class NaiveForecaster(BaseForecaster):
    """
    Naive baseline model: "The future is the same as the last seen value."
    """

    def __init__(self):
        self.last_value = None
        self.last_date = None

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fits the model by simply remembering the very last observed sales value.
        """
        if df.empty:
            raise ValueError("Training data is empty.")

        self.last_value = df['sales'].iloc[-1]
        self.last_date = df.index[-1]

    def predict(self, periods: int) -> pd.DataFrame:
        """
        Creates a flat forecast by repeating the last value.
        """
        if self.last_value is None:
            raise RuntimeError("Model must be fitting before calling predict().")

        forecast_dates = pd.date_range(
            start=self.last_date + pd.Timedelta(days=1),
            periods=periods,
            freq='D'
        )

        forecast_df = pd.DataFrame({
            'forecast': np.full(periods, self.last_value)
        }, index=forecast_dates)

        return forecast_df