from abc import ABC, abstractmethod

import pandas as pd


class BaseForecaster(ABC):
    """
    Abstract Base Class for all forecasting models.
    This acts as an Interface.
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """
        Train the model on the provided DataFrame.
        Expected format: Date as index, 'sales' as column
        """
        pass  # pragma: no cover

    @abstractmethod
    def predict(self, periods: int, include_history: bool = True) -> pd.DataFrame:
        """
        Predict the next 'periods' days of sales.
        Returns: DataFrame with Date Index and 'forecast' column
        """
        pass  # pragma: no cover
