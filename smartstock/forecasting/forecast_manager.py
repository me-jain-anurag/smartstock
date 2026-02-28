import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from smartstock.models.base import BaseForecaster


class ForecastManager:
    """
    Manages multiple forecasting models and provides comparison capabilities.

    The ForecastManager orchestrates the lifecycle of forecasting models,
    allowing you to register multiple models, train them on the same data,
    and compare their performance using standard metrics.

    Key Responsibilities:
    1. Model registration and lifecycle management
    2. Training all registered models on the same dataset
    3. Comparing model performance using MAE, RMSE, MAPE metrics
    4. Caching trained models for efficiency

    Example Usage:
    ---------------
    >>> manager = ForecastManager()
    >>> manager.add_model("prophet", ProphetForecaster())
    >>> manager.add_model("sarima", SARIMAForecaster())
    >>> manager.train_all(training_data)
    >>> comparison = manager.compare_models(test_data)
    """

    def __init__(self) -> None:
        """
        Initialize an empty ForecastManager.

        Creates empty dictionaries to store models and their metrics.
        Models are stored by name for easy reference.
        """
        self.models: Dict[str, BaseForecaster] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
        self.training_data: Optional[pd.DataFrame] = None

    def add_model(self, name: str, model: BaseForecaster) -> None:
        """
        Register a forecasting model with the manager.

        Parameters:
        -----------
        name : str
            Unique identifier for the model (e.g., "prophet", "sarima")
        model : BaseForecaster
            Instance of a forecasting model that implements BaseForecaster

        Raises:
        -------
        ValueError: If model name already exists in the manager
        TypeError: If model is not an instance of BaseForecaster
        """
        if name in self.models:
            raise ValueError(f"Model with name '{name}' already exists")

        if not isinstance(model, BaseForecaster):
            raise TypeError(
                "Model must be an instance of BaseForecaster, " f"got {type(model)}"
            )

        self.models[name] = model

    def remove_model(self, name: str) -> None:
        """
        Remove a model from the manager.

        Parameters:
        -----------
        name : str
            Name of the model to remove

        Raises:
        -------
        KeyError: If model name doesn't exist
        """
        if name not in self.models:
            raise KeyError(f"Model '{name}' not found in manager")

        del self.models[name]

        if name in self.metrics:
            del self.metrics[name]

    def get_model(self, name: str) -> BaseForecaster:
        """
        Get a registered model by name.

        Parameters:
        -----------
        name : str
            Name of the model to retrieve

        Returns:
        --------
        BaseForecaster
            The requested forecasting model

        Raises:
        -------
        KeyError: If model name doesn't exist
        """
        if name not in self.models:
            raise KeyError(f"Model '{name}' not found in manager")

        return self.models[name]

    def list_models(self) -> List[str]:
        """
        Get list of all registered model names.

        Returns:
        --------
        List[str]
            List of model names in the order they were added
        """
        return list(self.models.keys())

    def train_all(self, train_data: pd.DataFrame) -> None:
        """
        Train all registered models on the same training data.

        This ensures fair comparison by training all models on identical data.
        The training data is cached for later use in predictions.

        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data with Date index and 'sales' column

        Raises:
        -------
        ValueError: If training data is empty or missing 'sales' column
        RuntimeError: If no models are registered
        """
        if train_data.empty:
            raise ValueError("Training data cannot be empty")

        if "sales" not in train_data.columns:
            raise ValueError("Training data must contain 'sales' column")

        if not self.models:
            raise RuntimeError("No models registered. Add models before training.")

        self.training_data = train_data.copy()

        for name, model in self.models.items():
            try:
                model.fit(train_data)
            except Exception as e:
                warnings.warn(
                    f"Failed to train model '{name}': {str(e)}. "
                    f"Skipping this model for comparison."
                )

    def predict_all(
        self, periods: int, include_history: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate predictions from all trained models.

        Parameters:
        -----------
        periods : int
            Number of periods to forecast
        include_history : bool
            Whether to include historical fitted values in the prediction

        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary mapping model names to their forecast DataFrames

        Raises:
        -------
        RuntimeError: If models haven't been trained yet
        """
        if self.training_data is None:
            raise RuntimeError(
                "Models must be trained before prediction. " "Call train_all() first."
            )

        predictions = {}
        for name, model in self.models.items():
            try:
                predictions[name] = model.predict(periods, include_history)
            except Exception as e:
                warnings.warn(
                    f"Failed to get predictions from model '{name}': {str(e)}"
                )

        return predictions

    def compare_models(
        self, test_data: pd.DataFrame, forecast_horizon: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Compare performance of all trained models on test data.

        Calculates standard forecasting metrics:
        - MAE (Mean Absolute Error): Avg absolute diff between actual and predicted
        - MAPE (Mean Absolute Percentage Error): Average % error
        - R² (R-squared): Proportion of variance explained by the model

        Parameters:
        -----------
        test_data : pd.DataFrame
            Test data with Date index and 'sales' column
        forecast_horizon : Optional[int]
            Number of periods to forecast. If None, uses length of test_data

        Returns:
        --------
        pd.DataFrame
            DataFrame with model names as index and metrics as columns,
            sorted by RMSE (best to worst)

        Raises:
        -------
        ValueError: If test data is empty or missing 'sales' column
        RuntimeError: If models haven't been trained yet
        """
        if test_data.empty:
            raise ValueError("Test data cannot be empty")

        if "sales" not in test_data.columns:
            raise ValueError("Test data must contain 'sales' column")

        if self.training_data is None:
            raise RuntimeError(
                "Models must be trained before comparison. " "Call train_all() first."
            )

        if forecast_horizon is None:
            forecast_horizon = len(test_data)

        if forecast_horizon <= 0:
            raise ValueError("Forecast horizon must be positive")

        comparison_results = []

        for name, model in self.models.items():
            try:
                predictions = model.predict(forecast_horizon, include_history=False)

                if predictions.empty:
                    warnings.warn(
                        f"Model '{name}' returned empty predictions. Skipping."
                    )
                    continue

                actual_values = test_data["sales"].values[:forecast_horizon]
                predicted_values = predictions["forecast"].values[:forecast_horizon]

                if len(actual_values) != len(predicted_values):
                    warnings.warn(
                        f"Model '{name}' prediction length "
                        f"({len(predicted_values)}) doesn't match test "
                        f"data length ({len(actual_values)}). "
                        f"Using min length for comparison."
                    )
                    min_length = min(len(actual_values), len(predicted_values))
                    actual_values = actual_values[:min_length]
                    predicted_values = predicted_values[:min_length]

                metrics = self._calculate_metrics(actual_values, predicted_values)
                metrics["model"] = name  # type: ignore
                comparison_results.append(metrics)

                self.metrics[name] = metrics

            except Exception as e:
                warnings.warn(f"Failed to evaluate '{name}': {str(e)}. Skipping.")

        if not comparison_results:
            raise RuntimeError("No models could be evaluated successfully")

        comparison_df = pd.DataFrame(comparison_results)
        comparison_df = comparison_df.set_index("model")

        return comparison_df.sort_values("rmse")

    def _calculate_metrics(
        self, actual: np.ndarray, predicted: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate forecasting metrics between actual and predicted values.

        This is a helper method that computes:
        - MAE: Mean Absolute Error
        - RMSE: Root Mean Squared Error
        - MAPE: Mean Absolute Percentage Error (handles zero actual values)
        - R²: R-squared (coefficient of determination)

        Parameters:
        -----------
        actual : np.ndarray
            Array of actual values
        predicted : np.ndarray
            Array of predicted values

        Returns:
        --------
        Dict[str, float]
            Dictionary containing all calculated metrics
        """
        if len(actual) != len(predicted):
            raise ValueError(
                f"Actual and predicted arrays must have same length. "
                f"Got {len(actual)} and {len(predicted)}"
            )

        if len(actual) == 0:
            return {
                "mae": np.nan,
                "rmse": np.nan,
                "mape": np.nan,
                "r2": np.nan,
                "n_samples": 0,
            }

        actual = np.asarray(actual, dtype=np.float64)
        predicted = np.asarray(predicted, dtype=np.float64)

        errors = actual - predicted
        absolute_errors = np.abs(errors)
        squared_errors = errors**2

        mae = np.mean(absolute_errors)

        rmse = np.sqrt(np.mean(squared_errors))

        mape = self._calculate_mape(actual, predicted)

        r2 = self._calculate_r2(actual, predicted)

        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
            "r2": float(r2),
            "n_samples": len(actual),
        }

    def _calculate_mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error, handling zero actual values.

        MAPE formula: (100/n) * Σ|(actual - predicted)/actual|
        For zero actual values, we skip those points to avoid division by zero.

        Parameters:
        -----------
        actual : np.ndarray
            Array of actual values
        predicted : np.ndarray
            Array of predicted values

        Returns:
        --------
        float
            MAPE value (percentage, e.g., 10.5 means 10.5% error)
        """
        non_zero_mask = actual != 0
        if not np.any(non_zero_mask):
            return float(np.nan)

        actual_non_zero = actual[non_zero_mask]
        predicted_non_zero = predicted[non_zero_mask]

        percentage_errors = (
            np.abs(actual_non_zero - predicted_non_zero) / actual_non_zero
        )
        mape = np.mean(percentage_errors) * 100

        return float(mape)

    def _calculate_r2(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate R-squared (coefficient of determination).

        R² = 1 - (SS_res / SS_tot)
        where:
        - SS_res = sum of squared residuals (errors)
        - SS_tot = total sum of squares (variance of actual values)

        Parameters:
        -----------
        actual : np.ndarray
            Array of actual values
        predicted : np.ndarray
            Array of predicted values

        Returns:
        --------
        float
            R-squared value (1.0 is perfect prediction, can be negative)
        """
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)

        if ss_tot == 0:
            return float(1.0) if ss_res == 0 else float(np.nan)

        r2 = 1 - (ss_res / ss_tot)
        return float(r2)

    def get_best_model(self, metric: str = "rmse") -> Tuple[str, Dict[str, float]]:
        """
        Get the best performing model based on specified metric.

        Parameters:
        -----------
        metric : str
            Metric to use for comparison ("mae", "rmse", "mape", or "r2")

        Returns:
        --------
        Tuple[str, Dict[str, float]]
            Tuple containing:
            - Name of the best model
            - Dictionary of all metrics for that model

        Raises:
        -------
        ValueError: If metric is not valid or no metrics are available
        """
        valid_metrics = {"mae", "rmse", "mape", "r2"}
        if metric not in valid_metrics:
            raise ValueError(f"Metric must be one of {valid_metrics}, got '{metric}'")

        if not self.metrics:
            raise ValueError("No metrics available. Run compare_models() first.")

        if metric == "r2":
            best_model = max(
                self.metrics.items(), key=lambda x: x[1].get(metric, -np.inf)
            )
        else:
            best_model = min(
                self.metrics.items(), key=lambda x: x[1].get(metric, np.inf)
            )

        return best_model

    def clear_metrics(self) -> None:
        """Clear all stored metrics."""
        self.metrics.clear()

    def clear_models(self) -> None:
        """Clear all registered models and their metrics."""
        self.models.clear()
        self.metrics.clear()
        self.training_data = None

    def __len__(self) -> int:
        """Return number of registered models."""
        return len(self.models)

    def __contains__(self, name: str) -> bool:
        """Check if a model with given name is registered."""
        return name in self.models

    def __repr__(self) -> str:
        """String representation of ForecastManager."""
        model_count = len(self.models)
        trained_status = "trained" if self.training_data is not None else "untrained"
        return f"ForecastManager(models={model_count}, status={trained_status})\n"
