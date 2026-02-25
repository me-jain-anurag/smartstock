from typing import Optional

import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

from smartstock.models.base import BaseForecaster


class ProphetForecaster(BaseForecaster):
    """
    Parameters:
    -----------
    growth : str
        Type of trend: 'linear' or 'logistic'
    seasonality_mode : str
        'additive' (default) or 'multiplicative'
    holidays : Optional[pd.DataFrame]
        DataFrame with columns 'holiday' (string) and 'ds' (date)
        Optional list of holidays to include
    changepoint_prior_scale : float
        Flexibility of trend changepoints (higher = more flexible)
    seasonality_prior_scale : float
        Strength of seasonality components
    holidays_prior_scale : float
        Strength of holiday effects
    """

    def __init__(
        self,
        growth: str = "linear",
        seasonality_mode: str = "additive",
        holidays: Optional[pd.DataFrame] = None,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
    ) -> None:
        """Initialize Prophet forecasting model with specified parameters."""
        self.growth = growth
        self.seasonality_mode = seasonality_mode
        self.holidays = holidays
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale

        self.model = Prophet(
            growth=growth,
            seasonality_mode=seasonality_mode,
            holidays=holidays,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            # Default to 80% confidence interval
            interval_width=0.8,
            # Include uncertainty in seasonality
            uncertainty_samples=1000,
        )

        self.fitted: bool = False
        self.training_data: Optional[pd.DataFrame] = None

    def fit(self, df: pd.DataFrame) -> None:
        """Fit Prophet model to historical sales data."""
        if df.empty:
            raise ValueError("Training data cannot be empty")

        if "sales" not in df.columns:
            raise ValueError("DataFrame must contain sales column")

        prophet_df = df.reset_index()
        prophet_df = prophet_df.rename(columns={df.index.name: "ds", "sales": "y"})

        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

        self.training_data = df.copy()

        self.model.fit(prophet_df)
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
            - 'ci_lower': Lower confidence bound (80%)
            - 'ci_upper': Upper confidence bound (80%)
            Index: DatetimeIndex
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")

        future = self.model.make_future_dataframe(
            periods=periods, include_history=include_history
        )

        forecast = self.model.predict(future)

        result_df = forecast[["ds", "yhat"]].copy()
        result_df = result_df.rename(columns={"yhat": "forecast"})

        if include_confidence:
            result_df["ci_lower"] = forecast["yhat_lower"]
            result_df["ci_upper"] = forecast["yhat_upper"]

        result_df = result_df.set_index("ds")
        result_df.index.name = "date"

        if not include_history:
            if self.training_data is None:
                raise RuntimeError("Training data is not available")
            last_training_data = self.training_data.index[-1]
            result_df = result_df[result_df.index > last_training_data]

        return result_df

    def cross_validate(
        self,
        horizon: str = "30 days",
        period: str = "15 days",
        initial: str = "365 days",
        parallel: str = "processes",
    ) -> pd.DataFrame:
        """
        Perform time series cross validation.

        Returns:
        --------
        pd.DataFrame
            Cross-validation metrics (MAE, RMSE, MAPE, etc.)
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before cross-validation")

        df_cv = cross_validation(
            self.model,
            horizon=horizon,
            period=period,
            initial=initial,
            parallel=parallel,
        )

        df_performance = performance_metrics(df_cv)

        return df_performance

    def get_model_components(self) -> pd.DataFrame:
        """
        Extract trend, seasonality, and holiday components from fitted model.

        Returns:
        --------
        pd.DataFrame
            DataFrame with columns for each component
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted to extract components")

        future = self.model.make_future_dataframe(periods=0)

        forecast = self.model.predict(future)

        components = forecast[["ds", "trend", "yearly", "weekly", "daily"]].copy()

        if "holidays" in forecast.columns:
            components["holidays"] = forecast["holidays"]

        return components.set_index("ds")

    def add_regressor(
        self, name: str, prior_scale: float = 10.0, standardize: bool = True
    ) -> None:
        """Add an additional regressor to the model."""
        if self.fitted:
            raise RuntimeError("Regressors must be added before fitting")

        self.model.add_regressor(
            name=name, prior_scale=prior_scale, standardize=standardize
        )

    def add_seasonality(
        self,
        name: str,
        period: float,
        fourier_order: int,
        prior_scale: float = 10.0,
        mode: str = "additive",
    ) -> None:
        """Add a custom seasonality component."""
        if self.fitted:
            raise RuntimeError("Seasonalities must be added before fitting")

        self.model.add_seasonality(
            name=name,
            period=period,
            fourier_order=fourier_order,
            prior_scale=prior_scale,
            mode=mode,
        )
