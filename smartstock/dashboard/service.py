"""
SmartStock Dashboard — Service Layer
=====================================
All dashboard pages call this module instead of importing backend modules directly.

Today: wraps Python function calls
Later: swap implementations to call FastAPI HTTP endpoints — UI is untouched.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# ── Column alias map (fuzzy detection) ──────────────────────────────────────
# Maps each required column to the set of aliases we'll auto-detect.
_COL_ALIASES: dict[str, list[str]] = {
    "date": ["date", "Date", "DATE", "datetime", "ds", "timestamp", "Timestamp"],
    "store": ["store", "Store", "store_id", "StoreID", "store_no", "shop", "Shop"],
    "item": ["item", "Item", "item_id", "ItemID", "product", "sku", "SKU", "Product"],
    "sales": [
        "sales",
        "Sales",
        "qty",
        "quantity",
        "Quantity",
        "revenue",
        "units",
        "Units",
    ],
}

REQUIRED_COLS = list(_COL_ALIASES.keys())  # ["date", "store", "item", "sales"]


# ── Data Classes ─────────────────────────────────────────────────────────────


@dataclass
class ForecastResult:
    """Holds everything returned by run_forecast()."""

    model_name: str
    history: pd.DataFrame  # historical data — date index, 'sales' column
    forecast: (
        pd.DataFrame
    )  # full forecast — date index, 'forecast', 'ci_lower', 'ci_upper'
    metrics: dict[str, float] = field(default_factory=dict)
    store_id: int = 0
    item_id: int = 0


# ── CSV Validation ────────────────────────────────────────────────────────────


def load_and_validate_csv(
    file: io.BytesIO | bytes,
) -> tuple[pd.DataFrame | None, list[str], str | None]:
    """
    Load and validate an uploaded CSV file.

    Returns
    -------
    (dataframe, warnings, error_message)
      - dataframe     : validated DataFrame with canonical column names, or None on failure
      - warnings      : list of auto-detection notices e.g. ["Mapped 'Date' → 'date'"]
      - error_message : human-readable string if validation fails, else None
    """
    # 1. Parse
    try:
        df = pd.read_csv(file)
    except Exception as exc:
        return None, [], f"Could not parse file as CSV: {exc}"

    # 2. Empty check
    if df.empty:
        return (
            None,
            [],
            "The uploaded file is empty. Please upload a CSV with data.",
        )

    # 3. Exact match — fastest path
    if set(REQUIRED_COLS).issubset(df.columns):
        return _cast_types(df), [], None

    # 4. Fuzzy detection
    rename_map: dict[str, str] = {}
    unmapped: list[str] = []
    col_lower = {c.lower(): c for c in df.columns}

    for canonical, aliases in _COL_ALIASES.items():
        if canonical in df.columns:
            continue  # already correct
        matched = None
        for alias in aliases:
            if alias in df.columns:
                matched = alias
                break
            if alias.lower() in col_lower:
                matched = col_lower[alias.lower()]
                break
        if matched:
            rename_map[matched] = canonical
        else:
            unmapped.append(canonical)

    if unmapped:
        found = ", ".join(f"`{c}`" for c in df.columns.tolist())
        needed = ", ".join(f"`{c}`" for c in REQUIRED_COLS)
        missing = ", ".join(f"`{c}`" for c in unmapped)
        return (
            None,
            [],
            (
                f"**Could not detect required columns.**\n\n"
                f"**Required:** {needed}\n\n"
                f"**Found in your file:** {found}\n\n"
                f"**Could not map:** {missing}\n\n"
                "Please rename your columns to match the required names exactly."
            ),
        )

    # All 4 mapped via fuzzy — rename and return with warnings
    df = df.rename(columns=rename_map)
    warnings = [
        f"Mapped `{original}` → `{canonical}`"
        for original, canonical in rename_map.items()
    ]
    return _cast_types(df), warnings, None


def _cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce date column to datetime and numeric columns to float.

    # TODO (future): After coercing, check the fraction of NaT values in the
    # 'date' column. If > N% of rows became NaT, warn the user — this usually
    # means the date format is ambiguous (e.g. mm/dd/yyyy vs dd/mm/yyyy when
    # day ≤ 12). Pandas silently mispasrses these with errors='coerce'.
    # Tracked issue: service.py _cast_types — ambiguous date format detection.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["store"] = pd.to_numeric(df["store"], errors="coerce").astype("Int64")
    df["item"] = pd.to_numeric(df["item"], errors="coerce").astype("Int64")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    return df


# ── Forecasting ───────────────────────────────────────────────────────────────


@pd.api.extensions.register_dataframe_accessor("ss")
class _SmartStockAccessor:
    """Internal helper — not for direct use."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df


def run_forecast(
    df: pd.DataFrame,
    store_id: int,
    item_id: int,
    model_name: str,
    periods: int,
) -> ForecastResult:
    """
    Run demand forecasting for a single store-item combination.

    Parameters
    ----------
    df         : Full validated DataFrame from session state
    store_id   : Store ID to filter
    item_id    : Item ID to filter
    model_name : "prophet" | "sarima" | "naive"
    periods    : Forecast horizon in days

    Returns
    -------
    ForecastResult with history, forecast DataFrame, and metrics
    """
    from smartstock.data.cleaner import clean_series
    from smartstock.data.loader import filter_series
    from smartstock.forecasting.forecast_manager import ForecastManager

    # Filter & clean
    series_df = filter_series(df, store_id, item_id)
    series_df = clean_series(series_df)

    # Train / test split (last `periods` days = test)
    test_len = min(periods, max(1, len(series_df) // 5))
    train_df = series_df.iloc[:-test_len]
    test_df = series_df.iloc[-test_len:]

    # Build model
    model = _build_model(model_name)

    manager = ForecastManager()
    manager.add_model(model_name, model)
    manager.train_all(train_df)

    # Metrics on test split
    try:
        metrics_df = manager.compare_models(test_df, forecast_horizon=test_len)
        metrics = metrics_df.loc[model_name].to_dict()
    except Exception:
        metrics = {}

    # Full forecast (retrain on full data for best forward prediction)
    model_full = _build_model(model_name)
    model_full.fit(series_df)
    raw = model_full.predict(periods, include_history=True)

    # Normalise to a canonical shape: date index, forecast / ci_lower / ci_upper
    forecast_df = _normalise_forecast(raw, series_df.index[-1])

    return ForecastResult(
        model_name=model_name,
        history=series_df,
        forecast=forecast_df,
        metrics=metrics,
        store_id=store_id,
        item_id=item_id,
    )


def _build_model(model_name: str) -> Any:
    """Instantiate the correct forecaster by name."""
    name = model_name.lower()
    if name == "prophet":
        from smartstock.forecasting.prophet_forecaster import ProphetForecaster

        return ProphetForecaster()
    elif name == "sarima":
        from smartstock.forecasting.sarima_forecaster import SARIMAForecaster

        return SARIMAForecaster()
    elif name == "naive":
        from smartstock.models.naive import NaiveForecaster

        return NaiveForecaster()
    else:
        raise ValueError(
            f"Unknown model: '{model_name}'. Choose prophet, sarima, or naive."
        )


def _normalise_forecast(
    raw: pd.DataFrame, last_history_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Convert model-specific forecast output into a canonical DataFrame:
      Index : DatetimeIndex
      Cols  : forecast, ci_lower, ci_upper

    Prophet returns: ds, yhat, yhat_lower, yhat_upper
    SARIMA/Naive return: forecast, ci_lower, ci_upper  (date index)
    """
    # Prophet shape
    if "yhat" in raw.columns:
        df = raw.copy()
        if "ds" in df.columns:
            df = df.set_index("ds")
        df.index = pd.to_datetime(df.index)
        return df.rename(
            columns={
                "yhat": "forecast",
                "yhat_lower": "ci_lower",
                "yhat_upper": "ci_upper",
            }
        )[["forecast", "ci_lower", "ci_upper"]]

    # SARIMA / Naive shape — already has correct columns or close to it
    df = raw.copy()
    df.index = pd.to_datetime(df.index)

    # Fill missing ci columns with point estimate (Naive has no intervals)
    if "forecast" not in df.columns and "sales" in df.columns:
        df = df.rename(columns={"sales": "forecast"})
    for col in ("ci_lower", "ci_upper"):
        if col not in df.columns:
            df[col] = df.get("forecast", np.nan)

    return df[["forecast", "ci_lower", "ci_upper"]]


# ── EOQ ───────────────────────────────────────────────────────────────────────


def run_eoq(
    forecast_series: pd.Series,
    ordering_cost: float,
    holding_cost_per_period: float,
    lead_time_periods: int = 0,
    uncertainty_series: pd.Series | None = None,
    batch_size: int = 1,
    service_level: float = 0.95,
) -> pd.DataFrame:
    """
    Run EOQ optimisation on a forecast series.

    Returns the EOQCalculator output DataFrame:
      expected_demand, eoq, safety_stock, reorder_point, total_order_quantity
    """
    from smartstock.optimization.eoq_calculator import EOQCalculator

    calc = EOQCalculator()
    return calc.calculate(
        forecast_series=forecast_series,
        ordering_cost=ordering_cost,
        holding_cost_per_period=holding_cost_per_period,
        lead_time_periods=lead_time_periods,
        uncertainty_series=uncertainty_series,
        batch_size=batch_size,
        service_level=service_level,
    )


# ── ABC Analysis ──────────────────────────────────────────────────────────────


def run_abc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run ABC analysis.

    Parameters
    ----------
    df : DataFrame with columns item_id, unit_cost, annual_demand

    Returns
    -------
    DataFrame with abc_category, annual_value, cumulative_value_pct added
    """
    from smartstock.optimization.abc_analyzer import ABCAnalyzer

    return ABCAnalyzer().analyze(df)


# ── Utility ───────────────────────────────────────────────────────────────────


def get_stores(df: pd.DataFrame) -> list[int]:
    """Return sorted list of store IDs in the dataset."""
    return sorted(df["store"].dropna().unique().tolist())


def get_items(df: pd.DataFrame, store_id: int | None = None) -> list[int]:
    """Return sorted list of item IDs, optionally filtered by store."""
    if store_id is not None:
        df = df[df["store"] == store_id]
    return sorted(df["item"].dropna().unique().tolist())


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to UTF-8 CSV bytes for st.download_button."""
    csv_out = df.to_csv(index=True)
    return str(csv_out).encode("utf-8")
