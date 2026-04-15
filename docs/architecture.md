# SmartStock Architecture

## High-level flow

1. **Load sales CSV** in the dashboard (`Data Upload` page)
2. **Validate and normalize columns** via `smartstock.dashboard.service.load_and_validate_csv`
3. **Filter + clean series** for selected `store`/`item`
4. **Train and run forecast model** (`prophet`, `sarima`, or `naive`)
5. **Compute inventory policy** (EOQ, safety stock, reorder point)
6. **Run ABC analysis** on annual value data

## Package responsibilities

### `smartstock.data`
- `loader.py`: reads and filters raw data
- `cleaner.py`: handles date continuity and outlier capping
- `features.py`: adds time-based model features

### `smartstock.forecasting`
- `forecast_manager.py`: common training/evaluation orchestration
- `prophet_forecaster.py`: Prophet implementation
- `sarima_forecaster.py`: SARIMA implementation

### `smartstock.models`
- `base.py`: base forecasting model contract
- `naive.py`: simple baseline forecaster

### `smartstock.optimization`
- `eoq_calculator.py`: EOQ + safety stock + reorder point calculations
- `abc_analyzer.py`: Pareto-based inventory categorization (A/B/C)

### `smartstock.dashboard`
- `app.py`: Streamlit entrypoint
- `pages/`: multi-page UI (Home, Upload, Forecasting, Optimization, References, API Docs)
- `service.py`: dashboard-facing service layer to keep page logic thin

### `smartstock.api`
- Reserved package scaffold for planned REST API implementation.

## Design notes

- Dashboard pages call the service layer instead of directly coupling to lower-level modules.
- Forecast outputs are normalized to a common schema (`forecast`, `ci_lower`, `ci_upper`) before downstream EOQ usage.
- Optimization supports uncertainty-aware safety stock using forecast confidence intervals.
