# SmartStock Project References

This document maintains a running list of all external tools, Python libraries, and mathematical formulas utilized throughout the creation of the SmartStock platform. It serves as a centralized glossary to explain _why_ specific technologies or equations were chosen.

## 1. Libraries and Tools

- **Pandas (`pandas`)**: Used extensively for data manipulation and analysis, primarily for structuring time-series data and handling DataFrames.
- **NumPy (`numpy`)**: Used for numerical operations, array handling, and complex mathematical calculations like calculating standard deviation or setting up dummy forecast distributions.
- **Prophet (`prophet`)**: Developed by Facebook, used for time-series forecasting. Selected because it intuitively handles daily seasonality, holiday effects, and missing data out-of-the-box.
- **Statsmodels (`statsmodels`)**: Used for the SARIMA model implementation (`SARIMAX`). Chosen for its strict statistical rigor and ability to mathematically address Autoregressive Integrated Moving Average concepts with Seasonality.
- **Pytest (`pytest`)**: The core testing framework used to verify unit logic and integration workflows.
- **Hypothesis (`hypothesis`)**: Used for Property-Based Testing. Chosen because it dynamically generates hundreds of extreme edge cases (e.g., massive negative numbers, zeros) to ensure the code's underlying mathematical properties hold true universally.
- **Pre-commit (`pre-commit`)**: A framework for managing git hook scripts, used to automatically run linters and formatters (`black`, `isort`, `flake8`, `mypy`) before any code is committed.

## 2. Mathematical Formulas

### Forecasting Metrics

When comparing the performance between different forecasting models (like Naive, Prophet, and SARIMA), the following metrics are used:

- **MAE (Mean Absolute Error)**:
  - _Formula_: `(1/n) * Σ |y_actual - y_pred|`
  - _Why_: It gives the average magnitude of the errors in a set of predictions, without considering their direction. It is robust to outliers.
- **RMSE (Root Mean Squared Error)**:
  - _Formula_: `√[ (1/n) * Σ (y_actual - y_pred)² ]`
  - _Why_: By squaring the errors before averaging, RMSE mathematically penalizes large errors much more heavily than MAE. It is useful when large miscalculations are particularly undesirable.
- **MAPE (Mean Absolute Percentage Error)**:
  - _Formula_: `(1/n) * Σ |(y_actual - y_pred) / y_actual| * 100`
  - _Why_: Expresses the error as a percentage, making it easy to understand across different scales of data (e.g., an error of "5%" is universally understood).
- **R² (R-squared / Coefficient of Determination)**:
  - _Formula_: `1 - (SS_res / SS_tot)`
  - _Why_: It represents the proportion of the variance in the dependent variable that is predictable from the independent variable. An R² close to 1.0 means the model perfectly fits the data.

### Inventory Optimization

- **Dynamic / ML-Enhanced EOQ**:
  - _Formula_: `√(2 * D_t * S / H_t)`
    - `D_t` = Forecasted Demand for specific time period `t`
    - `S` = Ordering Cost per order
    - `H_t` = Holding Cost per unit for specific time period `t`
  - _Why_: As proposed in modern Hybrid Supply Chain literature (e.g., _Optimizing Economic Order Quantity Using Machine Learning_, 2024), standard EOQ fails under the assumption of constant, static demand. By replacing annual `D` with highly accurate, seasonal time-series forecasts (from Prophet/SARIMA/LSTM), the EOQ model becomes dynamic. It adapts order quantities period-by-period.
  - _Scholarly Relevance_: Research in 2024-2025 emphasizes using generative seasonality models (like Prophet) paired with dynamic EOQ logic to address uncertain supply chains and variable prices. Dynamic forecasting solves the "cold-start" problem and drastically outperforms static averages in high-mix, non-linear environments.

### System Safety & Risk Management

- **Safety Stock (Buffer for the Bullwhip Effect)**:
  - _Formula_: `Z * σ_forecast * √(L)`
    - `Z` = Z-score based on desired service level
    - `σ_forecast` = The error/uncertainty provided directly by the ML model (e.g., RMSE or Prophet's uncertainty bounds)
    - `L` = Lead Time
  - _Why_: When switching to Dynamic EOQ, businesses risk "chasing" every minor ML forecast fluctuation, creating massive supply chain turbulence known as the _Bullwhip Effect_. To mathematically absorb this shock, Safety Stock relies on the ML model's own proven standard deviation (`σ_forecast`) rather than a simple historical standard deviation.
