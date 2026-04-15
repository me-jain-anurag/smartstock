# SmartStock Workflows

## 1) Data upload workflow

Input: CSV with `date`, `store`, `item`, `sales`

Behavior:
- Validates CSV readability and non-empty input
- Accepts common aliases (for example `Date`, `qty`, `item_id`) and remaps them
- Casts data types to canonical forms for downstream processing
- Stores validated dataframe in Streamlit session state

Output:
- Session-scoped dataset available to Forecasting and Optimization pages

## 2) Forecasting workflow

Input:
- Uploaded dataset
- Selected store and item
- Model (`prophet`, `sarima`, `naive`)
- Forecast horizon

Behavior:
- Filters selected series
- Cleans series
- Trains model and computes evaluation metrics
- Produces future forecast with confidence bounds

Output:
- Forecast chart and metrics (MAE, RMSE, MAPE, R²)
- Downloadable forecast CSV

## 3) EOQ optimization workflow

Input:
- Forecast output from Forecasting page
- Ordering cost, holding cost, lead time, batch size, service level

Behavior:
- Computes period-level EOQ
- Computes safety stock (optionally from forecast uncertainty)
- Computes reorder point and total order quantity

Output:
- EOQ summary metrics
- Period-level optimization table
- Downloadable EOQ CSV

## 4) ABC analysis workflow

Input: CSV with `item_id`, `unit_cost`, `annual_demand`

Behavior:
- Calculates annual inventory value
- Sorts by cumulative contribution
- Assigns A/B/C class using Pareto-style thresholds

Output:
- Category counts and value distribution chart
- Styled table with categories and cumulative contribution
- Downloadable ABC CSV

## 5) EOQ Explorer workflow

Input: slider-driven scenario parameters (no CSV required)

Behavior:
- Calculates EOQ, safety stock, reorder point, and annual cost live
- Displays formula-level outputs for learning and what-if analysis

Output:
- Instant inventory policy metrics for scenario exploration
