# SmartStock Project References

This document maintains a comprehensive, cited reference list for all external tools, Python libraries, mathematical formulas, and academic research underpinning the SmartStock platform. Each entry includes the primary source, version (where applicable), codebase location, and the scholarly justification for its inclusion.

---

## 1. Core Python Libraries

### 1.1 Pandas (`pandas`)

- **Version used:** `2.3.3`
- **PyPI:** https://pypi.org/project/pandas/
- **GitHub:** https://github.com/pandas-dev/pandas
- **Used in:** `smartstock/data/loader.py`, `smartstock/data/cleaner.py`, `smartstock/data/features.py`, `smartstock/forecasting/`, `smartstock/optimization/eoq_calculator.py`
- **Why:** Pandas is the foundational library for all DataFrame operations in SmartStock. Used for loading raw CSV sales data, resampling time series to daily frequency, interpolating missing values, IQR-based outlier capping, and constructing the final EOQ output DataFrames. Its `DatetimeIndex` and `resample()` API are central to the entire pipeline.
- **Citation:** McKinney, W. (2010). Data structures for statistical computing in Python. _Proceedings of the 9th Python in Science Conference_, 445, 51–56. https://doi.org/10.25080/Majora-92bf1922-00a

---

### 1.2 NumPy (`numpy`)

- **Version used:** `1.26.4`
- **PyPI:** https://pypi.org/project/numpy/
- **GitHub:** https://github.com/numpy/numpy
- **Used in:** `smartstock/models/naive.py`, `smartstock/optimization/eoq_calculator.py`, `smartstock/forecasting/forecast_manager.py`
- **Why:** Used for all vectorized array operations — particularly `np.maximum()`, `np.sqrt()`, `np.ceil()`, and `np.full()` — enabling efficient, loop-free computation of EOQ, safety stock, and forecast metrics across thousands of periods simultaneously. **Performance considerations:** The vectorized NumPy operations in `EOQCalculator` are specifically mathematically optimized for large datasets, which is absolutely critical for enterprise-scale inventory systems processing millions of item-period combinations without running into python `for`-loop bottlenecks.
- **Citation:** Harris, C.R., Millman, K.J., van der Walt, S.J., et al. (2020). Array programming with NumPy. _Nature_, 585, 357–362. https://doi.org/10.1038/s41586-020-2649-2

---

### 1.3 SciPy (`scipy`)

- **Version used:** via `statsmodels` dependency
- **PyPI:** https://pypi.org/project/scipy/
- **GitHub:** https://github.com/scipy/scipy
- **Used in:** `smartstock/optimization/eoq_calculator.py` — `scipy.stats.norm.ppf(service_level)` for Z-score computation
- **Why:** The `stats.norm.ppf()` function converts a target service level (e.g., 0.95) into the corresponding Z-score used in the safety stock formula. This is mathematically correct and eliminates hardcoding Z-score lookup tables.
- **Citation:** Virtanen, P., Gommers, R., Oliphant, T.E., et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. _Nature Methods_, 17, 261–272. https://doi.org/10.1038/s41592-019-0686-2

---

### 1.4 Prophet (`prophet`)

- **Version used:** `1.3.0`
- **PyPI:** https://pypi.org/project/prophet/
- **GitHub:** https://github.com/facebook/prophet
- **Used in:** `smartstock/forecasting/prophet_forecaster.py`
- **Why:** Prophet was selected as the primary ML forecasting model because it natively handles daily/weekly/yearly seasonality, missing data, and holiday effects with minimal manual configuration — critical properties for retail demand data. Its decomposable model structure (trend + seasonality + holidays) allows `ProphetForecaster.get_model_components()` to expose interpretable components to downstream analysts. The `yhat_upper` and `yhat_lower` uncertainty bounds are piped directly into the EOQ calculator as the `uncertainty_series` parameter.
- **Primary Paper:** Taylor, S.J., & Letham, B. (2018). Forecasting at scale. _The American Statistician_, 72(1), 37–45. https://doi.org/10.1080/00031305.2017.1380080. Preprint: https://peerj.com/preprints/3190.pdf
- **Applied retail paper:** Žunić, E., et al. (2020). Application of Facebook's Prophet Algorithm for Successful Sales Forecasting Based on Real-world Data. SSRN. https://ssrn.com/abstract=3603839

---

### 1.5 Statsmodels (`statsmodels`)

- **Version used:** `0.14.6`
- **PyPI:** https://pypi.org/project/statsmodels/
- **GitHub:** https://github.com/statsmodels/statsmodels
- **Used in:** `smartstock/forecasting/sarima_forecaster.py` — `statsmodels.tsa.statespace.sarimax.SARIMAX`
- **Why:** Statsmodels provides the SARIMAX class used to implement `SARIMAForecaster`. It was chosen for its rigorous statistical foundations, AIC/BIC model selection tools exposed via `get_model_summary()`, access to residuals via `get_residuals()`, and first-class support for confidence intervals on out-of-sample forecasts.
- **Citation:** Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. _Proceedings of the 9th Python in Science Conference_. https://doi.org/10.25080/Majora-92bf1922-011
- **Comparative retail paper:** Falatouri, T., Darbanian, F., Brandtner, P., & Udokwu, C. (2022). Predictive analytics for demand forecasting — A comparison of SARIMA and LSTM in retail SCM. _Procedia Computer Science_, 200, 993–1003. https://doi.org/10.1016/j.procs.2022.01.297

---

## 2. Testing Libraries

### 2.1 Pytest (`pytest`)

- **Version used:** `9.0.2`
- **Configured in:** `pytest.ini`
- **Why:** Pytest is the industry-standard Python test runner used to execute the entire SmartStock test suite. Custom markers (`unit`, `integration`, `property`, `slow`) defined in `pytest.ini` enable selective test execution.

---

### 2.2 Hypothesis (`hypothesis`)

- **Version used:** `6.109.0`
- **Configured in:** `pytest.ini` — `--hypothesis-show-statistics`
- **Why:** Hypothesis is used for property-based testing (PBT) throughout SmartStock. Rather than writing fixed example tests, PBT defines mathematical invariants that must hold for _all_ valid inputs, and Hypothesis automatically generates hundreds of edge-case inputs to try to falsify them. This is particularly valuable to mathematically enforce the optimization properties discussed in the SmartStock specification:

  - **Property 6: EOQ Calculation Preserves Invariants:** Validates that EOQ stays mathematically robust for any valid floating point costs/demand scenarios.
  - **Property 7: ROP Increases with Safety Stock:** Enforces monotonic increases in the reorder point functions across the numerical spectrum.
  - **Property 8: Safety Stock Increases with Service Level:** Tests that the cumulative distribution (Z-score mapping) always scales safety stocks correctly without causing overflows or unexpected NaN boundaries.

  Hypothesis naturally shrinks real-world failing inputs to their minimal reproducing case, making optimization logic incredibly resilient.

- **Primary Paper:** MacIver, D.R., Hatfield-Dodds, Z., et al. (2019). Hypothesis: A new approach to property-based testing. _Journal of Open Source Software_, 4(43), 1891. https://doi.org/10.21105/joss.01891
- **Empirical evaluation:** Ravi, S., & Coblenz, M. (2025). An empirical evaluation of property-based testing in Python. _Proceedings of the ACM on Programming Languages_, 9(OOPSLA2), Article 412. https://doi.org/10.1145/3764068

---

### 2.3 pytest-cov (`pytest-cov`)

- **Version used:** `5.0.0`
- **Configured in:** `pytest.ini`
- **Why:** Generates line and branch coverage reports for the `smartstock` package.

---

## 3. Code Quality & Tooling

### 3.1 Pre-commit (`pre-commit`)

- **Version used:** `3.7.1`
- **Configured in:** `.pre-commit-config.yaml`
- **Why:** Enforces code quality gates automatically on every `git commit`. The SmartStock configuration runs: `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-toml`, `check-json`, `check-merge-conflict`, `black`, `isort`, `flake8`, and `mypy`.

### 3.2 Black & isort

- **Why:** Opinionated formatting. Black handles line lengths, isort sorts dependencies safely.

### 3.3 Flake8 & Mypy

- **Why:** Flake8 acts as the primary linter for PEP8 violations. Mypy is triggered in `--strict` mode ensuring all Python calculations have rock-solid static types across `models`, `forecasting`, and `optimization` folders.

---

## 4. Mathematical Formulas & Theoretical Foundations

### 4.1 Forecasting Evaluation Metrics

Used in `ForecastManager._calculate_metrics()`, `_calculate_mape()`, and `_calculate_r2()`.

**MAE (Mean Absolute Error):** Robust to outliers. Primary metric for human-interpretable error magnitude.
**RMSE (Root Mean Squared Error):** Penalizes large errors heavily. Useful when large stockouts are unacceptable.
**MAPE:** Scale-independent percentage error.
**R²:** Proportion of variance mapped from real sales vs prediction traces.

- **Reference:** Hyndman, R.J., & Koehler, A.B. (2006). Another look at measures of forecast accuracy. _International Journal of Forecasting_, 22(4), 679–688.

---

### 4.2 Dynamic / ML-Enhanced EOQ (Economic Order Quantity)

**Formula — implemented in `EOQCalculator.calculate()`:**

```
EOQ_t = √(2 * D_t * S / H_t)
```

Where:

- `D_t` = Forecasted demand for period `t`
- `S` = Ordering cost per order
- `H_t` = Holding cost per unit per period

**Data validation approach:** The `EOQCalculator` implements strict, early input validation for negative or zero-bound costs and strict bounds checking on service levels (`0.01 <= service_level <= 0.999`). This rigorously prevents garbage-in/garbage-out scenarios such as accidental negative square roots or infinite z-scores from propagating silently through the mathematical matrix.

**Why dynamic rather than classical EOQ:** The classical EOQ model assumes constant, static annual demand. SmartStock replaces this with period-by-period forecasted demand from Prophet or SARIMA, making EOQ adaptive to seasonal fluctuations and trend changes.

**Scholarly context:**

- Harris, F.W. (1913). How many parts to make at once. _Factory, The Magazine of Management_, 10(2), 135–136. _(Original EOQ paper.)_
- Sachdeva, A., et al. (2024). AI-enhanced inventory and demand forecasting. _World Journal of Advanced Research and Reviews_, 23(1), 1931–1944.

---

### 4.3 Safety Stock with ML Uncertainty (Bullwhip Effect Mitigation)

**Formula — implemented in `EOQCalculator.calculate()`:**

```
Safety Stock = Z * σ_forecast * √(L)
```

**Why use the model's own uncertainty:** Naively following each forecast fluctuation amplifies ordering variability across the supply chain — the Bullwhip Effect. By anchoring safety stock to `σ_forecast` (the ML model's own proven prediction interval width), SmartStock mathematically absorbs uncertainty at the correct scale rather than overreacting to noise.

**Foundational Bullwhip Effect research:**

- Lee, H.L., Padmanabhan, V., & Whang, S. (1997). Information distortion in a supply chain: The bullwhip effect. _Management Science_, 43(4), 546–558. https://doi.org/10.1287/mnsc.43.4.546

---

### 4.4 Time Series Decomposition & Stationarity

Used as the theoretical basis for both `ProphetForecaster` (additive decomposition) and `SARIMAForecaster` (differencing for stationarity).
**Reference:** Box, G.E.P., Jenkins, G.M., Reinsel, G.C., & Ljung, G.M. (2015). _Time Series Analysis: Forecasting and Control_ (5th ed.). Wiley. ISBN: 978-1-118-67502-1.

---

### 4.5 IQR-Based Outlier Capping

Used in `smartstock/data/cleaner.py`. Values above `upper_bound = Q3 + 1.5 * IQR` are clipped to prevent arbitrary demand spikes.
**Reference:** Tukey, J.W. (1977). _Exploratory Data Analysis_. Addison-Wesley. ISBN: 978-0-201-07616-5.

---

### 4.6 ABC Analysis (Pareto Principle)

Used in `smartstock/optimization/abc_analyzer.py`.

**Logic:**
Products are ranked by their annual financial value (Annual Demand × Unit Cost). The inventory is then divided into three classes based on cumulative value:

- **A Items:** Top 80% of total financial value (The Critical Few)
- **B Items:** Next 15% of total financial value (The Middle Ground)
- **C Items:** Bottom 5% of total financial value (The Trivial Many)

**Why ABC Analysis:** Managing every item in a catalog with the same rigorous ML forecasting and dynamic EOQ math is computationally expensive and practically impossible for human analysts. ABC Analysis uses the Pareto Principle to isolate the high-impact items so attention and ML compute can be targeted where they matter most.

**References:**

- _Pareto Principle (80/20 rule):_ Pareto, V. (1896). Cours d'Économie Politique.
- _ABC Analysis in inventory:_ Flores, B.E., & Whybark, D.C. (1987). Implementing multiple criteria ABC analysis. _Journal of Operations Management_, 7(1-2), 79-85.

---

## 5. Dataset Reference

SmartStock is built around the **Kaggle Store Item Demand Forecasting Challenge** dataset:

- **Source:** https://www.kaggle.com/competitions/demand-forecasting-kernels-only
- **Format:** Daily sales data across 10 stores and 50 items (500 store-item combinations), spanning 5 years.

---

## 6. Summary of Key Research Papers

| Paper                    | Year | Relevance to SmartStock                                               |
| ------------------------ | ---- | --------------------------------------------------------------------- |
| Taylor & Letham          | 2018 | Theoretical foundation of `ProphetForecaster`                         |
| Falatouri et al.         | 2022 | Empirical justification for SARIMA's strength on seasonal retail data |
| Arunraj et al.           | 2016 | Retail SARIMA application and use of external variables               |
| Lee, Padmanabhan & Whang | 1997 | Foundational theory behind safety stock design and Bullwhip           |
| MacIver et al.           | 2019 | Justification for Hypothesis use in test suite                        |
| Ravi & Coblenz           | 2025 | Quantitative evidence that PBT finds 50× more bugs than unit tests    |
| Harris et al.            | 2020 | Vectorized numpy performance and metric computation                   |
| Hyndman & Koehler        | 2006 | Theoretical grounding for MAE, RMSE, MAPE, R²                         |
