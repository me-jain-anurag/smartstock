# SmartStock

SmartStock is an AI-assisted inventory planning project that combines demand forecasting with inventory optimization.

It provides:
- **Demand forecasting** (Prophet, SARIMA, and Naive baseline)
- **EOQ optimization** with safety stock and reorder point calculations
- **ABC analysis** for product prioritization
- **Interactive Streamlit dashboard** for end-to-end workflow

## Project structure

```text
smartstock/
├── smartstock/
│   ├── data/            # Loading, cleaning, feature engineering
│   ├── forecasting/     # Forecast manager + Prophet/SARIMA wrappers
│   ├── models/          # Base model contracts + naive model
│   ├── optimization/    # EOQ calculator + ABC analyzer
│   ├── dashboard/       # Streamlit app and pages
│   └── api/             # API package scaffold (Phase 2)
├── tests/               # Unit, integration, and property-based tests
├── scripts/             # Utility scripts (sample data generation)
├── references.md        # Research citations and formula references
└── requirements*.txt    # Runtime and development dependencies
```

## Prerequisites

- Python **3.9+** (3.11+ recommended for tooling compatibility)
- `pip`

## Installation

### Runtime only

```bash
pip install -r requirements.txt
```

### Development/testing setup

```bash
pip install -r requirements.txt
pip install pytest pytest-cov hypothesis pre-commit black isort flake8 mypy
```

## Run the dashboard

From the repository root:

```bash
streamlit run smartstock/dashboard/app.py
```

Then open the local URL shown in the terminal.

## Sample data

Generate dashboard test datasets:

```bash
python scripts/generate_sample_data.py
```

This creates files under `data/raw/`, including:
- `sample_quick.csv` (small, fast to test)
- `sample_test.csv` (larger scale dataset)
- `sample_abc.csv` (ABC analysis schema)

## Data schemas

### Forecasting/EOQ workflow CSV
Required columns:
- `date`
- `store`
- `item`
- `sales`

### ABC analysis CSV
Required columns:
- `item_id`
- `unit_cost`
- `annual_demand`

## Testing and quality checks

Run tests:

```bash
python -m pytest
```

Run repository linters/hooks:

```bash
pre-commit run --all-files
```

## Documentation

- `references.md` — scientific and library references backing formulas and implementation choices
- `docs/architecture.md` — package responsibilities and execution flow
- `docs/workflows.md` — dashboard workflows and expected inputs/outputs

## Current status

- Dashboard workflow is production-ready for local use.
- REST API page is currently documented in-app as **Coming Soon** (planned Phase 2 implementation).
