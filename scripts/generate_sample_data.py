"""
Generate sample CSV files for SmartStock dashboard testing.
Run from the project root:
    python scripts/generate_sample_data.py

Outputs (written to data/raw/):
  Core datasets
  ─────────────────────────────────────────────────────
  sample_quick.csv             — 4,380 rows (2 stores × 3 items × 730d), fast test
  sample_test.csv              — full dataset (20 stores × 50 items × 730d)
  sample_fuzzy_columns.csv     — aliased column names (auto-detection test)
  sample_bad_columns.csv       — unrecognisable column names (error case)
  sample_empty.csv             — headers only, no rows (warning case)

  Edge-case datasets
  ─────────────────────────────────────────────────────
  sample_date_gaps.csv         — random missing dates per store/item
  sample_negative_stock.csv    — stock_level contains 0 and negative values
  sample_duplicates.csv        — deliberate duplicate rows
  sample_mixed_dates.csv       — mixed date formats in one column

  ⚠ Known issue — sample_mixed_dates.csv:
    The mm/dd/yyyy and dd/mm/yyyy formats are ambiguous when day ≤ 12.
    pandas pd.to_datetime(errors='coerce') will silently misparse some of
    these rows and produce NaT instead of raising an error.
    The upload page does not yet warn the user when this happens.
    Future improvement: in service._cast_types(), check what fraction of
    'date' values became NaT after coercion. If above a threshold (e.g. 5%),
    surface a warning banner so the user knows to check their date format.
    Tracked: smartstock/dashboard/service.py — _cast_types TODO comment.
"""

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
random.seed(42)

N_STORES = 20
N_ITEMS = 50
DATES = pd.date_range("2020-01-01", periods=365 * 2, freq="D")
N_DATES = len(DATES)

item_base_sales = {i: np.random.randint(10, 100) for i in range(1, N_ITEMS + 1)}
item_base_price = {
    i: round(np.random.uniform(2.99, 299.99), 2) for i in range(1, N_ITEMS + 1)
}
item_reorder_pt = {i: np.random.randint(20, 80) for i in range(1, N_ITEMS + 1)}


# ── Row generator ─────────────────────────────────────────────────────────────
def build_rows(
    stores: range = range(1, N_STORES + 1),
    items: range = range(1, N_ITEMS + 1),
    dates: list[Any] = DATES,
) -> list[dict[str, Any]]:
    """Return a list of dicts (one per store × item × date)."""
    rows: list[dict[str, Any]] = []
    t = np.arange(len(dates))

    for store in stores:
        for item in items:
            base = item_base_sales[item]
            trend = np.linspace(0, 8, len(dates))
            seasonal = 5 * np.sin(2 * np.pi * t / 7)
            seasonal += 10 * np.sin(2 * np.pi * t / 365)
            noise = np.random.normal(0, 3, len(dates))
            raw_sales = np.maximum(0, base + trend + seasonal + noise)
            sales = raw_sales.round().astype(int)

            reorder = item_reorder_pt[item]
            stock = np.zeros(len(dates), dtype=int)
            stock[0] = np.random.randint(reorder + 20, reorder + 120)
            for idx in range(1, len(dates)):
                level = stock[idx - 1] - sales[idx]
                if idx % 7 == 0 and level < reorder:
                    level += np.random.randint(reorder, reorder * 3)
                stock[idx] = level

            stockout = stock <= 0
            promo_mask = np.random.rand(len(dates)) < 0.10
            discount = np.where(
                promo_mask,
                np.round(np.random.uniform(0.05, 0.40, len(dates)), 2),
                0.0,
            )
            base_price = item_base_price[item]
            actual_price = np.round(base_price * (1 - discount), 2)

            for idx, d in enumerate(dates):
                rows.append(
                    {
                        "date": d.strftime("%Y-%m-%d"),
                        "store": store,
                        "item": item,
                        "sales": int(sales[idx]),
                        "stock_level": int(stock[idx]),
                        "reorder_point": reorder,
                        "stockout": bool(stockout[idx]),
                        "on_promotion": bool(promo_mask[idx]),
                        "discount_pct": float(discount[idx]),
                        "price": float(actual_price[idx]),
                    }
                )

    return rows


# ── Build base dataframe ───────────────────────────────────────────────────────
print("Building base dataset (20 stores × 50 items × 730 days) …")
rows = build_rows()
good = pd.DataFrame(rows)
total = len(good)
print(f"  → {total:,} rows generated\n")


# ── 1. Quick CSV (small — for fast UI testing) ────────────────────────────────
quick = good[(good["store"].isin([1, 2])) & (good["item"].isin([1, 2, 3]))].copy()
quick.to_csv(OUTPUT_DIR / "sample_quick.csv", index=False)
print(f"[OK] sample_quick.csv             — {len(quick):,} rows (2 stores × 3 items)")

# ── 2. Full good CSV ──────────────────────────────────────────────────────────
good.to_csv(OUTPUT_DIR / "sample_test.csv", index=False)
print(f"[OK] sample_test.csv              — {total:,} rows, exact columns")

# ── 3. Fuzzy columns ──────────────────────────────────────────────────────────
fuzzy = good.rename(
    columns={
        "date": "Date",
        "store": "Store",
        "item": "item_id",
        "sales": "qty",
        "stock_level": "StockQty",
        "reorder_point": "ReorderThreshold",
        "stockout": "out_of_stock",
        "on_promotion": "IsPromo",
        "discount_pct": "DiscountRate",
        "price": "UnitPrice",
    }
)
fuzzy.to_csv(OUTPUT_DIR / "sample_fuzzy_columns.csv", index=False)
print("[OK] sample_fuzzy_columns.csv     — aliased column names (auto-detection test)")

# ── 4. Bad columns ────────────────────────────────────────────────────────────
bad = good.rename(columns={c: f"col_{chr(97 + i)}" for i, c in enumerate(good.columns)})
bad.to_csv(OUTPUT_DIR / "sample_bad_columns.csv", index=False)
print("[OK] sample_bad_columns.csv       — unrecognisable column names (error case)")

# ── 5. Empty CSV ──────────────────────────────────────────────────────────────
pd.DataFrame(columns=good.columns).to_csv(OUTPUT_DIR / "sample_empty.csv", index=False)
print("[OK] sample_empty.csv             — headers only, no rows (warning case)")

# ── 6. Date-gaps CSV ──────────────────────────────────────────────────────────
gap_mask = np.random.rand(total) > 0.15
date_gaps = good[gap_mask].copy()
date_gaps.to_csv(OUTPUT_DIR / "sample_date_gaps.csv", index=False)
n_dropped = total - len(date_gaps)
print(
    f"[OK] sample_date_gaps.csv         — {n_dropped:,} rows removed (~15% date gaps)"
)

# ── 7. Negative / zero stock CSV ─────────────────────────────────────────────
neg_stock = good.copy()
neg_idx = neg_stock.sample(frac=0.05, random_state=7).index
neg_stock.loc[neg_idx, "stock_level"] = np.random.randint(-50, 1, size=len(neg_idx))
neg_stock.to_csv(OUTPUT_DIR / "sample_negative_stock.csv", index=False)
print(f"[OK] sample_negative_stock.csv    — {len(neg_idx):,} rows with stock_level ≤ 0")

# ── 8. Duplicate rows CSV ─────────────────────────────────────────────────────
dup_sample = good.sample(n=min(5_000, total // 10), random_state=3)
duplicates = pd.concat([good, dup_sample], ignore_index=True).sample(
    frac=1, random_state=9
)
duplicates.to_csv(OUTPUT_DIR / "sample_duplicates.csv", index=False)
print(
    f"[OK] sample_duplicates.csv        — {len(dup_sample):,} duplicate rows injected"
)

# ── 9. Mixed date formats CSV ─────────────────────────────────────────────────
# ⚠ Known limitation: mm/dd/yyyy and dd/mm/yyyy are ambiguous when day ≤ 12.
# pandas silently misparsees these with errors='coerce' → NaT with no warning.
# Future improvement tracked in: smartstock/dashboard/service.py _cast_types TODO.
FORMATS = [
    "%Y-%m-%d",  # 2020-01-15  (ISO — unambiguous)
    "%d/%m/%Y",  # 15/01/2020  ← ambiguous when day ≤ 12
    "%m/%d/%Y",  # 01/15/2020  ← ambiguous when day ≤ 12
    "%d-%b-%Y",  # 15-Jan-2020
    "%B %d, %Y",  # January 15, 2020
    "%Y%m%d",  # 20200115
]
mixed = good.copy()
fmt_choices = np.random.choice(len(FORMATS), size=total)
mixed["date"] = [
    pd.to_datetime(d).strftime(FORMATS[f]) for d, f in zip(mixed["date"], fmt_choices)
]
mixed.to_csv(OUTPUT_DIR / "sample_mixed_dates.csv", index=False)
print(f"[OK] sample_mixed_dates.csv       — {len(FORMATS)} date formats mixed")

# ── 10. ABC Analysis CSV ──────────────────────────────────────────────────────
# Needs completely different schema: item_id, unit_cost, annual_demand
# NOT the same as the sales CSV — the Optimization page warns about this.
np.random.seed(99)
abc_items = []
for i in range(1, N_ITEMS + 1):
    # Mix of high-value/low-volume (A) and low-value/high-volume (C) items
    unit_cost = round(
        np.random.choice(
            [
                np.random.uniform(50, 500),  # expensive items
                np.random.uniform(5, 49),  # mid-range
                np.random.uniform(0.5, 4.9),  # cheap items
            ],
            p=[0.2, 0.3, 0.5],
        ),
        2,
    )
    annual_demand = int(
        np.random.choice(
            [
                np.random.randint(100, 800),  # low volume
                np.random.randint(800, 5000),  # medium volume
                np.random.randint(5000, 30000),  # high volume
            ],
            p=[0.2, 0.3, 0.5],
        )
    )
    abc_items.append(
        {
            "item_id": f"SKU{i:03d}",
            "unit_cost": unit_cost,
            "annual_demand": annual_demand,
        }
    )

pd.DataFrame(abc_items).to_csv(OUTPUT_DIR / "sample_abc.csv", index=False)
print(f"[OK] sample_abc.csv               — {len(abc_items)} items")

# ── Summary ───────────────────────────────────────────────────────────────────
print(
    f"""
Done. All CSVs written to {OUTPUT_DIR}/

Column reference (sales schemas)
  date           — YYYY-MM-DD transaction date
  store          — store ID (1–{N_STORES})
  item           — item ID (1–{N_ITEMS})
  sales          — units sold
  stock_level    — end-of-day stock on hand
  reorder_point  — threshold that triggers replenishment
  stockout       — True when stock_level ≤ 0 at start of day
  on_promotion   — True when a discount is active
  discount_pct   — fractional discount (0.0–0.40); 0 when no promo
  price          — actual unit selling price after discount

Column reference (sample_abc.csv — ABC Analysis only)
  item_id        — product identifier (e.g. SKU001)
  unit_cost      — cost per unit in Rs
  annual_demand  — units expected to sell in a year

For quick UI testing use sample_quick.csv (4,380 rows).
For performance/scale testing use sample_test.csv ({total:,} rows).
For ABC Analysis testing use sample_abc.csv (separate schema).
"""
)
