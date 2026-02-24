from smartstock.data.loader import load_raw, filter_series
from smartstock.data.cleaner import clean_series
from smartstock.data.features import add_time_features

def run_pipeline():
    print("Loading raw data...")
    df = load_raw("data/raw/train.csv")

    print("Filtering Store 1, Item 1...")
    s1i1 = filter_series(df, store_id=1, item_id=1)
    print(f"Original sequence length: {len(s1i1)}")

    print("Cleaning data (filling gaps, capping outliers)...")
    cleaned = clean_series(s1i1)

    print("Adding time features...")
    final = add_time_features(cleaned)

    print("\n--- Pipeline Results ---")
    print(final.head())
    print("\nColumns:", final.columns.tolist())
    print("Shape:", final.shape)
    print("---Done---")

if __name__ == "__main__":
    run_pipeline()