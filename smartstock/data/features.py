import pandas as pd

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from the index (which must be a DatetimeIndex).

    Returns:
    A DataFrame with new columns: day_of_week, month, is_weekend
    """
    df_feat = df.copy()

    df_feat['day_of_week'] = df_feat.index.dayofweek
    df_feat['month'] = df_feat.index.month
    df_feat['is_weekend'] = df_feat['day_of_week'].isin([5, 6]).astype(int)

    return df_feat