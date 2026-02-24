import pandas as pd
import numpy as np

def clean_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a single-item store-item series
    Assumes 'date' is the index

    Steps:
    1. Resample to daily frequency (fill missing details)
    2. Handle negative sales
    3. Fill missing sales values (interpolation)
    4. Cap outliers at 1.5x IQR
    """
    cleaned = df.resample('D').asfreq()

    cleaned['sales'] = cleaned['sales'].clip(lower=0)
    cleaned['sales'] = cleaned['sales'].interpolate(method='linear')

    q1 = cleaned['sales'].quantile(0.25)
    q3 = cleaned['sales'].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr

    cleaned['sales'] = cleaned['sales'].clip(upper=upper_bound)

    return cleaned