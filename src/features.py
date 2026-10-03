import pandas as pd
import numpy as np


def preprocess(df):
    # Sort properly
    df = df.sort_values(["city", "year", "weekofyear"]).reset_index(drop=True)

    # Select ONLY numeric climate columns
    climate_cols = [
        col for col in df.columns if col not in ["city", "year", "weekofyear", "cases"]
    ]

    # Ensure numeric types (fixes FutureWarning too)
    df[climate_cols] = df[climate_cols].infer_objects(copy=False)

    # Use transform (NOT apply)
    df[climate_cols] = df.groupby("city")[climate_cols].transform(
        lambda x: x.interpolate().ffill().bfill()
    )

    return df


def create_features(df):
    df = df.copy()

    # --- LAG FEATURES (MOST IMPORTANT) ---
    for lag in [1, 2, 3, 4]:
        df[f"cases_lag_{lag}"] = df.groupby("city")["cases"].shift(lag)

    # --- MOVING AVERAGE ---
    df["cases_ma_4"] = df.groupby("city")["cases"].transform(
        lambda x: x.rolling(4).mean()
    )

    # --- SEASONALITY (FOURIER FEATURES) ---
    df["week_sin"] = np.sin(2 * np.pi * df["weekofyear"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["weekofyear"] / 52)

    # --- CLIMATE LAGS (BIOLOGICAL DELAY) ---
    climate_cols = [
        "reanalysis_air_temp_k",
        "reanalysis_specific_humidity_g_per_kg",
        "precipitation_amt_mm",
    ]

    for col in climate_cols:
        if col in df.columns:
            df[f"{col}_lag_2"] = df.groupby("city")[col].shift(2)

    # --- DROP ROWS WITH NaNs (from lagging) ---
    df = df.dropna().reset_index(drop=True)

    return df
