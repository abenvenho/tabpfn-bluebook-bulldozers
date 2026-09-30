"""Feature transformations.

- clean_table: the 2013-style cleaning given to the `clean` TabPFN arm.
- engineer_lgbm: cleaning + categorical dtypes for the LightGBM baseline.
- join_appendix: the `appendix` arm (Machine_Appendix "corrected" attributes).
The `raw` arm bypasses this module entirely.
"""
import numpy as np
import pandas as pd

from .schema import (APPENDIX_KEY, APPENDIX_USE_COLS, DATE_COL, DATE_PARSED,
                     ID_COL, NUMERIC_COLS, TARGET)


def add_date_parts(df: pd.DataFrame) -> pd.DataFrame:
    d = df[DATE_PARSED]
    df["saleYear"] = d.dt.year
    df["saleMonth"] = d.dt.month
    df["saleDayOfWeek"] = d.dt.dayofweek
    df["saleElapsedDays"] = (d - pd.Timestamp("1989-01-01")).dt.days
    return df


def clean_table(df: pd.DataFrame) -> pd.DataFrame:
    """2013-style cleaning: plausible years, zero hour meters to missing, age at sale,
    decomposed sale date. Returns a new frame without `saledate` (original or parsed)."""
    out = df.copy()
    out.loc[out["YearMade"] < 1900, "YearMade"] = np.nan
    out.loc[out["MachineHoursCurrentMeter"] == 0, "MachineHoursCurrentMeter"] = np.nan
    out = add_date_parts(out)
    out["AgeAtSale"] = (out["saleYear"] - out["YearMade"]).clip(lower=0)
    return out.drop(columns=[DATE_COL, DATE_PARSED], errors="ignore")


def join_appendix(df: pd.DataFrame, appendix: pd.DataFrame) -> pd.DataFrame:
    """Raw table + the machine appendix's "corrected" attributes (suffix _app)."""
    use = [c for c in APPENDIX_USE_COLS if c in appendix.columns]
    app = appendix[use].drop_duplicates(subset=[APPENDIX_KEY])
    out = df.merge(app, on=APPENDIX_KEY, how="left", suffixes=("", "_app"))
    return out


def engineer_lgbm(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaned table with object columns as pandas `category` (LightGBM native)."""
    out = clean_table(df)
    for c in out.columns:
        if c in (TARGET, ID_COL):
            continue
        if out[c].dtype == object:
            out[c] = out[c].astype("category")
    return out


def to_matrix(df: pd.DataFrame, feature_cols=None):
    """(X, feature_cols) with target/id removed."""
    drop = [c for c in (TARGET, ID_COL) if c in df.columns]
    X = df.drop(columns=drop)
    if feature_cols is not None:
        for c in feature_cols:
            if c not in X.columns:
                X[c] = np.nan
        X = X[feature_cols]
    return X, list(X.columns)


def ordinal_encode(train_X: pd.DataFrame, *others):
    """Shared integer codes for object/category columns (for the mock predictor)."""
    frames = [train_X] + [o for o in others if o is not None]
    encoded = [f.copy() for f in frames]
    for c in train_X.columns:
        if train_X[c].dtype == object or str(train_X[c].dtype) == "category":
            cats = pd.Categorical(pd.concat([f[c].astype(object) for f in frames]))
            offset = 0
            for i, f in enumerate(encoded):
                n = len(frames[i])
                f[c] = cats.codes[offset:offset + n]
                offset += n
        elif np.issubdtype(train_X[c].dtype, np.datetime64):
            for f in encoded:
                f[c] = pd.to_datetime(f[c]).astype("int64") // 10**9
    return encoded if len(encoded) > 1 else encoded[0]
