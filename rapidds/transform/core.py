import numpy as np
import pandas as pd


def scale(df, columns=None, method="standard"):
    out = df.copy(); columns = list(columns or out.select_dtypes(include=np.number).columns)
    for col in columns:
        s = out[col]
        if method == "standard":
            std = s.std(); out[col] = (s - s.mean()) / std if std else 0.0
        elif method == "minmax":
            span = s.max() - s.min(); out[col] = (s - s.min()) / span if span else 0.0
        else: raise ValueError("method must be 'standard' or 'minmax'")
    return out


def encode_categorical(df, columns=None, drop_first=False):
    cols = list(columns or df.select_dtypes(exclude=np.number).columns)
    return pd.get_dummies(df, columns=cols, drop_first=drop_first)


def datetime_features(df, columns):
    out = df.copy()
    for col in columns:
        dt = pd.to_datetime(out[col], errors="coerce")
        out[f"{col}_year"] = dt.dt.year
        out[f"{col}_month"] = dt.dt.month
        out[f"{col}_day"] = dt.dt.day
        out[f"{col}_weekday"] = dt.dt.weekday
        out[f"{col}_is_weekend"] = dt.dt.weekday >= 5
    return out
