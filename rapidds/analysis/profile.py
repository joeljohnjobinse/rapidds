import numpy as np
import pandas as pd


def profile_dataframe(df: pd.DataFrame) -> dict:
    rows, cols = df.shape
    columns = {}
    for col in df.columns:
        s = df[col]
        info = {
            "dtype": str(s.dtype),
            "non_null": int(s.notna().sum()),
            "missing": int(s.isna().sum()),
            "missing_ratio": float(s.isna().mean()) if rows else 0.0,
            "unique": int(s.nunique(dropna=True)),
            "unique_ratio": float(s.nunique(dropna=True) / rows) if rows else 0.0,
        }
        if pd.api.types.is_numeric_dtype(s):
            desc = s.describe()
            info.update({k: float(desc[k]) for k in ["mean", "std", "min", "25%", "50%", "75%", "max"] if k in desc and pd.notna(desc[k])})
            info["skewness"] = float(s.skew()) if s.notna().sum() > 2 else 0.0
            info["kurtosis"] = float(s.kurtosis()) if s.notna().sum() > 3 else 0.0
        else:
            top = s.dropna().astype(str).value_counts().head(5)
            info["top_values"] = top.to_dict()
        columns[str(col)] = info

    duplicate_rows = int(df.duplicated().sum())
    return {
        "shape": {"rows": rows, "columns": cols},
        "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
        "duplicate_rows": duplicate_rows,
        "duplicate_ratio": float(duplicate_rows / rows) if rows else 0.0,
        "columns": columns,
        "numeric_columns": df.select_dtypes(include=np.number).columns.tolist(),
        "categorical_columns": df.select_dtypes(exclude=np.number).columns.tolist(),
    }
