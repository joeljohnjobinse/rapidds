import numpy as np
import pandas as pd


def quality_checks(df: pd.DataFrame) -> dict:
    checks = []
    rows = len(df)
    for col in df.columns:
        s = df[col]
        missing = int(s.isna().sum())
        if missing:
            checks.append({"check": "missing", "column": col, "count": missing, "ratio": missing / rows if rows else 0.0})
        if s.nunique(dropna=False) <= 1:
            checks.append({"check": "constant", "column": col, "count": 1})
        if pd.api.types.is_numeric_dtype(s):
            vals = s.dropna()
            if len(vals) >= 4:
                q1, q3 = vals.quantile([0.25, 0.75])
                iqr = q3 - q1
                if iqr > 0:
                    count = int(((vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)).sum())
                    if count:
                        checks.append({"check": "outliers", "column": col, "count": count, "ratio": count / len(vals)})
        if s.dtype == "object":
            values = s.dropna().astype(str)
            if not values.empty:
                numeric_like = pd.to_numeric(values, errors="coerce").notna().mean()
                if numeric_like >= 0.8:
                    checks.append({"check": "numeric_string", "column": col, "ratio": float(numeric_like)})
                if values.str.strip().nunique() < values.nunique():
                    checks.append({"check": "whitespace", "column": col})
                if values.str.lower().nunique() < values.nunique():
                    checks.append({"check": "casing", "column": col})
    duplicates = int(df.duplicated().sum())
    if duplicates:
        checks.append({"check": "duplicates", "count": duplicates, "ratio": duplicates / rows if rows else 0.0})
    return {"checks": checks, "passed": not checks, "issue_count": len(checks)}
