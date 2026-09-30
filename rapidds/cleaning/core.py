import pandas as pd
import numpy as np


def fill_missing(df, numeric="median", categorical="mode"):
    out = df.copy()
    actions = []
    for col in out.columns:
        if not out[col].isna().any():
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            if numeric in {"mean", "median"}:
                value = getattr(out[col], numeric)()
                if pd.notna(value):
                    out[col] = out[col].fillna(value); actions.append({"action": "impute", "column": col, "strategy": numeric, "value": float(value)})
        elif categorical == "mode":
            mode = out[col].mode(dropna=True)
            if not mode.empty:
                out[col] = out[col].fillna(mode.iloc[0]); actions.append({"action": "impute", "column": col, "strategy": "mode", "value": mode.iloc[0]})
    return out, actions


def standardize_text(df, columns=None, case=None, strip=True):
    out = df.copy(); columns = columns or out.select_dtypes(include="object").columns
    actions = []
    for col in columns:
        s = out[col].astype("string")
        before = s.copy()
        if strip: s = s.str.strip()
        if case == "lower": s = s.str.lower()
        elif case == "upper": s = s.str.upper()
        elif case == "title": s = s.str.title()
        out[col] = s
        if not s.equals(before): actions.append({"action": "standardize_text", "column": col, "case": case, "strip": strip})
    return out, actions
