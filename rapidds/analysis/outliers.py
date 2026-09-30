import numpy as np
import pandas as pd


def detect_outliers(series, method="iqr", threshold=3.0):
    s = pd.Series(series)
    if not pd.api.types.is_numeric_dtype(s):
        raise TypeError("Outlier detection requires numeric data")
    if method == "iqr":
        q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
        mask = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)
        bounds = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    elif method == "zscore":
        mean, std = s.mean(), s.std()
        z = (s - mean) / std if std else pd.Series(0, index=s.index)
        mask = z.abs() > threshold
        bounds = (mean - threshold * std, mean + threshold * std)
    elif method == "modified_zscore":
        median = s.median(); mad = np.median(np.abs(s.dropna() - median))
        score = 0.6745 * (s - median) / mad if mad else pd.Series(0, index=s.index)
        mask = score.abs() > threshold
        bounds = (None, None)
    else:
        raise ValueError("method must be 'iqr', 'zscore', or 'modified_zscore'")
    return {"mask": mask.fillna(False), "count": int(mask.sum()), "bounds": bounds, "method": method}
