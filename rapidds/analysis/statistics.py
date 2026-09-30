import pandas as pd
import numpy as np


def describe(df, columns=None):
    return df[columns].describe(include="all") if columns else df.describe(include="all")


def correlation(df, method="pearson"):
    return df.select_dtypes(include=np.number).corr(method=method)


def covariance(df):
    return df.select_dtypes(include=np.number).cov()


def bootstrap(series, statistic="mean", n_boot=1000, random_state=42):
    values = pd.Series(series).dropna().to_numpy()
    if len(values) == 0:
        raise ValueError("Cannot bootstrap an empty series")
    rng = np.random.default_rng(random_state)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True)
    if statistic == "mean":
        stats = samples.mean(axis=1)
    elif statistic == "median":
        stats = np.median(samples, axis=1)
    else:
        raise ValueError("statistic must be 'mean' or 'median'")
    return {"statistic": statistic, "estimate": float(getattr(np, statistic)(values)), "samples": stats, "ci_95": (float(np.quantile(stats, .025)), float(np.quantile(stats, .975)))}


def t_test(a, b, equal_var=True):
    from scipy import stats
    result = stats.ttest_ind(pd.Series(a).dropna(), pd.Series(b).dropna(), equal_var=equal_var)
    return {"test": "independent t-test", "statistic": float(result.statistic), "p_value": float(result.pvalue)}


def chi_square(table):
    from scipy import stats
    result = stats.chi2_contingency(table)
    return {"test": "chi-square", "statistic": float(result.statistic), "p_value": float(result.pvalue), "dof": int(result.dof)}


def anova(*groups):
    from scipy import stats
    result = stats.f_oneway(*[pd.Series(g).dropna() for g in groups])
    return {"test": "one-way ANOVA", "statistic": float(result.statistic), "p_value": float(result.pvalue)}
