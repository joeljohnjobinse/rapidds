import matplotlib.pyplot as plt
import pandas as pd


def _get_df(data):
    """
    Internal helper to accept either a Dataset or a DataFrame.
    """
    if hasattr(data, "df"):
        return data.df
    if isinstance(data, pd.DataFrame):
        return data
    raise TypeError("Expected a Dataset or pandas DataFrame")


# -------------------------------------------------
# 1. Missing values visualization
# -------------------------------------------------
def plot_missing(data, show: bool = True):
    """
    Plot missing value percentage per column.
    """
    df = _get_df(data)

    missing_pct = df.isnull().mean() * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))

    if missing_pct.empty:
        ax.text(0.5, 0.5, "No missing values detected",
                ha="center", va="center")
        ax.axis("off")
    else:
        missing_pct.plot(kind="bar", ax=ax)
        ax.set_ylabel("Missing Percentage (%)")
        ax.set_title("Missing Values by Column")

    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax


# -------------------------------------------------
# 2. Target distribution
# -------------------------------------------------
def plot_target(data, target: str, show: bool = True):
    """
    Plot distribution of a target column.
    """
    df = _get_df(data)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found")

    fig, ax = plt.subplots(figsize=(6, 4))

    df[target].value_counts().plot(kind="bar", ax=ax)
    ax.set_title(f"Target Distribution: {target}")
    ax.set_ylabel("Count")
    ax.set_xlabel("Class")

    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax


# -------------------------------------------------
# 3. Numeric distributions
# -------------------------------------------------
def plot_distributions(data, max_cols: int = 6, show: bool = True):
    """
    Plot histograms for numeric columns (limited to max_cols).
    """
    df = _get_df(data)

    numeric_cols = df.select_dtypes(include="number").columns[:max_cols]

    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns to plot")

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(numeric_cols),
        figsize=(4 * len(numeric_cols), 4)
    )

    if len(numeric_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, numeric_cols):
        df[col].plot(kind="hist", ax=ax, bins=30)
        ax.set_title(col)
        ax.set_ylabel("Frequency")

    plt.tight_layout()

    if show:
        plt.show()

    return fig, axes
