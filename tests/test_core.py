import pandas as pd
from rapidds import Dataset


def sample_df():
    return pd.DataFrame({
        "A": [1, 2, None, 4],
        "B": ["x", None, "x", "x"],
        "C": [1, 1, 1, 1]
    })


def test_dataset_init():
    ds = Dataset(sample_df())
    assert ds.df.shape == (4, 3)


def test_analyze_detects_missing():
    ds = Dataset(sample_df())
    report = ds.analyze()
    assert "A" in report["missing"]
    assert "B" in report["missing"]


def test_suggest_returns_items():
    ds = Dataset(sample_df())
    suggestions = ds.suggest(as_dict=True)
    assert len(suggestions) > 0
    assert "severity_score" in suggestions[0]


def test_clean_executes_explicitly():
    ds = Dataset(sample_df())
    ds.clean(num_missing="mean", cat_missing="mode")
    assert ds.df.isnull().sum().sum() == 0
