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


def test_profile_contains_column_statistics():
    ds = Dataset(sample_df())
    profile = ds.profile()
    assert profile["shape"]["rows"] == 4
    assert "A" in profile["columns"]
    assert profile["columns"]["A"]["missing"] == 1


def test_auto_clean_applies_safe_suggestions():
    df = pd.DataFrame({
        "age": [20, None, 30],
        "city": [" Chennai", "Chennai ", "Chennai"],
        "amount": ["10", "20", "30"],
    })
    ds = Dataset(df)
    result = ds.auto_clean()
    assert ds.df["age"].isna().sum() == 0
    assert ds.df["city"].tolist() == ["Chennai", "Chennai", "Chennai"]
    assert pd.api.types.is_numeric_dtype(ds.df["amount"])
    assert ds.last_cleaning_report["action_count"] >= 3


def test_auto_clean_dry_run_does_not_mutate():
    ds = Dataset(sample_df())
    original = ds.df.copy()
    cleaned, report = ds.auto_clean(dry_run=True)
    assert ds.df.equals(original)
    assert report["dry_run"] is True
    assert cleaned.isna().sum().sum() == 0


def test_smart_suggest_detects_contextual_issues():
    df = pd.DataFrame({
        "customer_id": list(range(1, 21)),
        "age": [20, 21, None, 23, 24] * 4,
        "income": [100, 110, 120, 130, 1000] * 4,
        "joined_at": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"] * 4,
        "target": [0] * 18 + [1, 1],
    })
    ds = Dataset(df)
    suggestions = ds.suggest(as_dict=True, target="target")
    actions = {s["action"] for s in suggestions}
    assert "inspect_identifier" in actions
    assert "impute" in actions
    assert "inspect_outliers" in actions
    assert "add_datetime_features" in actions
    assert "stratify_target" in actions
    assert all("evidence" in s and "risk" in s and "confidence" in s for s in suggestions)


def test_smart_suggest_flags_possible_leakage_and_correlation():
    df = pd.DataFrame({
        "feature_a": [0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
        "feature_b": [0.0, 0.1, 0.2, 1.0, 1.1, 0.9, 0.05, 1.05, 0.02, 1.02],
        "target": [0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
    })
    ds = Dataset(df)
    suggestions = ds.suggest(as_dict=True, target="target")
    categories = {s["category"] for s in suggestions}
    assert "modeling" in categories
    assert any(s["action"] == "inspect_leakage" for s in suggestions)
    assert any(s["action"] == "inspect_correlation" for s in suggestions)


def test_cleaning_policies_change_behavior():
    df = pd.DataFrame({
        "empty": [None, None, None, None],
        "label": ["A", "a", " A ", "a"],
        "value": [1, None, 3, 4],
        "date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
    })
    conservative = Dataset(df)
    conservative.auto_clean(policy="conservative")
    assert "empty" in conservative.df.columns
    assert conservative.df["value"].isna().sum() == 0

    balanced = Dataset(df)
    balanced.auto_clean(policy="balanced")
    assert "empty" not in balanced.df.columns

    aggressive = Dataset(df)
    aggressive.auto_clean(policy="aggressive")
    assert "date_year" in aggressive.df.columns


def test_auto_clean_report_records_skipped_risky_actions():
    df = pd.DataFrame({"id": range(20), "value": [1] * 19 + [1000]})
    ds = Dataset(df)
    ds.auto_clean(policy="conservative", target=None)
    assert "skipped" in ds.last_cleaning_report
    assert ds.last_cleaning_report["policy"] == "conservative"
