# rapidds v0.1.0 — Initial Stable Release

This is the first stable release of **rapidds**, a lightweight data inspection and preparation library designed to help analysts **understand, assess, and explicitly prepare datasets for modeling**.

rapidds focuses on *reasoning and guardrails*, not automation or black-box decisions.

---

## What rapidds is

rapidds helps you:

- Inspect datasets safely
- Surface data quality and modeling risks
- Receive reasoned, transparent suggestions
- Apply **minimal, explicit preparation steps**
- Hand off cleanly to pandas / sklearn workflows

rapidds does **not** train models, encode features automatically, or make irreversible decisions.

---

## Installation (v0.1.0)

Recommended (editable install for development or internal use):

```bash
git clone https://github.com/<your-username>/rapidds
cd rapidds
pip install -e .
```

Alternative (direct GitHub install):

```bash
pip install git+https://github.com/<your-username>/rapidds.git
```

---

## Canonical usage example

```python
import pandas as pd
from rapidds import Dataset
from rapidds.visualize import plot_missing

df = pd.read_csv("data.csv")

ds = Dataset(df)

report = ds.inspect()
report.show()

plot_missing(ds)

df_clean = ds.clean(
    num_missing="median",
    cat_missing="mode"
)

X, y, warnings = ds.prepare_for_modeling(target="target_column")
```

---

## Public API (Stable in v0.1.0)

The following APIs are considered **stable** in this release.

### Dataset

#### Dataset(df: pandas.DataFrame)

Wraps a pandas DataFrame for inspection and preparation.

---

### Inspection & reasoning

#### inspect() → InspectionReport

Runs a full dataset inspection and returns a structured report.

Includes:
- dataset shape
- overall risk assessment
- prioritized suggestions
- consistency issues

---

#### analyze() → dict

Returns raw, structured facts about the dataset:
- shape
- column types
- missing value ratios

No suggestions or mutations.

---

#### suggest(as_dict: bool = False)

Returns prioritized suggestions about data quality and modeling risks.

Parameters:
- `as_dict` (bool): return structured suggestion objects if True

Each suggestion includes:
- priority level (high / medium / low)
- severity score
- category
- message
- confidence note

---

#### risk_summary() → dict

Returns a high-level risk assessment suitable for reviews or CI.

Includes:
- overall risk level
- summary text
- issue count
- top concerns

---

#### check_consistency() → list[str]

Detects common consistency issues such as:
- mixed data types
- numeric values stored as strings
- inconsistent casing
- inconsistent whitespace

Read-only; does not modify data.

---

### Explicit preparation

#### clean(
    num_missing: {"mean","median",None} = None,
    cat_missing: {"mode",None} = None,
    drop_duplicates: bool = True
) → pandas.DataFrame

Applies **explicit, minimal** cleaning steps.

---

#### standardize_labels(
    columns: list[str],
    case: {"lower","upper",None} = "lower",
    strip_whitespace: bool = True
)

Explicitly standardizes categorical labels.

This method mutates the dataset **only when called explicitly**.

---

#### prepare_for_modeling(target: str) → (X, y, warnings)

Prepares dataset for downstream modeling by:

- validating target column
- preventing missing targets
- surfacing encoding or imputation requirements

Does **not** perform transformations.

---

#### split(
    target: str,
    test_size: float = 0.2,
    stratify: bool = False,
    random_state: int = 42
) → (X_train, X_test, y_train, y_test, warnings)

Thin wrapper around `sklearn.model_selection.train_test_split` with safety checks.

---

### Reporting

#### InspectionReport

Returned by `inspect()`.

Methods:
- `show()` — human-readable console output
- `to_markdown()` — report-ready Markdown export

---

### Visualization (diagnostic only)

All visualizations live in `rapidds.visualize` and are **pure functions**.

#### plot_missing(data)

Visualizes missing-value percentages per column.

---

#### plot_target(data, target: str)

Visualizes target distribution for imbalance inspection.

---

#### plot_distributions(data, max_cols: int = 6)

Plots numeric feature distributions (histograms).

---

## Design guarantees

rapidds guarantees:

- No silent data mutation
- No automatic modeling
- No irreversible transformations
- Conservative, transparent defaults
- Composability with pandas and sklearn

---

## Versioning

rapidds follows semantic versioning:

- MAJOR — breaking changes
- MINOR — new features or behavior changes
- PATCH — bug fixes only

---

## Release philosophy

> rapidds accelerates judgment — it does not replace it.

This release establishes a stable, trustworthy foundation.
