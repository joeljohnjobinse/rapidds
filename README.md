# rapidds

**rapidds** is a lightweight data assessment library designed to help analysts and data scientists
understand, evaluate, and prepare datasets **without introducing hidden assumptions**.

It focuses on surfacing data risks early, providing transparent recommendations,
and keeping the analyst fully in control.

---

## Why rapidds exists

Many data science workflows fail early due to:

- Silent data quality issues
- Implicit or automatic cleaning decisions
- Poor visibility into data risks before modeling

rapidds addresses these problems by **separating judgment from execution**.

---

## Core philosophy

rapidds follows a strict three-step approach:

### 1. Detect
Identify measurable facts about the dataset  
(e.g. missing values, imbalance, low-variance columns).

### 2. Suggest
Provide **reasoned, explainable recommendations**  
without modifying the data.

### 3. Execute
Apply changes **only when explicitly requested** by the analyst.

This mirrors how experienced analysts work in practice.

---

## What rapidds is (and is not)

### rapidds **is**
- A data quality assessment tool
- A thinking companion for analysts
- A safe pre-EDA and pre-modeling step
- Fully composable with pandas and sklearn

### rapidds **is not**
- An AutoML system
- A replacement for pandas or sklearn
- A black-box decision maker
- A tool that silently mutates data

---

## Installation

For now, rapidds is intended for local development or internal use.

```bash
pip install -e .
```

(Public distribution may follow once the API stabilizes.)

---

## Quick example

```python
from rapidds import Dataset

ds = Dataset(df)

# Inspect the dataset
ds.explain()

# Apply cleaning explicitly
df_clean = ds.clean(
    num_missing="median",
    cat_missing="mode"
)
```

rapidds will **never** clean or modify data unless you ask it to.

---

## Example output

```
rapidds — dataset assessment
----------------------------

HIGH PRIORITY:
 • 'Cabin' has 77% missing values, which may limit its usefulness.

MEDIUM PRIORITY:
 • 'Age' has substantial missing values; targeted imputation may be appropriate.
 • 'Survived' shows class imbalance; stratified sampling is recommended.

LOW PRIORITY:
 • 'Embarked' contains missing values; mode imputation is commonly used.

Review suggestions before applying any cleaning steps.
```

---

## Design principles

- Explicit over implicit
- No silent data mutation
- Conservative defaults
- Transparent recommendations
- Analyst remains in control

These principles are enforced by design, not convention.

---

## Current scope

rapidds currently supports:

- Dataset inspection
- Missing value detection
- Prioritized recommendations
- Human-readable explanations
- Explicit cleaning execution
- Markdown-ready reporting

Future features will prioritize **clarity and trust over automation**.

---

## License

MIT License
