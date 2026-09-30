# rapidds – Design Manifesto

rapidds exists to accelerate data understanding, not replace analyst judgment.

## Core Principles

### 1. Explicit over implicit
rapidds never performs an automatic transformation merely because it detected an issue. Calling `auto_clean()` is itself explicit user consent to execute its conservative cleaning policy.

### 2. Detect → Suggest → Execute
- **Detect** facts about the data.
- **Suggest** possible actions with severity, confidence, and an executable action.
- **Execute** only when the user requests an action.

### 3. Conservative defaults
`auto_clean()` applies only high-confidence, low-risk operations by default: exact duplicate removal, straightforward missing-value imputation, safe numeric-string conversion, and whitespace trimming. Dropping columns, changing semantic labels, and removing outliers require explicit opt-in.

### 4. Transparent automation
Every automatic change produces an audit record containing the action, affected column, strategy, and counts where applicable. `dry_run=True` previews changes without mutating the dataset.

### 5. No silent data leakage
rapidds avoids automatic modeling or feature engineering that could introduce leakage.

### 6. Composable by design
rapidds integrates with pandas, sklearn, and scientific Python workflows instead of replacing them.

## What rapidds is NOT
- An AutoML tool
- A replacement for pandas or sklearn
- A black-box decision system

## Public workflow

```python
from rapidds import Dataset

ds = Dataset("data.csv")
ds.inspect()       # understand the dataset
ds.suggest()       # receive structured recommendations
preview, plan = ds.auto_clean(dry_run=True)
ds.auto_clean()    # explicitly apply conservative fixes
```
