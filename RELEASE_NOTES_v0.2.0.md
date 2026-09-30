# rapidds v0.2.0

## What changed

rapidds now goes beyond basic dataset inspection with a structured analysis and data-preparation layer.

### Smarter suggestions
Suggestions now carry:
- severity
- confidence
- category
- executable action
- column metadata where relevant

### Auto-clean
`Dataset.auto_clean()` applies conservative, high-confidence suggestions when explicitly called. It supports `dry_run=True` so users can preview the resulting DataFrame and action plan before changing data.

### Analysis
- dataset profiling
- data-quality checks
- outlier detection
- descriptive statistics
- correlation/covariance
- bootstrap intervals
- common statistical tests

### Transformation
- missing-value filling
- text standardization
- numeric conversion
- scaling
- categorical encoding
- datetime features
- log transforms

### Modeling
- train/test splitting
- classification evaluation
- regression evaluation
- clustering evaluation

## Design direction

The central rapidds workflow remains:

**Detect → Suggest → Execute**

Automatic behavior is deliberately conservative and auditable.
