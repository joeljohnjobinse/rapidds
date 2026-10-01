# Changelog
## 0.2.2

- Added transformation provenance/history tracking to `Dataset`.
- Added `Dataset.export()` for CSV, Excel, JSON, Parquet, and HTML output.
- Added provenance-aware Excel reports with `Cleaned_Data`, `Changes`, `Quality`, `Suggestions`, `Analysis`, and `Metadata` sheets.
- Added `Dataset.report()` for human-readable HTML/Excel reports.
- Added JSON-safe report serialization and informative optional Parquet dependency errors.
- Existing cleaning/transformation APIs remain backward compatible.
- Added evidence-scored suggestion confidence and priority ranking.
- Added dataset context and semantic column-role inference.
- Added conservative target/task inference for classification and regression workflows.
- Added compound reasoning for skew + missingness, outliers + skew, high-cardinality features, target imbalance, and possible leakage.
- Added `SuggestionPlan` with `summary()`, `show()`, approval helpers, and `auto_clean(plan=...)` integration.
- Added workflow stages and explicit recommendation dependencies.
- Added tests for semantic inference, task inference, suggestion plans, and plan-driven cleaning.


## 0.2.0

- Added structured dataset profiling.
- Added data-quality checks for missing values, duplicates, constants, numeric strings, formatting, and outliers.
- Expanded `suggest()` with structured actions, confidence, and metadata.
- Added conservative `auto_clean()` with dry-run support and an audit report.
- Added explicit cleaning helpers for missing values, duplicate rows, columns, numeric conversion, and label standardization.
- Added scaling, categorical encoding, datetime feature generation, and log transforms.
- Added descriptive statistics, correlation, covariance, bootstrap confidence intervals, t-test, chi-square, and ANOVA helpers.
- Added IQR, z-score, and modified z-score outlier detection.
- Added classification, regression, and clustering evaluation utilities.
- Expanded tests and documentation.
