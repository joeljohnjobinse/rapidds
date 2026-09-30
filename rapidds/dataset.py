from __future__ import annotations

import copy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from rapidds.analysis import (
    profile_dataframe, quality_checks, correlation, covariance, bootstrap,
    t_test, chi_square, anova, detect_outliers,
)
from rapidds.cleaning import fill_missing, standardize_text, CleaningPolicy, action_allowed
from rapidds.transform import scale, encode_categorical, datetime_features
from rapidds.modeling import evaluate_classification, evaluate_regression, evaluate_clustering
from rapidds.report import InspectionReport

_SEVERITY_MAP = {"high": 3, "medium": 2, "low": 1}


class Dataset:
    """Guided data-science companion around a pandas DataFrame.

    rapidds follows Detect -> Suggest -> Execute. ``auto_clean`` is explicit
    user consent to execute only conservative, high-confidence suggestions.
    Every automatic change is recorded in ``last_cleaning_report``.
    """

    def __init__(self, data):
        if isinstance(data, (str, bytes)):
            try:
                self.df = pd.read_csv(data)
            except Exception as e:
                raise ValueError(f"Could not read file: {e}") from e
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()
        else:
            raise ValueError("Dataset expects a pandas DataFrame or a CSV file path.")
        self.last_cleaning_report = None

    def __repr__(self):
        rows, cols = self.df.shape
        max_missing = self.df.isnull().mean().max() * 100 if rows > 0 else 0
        return f"<rapidds.Dataset | rows={rows}, cols={cols}, max_missing={max_missing:.1f}%>"

    # ------------------------------------------------------------------
    # DETECT / PROFILE
    # ------------------------------------------------------------------
    def analyze(self) -> dict:
        p = profile_dataframe(self.df)
        return {
            "shape": p["shape"],
            "column_types": {"numeric": p["numeric_columns"], "categorical": p["categorical_columns"]},
            "missing": {c: v["missing_ratio"] for c, v in p["columns"].items() if v["missing"]},
            "duplicates": p["duplicate_rows"],
            "memory_usage_bytes": p["memory_usage_bytes"],
        }

    def profile(self) -> dict:
        return profile_dataframe(self.df)

    def quality(self) -> dict:
        return quality_checks(self.df)

    # ------------------------------------------------------------------
    # SUGGEST
    # ------------------------------------------------------------------
    def suggest(self, as_dict: bool = False, target=None, include_low_confidence=True, profile="general", aggressiveness="balanced"):
        """Generate context-aware, actionable recommendations.

        Suggestions are derived from multiple signals rather than isolated rules.
        Each item includes evidence, risk, confidence, and an executable action.
        ``target`` can be supplied when the dataset has a known modeling target;
        otherwise rapidds only makes conservative target-related inferences.
        """
        df = self.df
        suggestions = []
        profile = str(profile).lower()
        if profile not in {"general", "ml"}:
            raise ValueError("profile must be 'general' or 'ml'")
        if aggressiveness not in {"conservative", "balanced", "aggressive"}:
            raise ValueError("aggressiveness must be conservative, balanced, or aggressive")
        rows = len(df)
        cols = list(df.columns)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        object_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        policy = CleaningPolicy.from_name(aggressiveness)

        if target is not None and target not in df.columns:
            raise ValueError(f"Target column '{target}' not found")

        def add(level, category, message, action=None, confidence="medium", risk="medium", evidence=None, **extra):
            item = {
                "level": level,
                "severity_score": _SEVERITY_MAP[level],
                "category": category,
                "message": message,
                "action": action,
                "confidence": confidence,
                "risk": risk,
                "evidence": evidence or {},
                **extra,
            }
            if include_low_confidence or confidence != "low":
                suggestions.append(item)

        # Dataset-level context -------------------------------------------------
        duplicate_count = int(df.duplicated().sum())
        if duplicate_count:
            ratio = duplicate_count / max(rows, 1)
            add(
                "medium", "quality",
                f"{duplicate_count} exact duplicate rows detected ({ratio:.1%} of rows).",
                "drop_duplicates", "high", "low",
                {"duplicate_count": duplicate_count, "duplicate_ratio": ratio},
                count=duplicate_count, ratio=ratio,
            )

        # Column-level reasoning ------------------------------------------------
        for col in cols:
            s = df[col]
            missing_ratio = float(s.isna().mean()) if rows else 0.0
            unique = int(s.nunique(dropna=True))
            unique_ratio = unique / max(rows, 1)
            is_numeric = pd.api.types.is_numeric_dtype(s)
            is_target = col == target

            # Missingness: distinguish target, sparse and heavily missing features.
            if missing_ratio > 0:
                if is_target:
                    add(
                        "high", "target", f"Target '{col}' contains {missing_ratio:.1%} missing values; dropping target rows may change the modeling population.",
                        "inspect_target_missing", "high", "high",
                        {"missing_ratio": missing_ratio}, column=col, ratio=missing_ratio,
                    )
                elif missing_ratio > 0.70:
                    add(
                        "high", "missing", f"'{col}' is {missing_ratio:.1%} missing. Dropping it may be preferable to aggressive imputation, but inspect its meaning first.",
                        "drop_column", "medium", "high",
                        {"missing_ratio": missing_ratio, "threshold": 0.70}, column=col, ratio=missing_ratio,
                    )
                else:
                    strategy = "median" if is_numeric else "mode"
                    # Median is safer for skewed numeric data than mean.
                    if is_numeric and s.notna().sum() >= 4 and abs(float(s.skew())) > 1:
                        strategy = "median"
                    level = "medium" if missing_ratio > 0.30 else "low"
                    add(
                        level, "missing",
                        f"'{col}' has {missing_ratio:.1%} missing values; {strategy} imputation is a conservative option.",
                        "impute", "high", "low",
                        {"missing_ratio": missing_ratio, "strategy": strategy, "dtype": str(s.dtype)},
                        column=col, strategy=strategy, ratio=missing_ratio,
                    )

            # Constant / near-constant columns.
            if unique <= 1 and not is_target and missing_ratio < 1.0:
                add(
                    "medium", "quality", f"'{col}' has no variation and cannot provide useful predictive information.",
                    "drop_constant", "high", "low", {"unique": unique}, column=col,
                )
            elif rows >= 20 and unique_ratio < 0.01 and not is_target:
                add(
                    "low", "quality", f"'{col}' has very low cardinality ({unique} unique values); inspect whether it is a flag or low-information feature.",
                    "inspect_cardinality", "medium", "medium", {"unique": unique, "unique_ratio": unique_ratio},
                    column=col, unique=unique,
                )

            if rows and missing_ratio == 1.0 and not is_target:
                add(
                    "high", "quality", f"'{col}' is completely empty and contains no usable observations.",
                    "drop_column", "high", "low", {"missing_ratio": 1.0}, column=col, ratio=1.0,
                )

            # Identifier-like columns: use both uniqueness and naming hints.
            name = str(col).lower()
            id_name = any(token in name for token in ("id", "uuid", "identifier", "key"))
            if rows >= 20 and unique_ratio >= 0.98 and not is_target:
                confidence = "high" if id_name else "medium"
                add(
                    "medium" if id_name else "low", "modeling",
                    f"'{col}' is almost entirely unique and may be an identifier rather than a predictive feature.",
                    "inspect_identifier", confidence, "medium",
                    {"unique_ratio": unique_ratio, "name_hint": id_name},
                    column=col, unique_ratio=unique_ratio,
                )

            # Numeric stored as text, with safeguards against mixed semantic text.
            if col in object_cols:
                values = s.dropna().astype(str)
                if not values.empty:
                    numeric_like = float(pd.to_numeric(values, errors="coerce").notna().mean())
                    if numeric_like == 1.0:
                        add(
                            "medium", "types", f"'{col}' contains only numeric-looking values but is stored as text; it can be safely converted.",
                            "convert_numeric", "high", "low",
                            {"numeric_like_ratio": numeric_like}, column=col, ratio=numeric_like,
                        )
                    elif numeric_like >= 0.8:
                        add(
                            "medium", "types", f"'{col}' is mostly numeric-looking but contains non-numeric values; inspect before converting.",
                            "inspect_numeric_conversion", "medium", "high",
                            {"numeric_like_ratio": numeric_like}, column=col, ratio=numeric_like,
                        )

                    stripped = values.str.strip()
                    if int((stripped != values).sum()) > 0:
                        add(
                            "low", "formatting", f"'{col}' contains leading or trailing whitespace; trimming is low-risk.",
                            "strip_whitespace", "high", "low",
                            {"affected_values": int((stripped != values).sum())}, column=col,
                        )

                    lower_unique = stripped.str.lower().nunique()
                    if lower_unique < stripped.nunique():
                        add(
                            "low", "formatting", f"'{col}' contains values that differ only by letter casing; inspect before standardizing labels.",
                            "standardize_case", "medium", "medium",
                            {"unique_before_casefold": int(stripped.nunique()), "unique_after_casefold": int(lower_unique)}, column=col,
                        )

                    # Low-cardinality text is useful for categorical modeling; very high
                    # cardinality can behave like an identifier.
                    if unique > 20 and unique_ratio > 0.5 and not is_target:
                        add(
                            "low", "modeling", f"'{col}' has high categorical cardinality ({unique} unique values); one-hot encoding may create a large feature space.",
                            "inspect_cardinality", "high", "medium",
                            {"unique": unique, "unique_ratio": unique_ratio}, column=col, unique=unique,
                        )

            # Outliers are suggestions, never automatic deletions.
            if is_numeric and not is_target:
                non_null = s.dropna()
                if len(non_null) >= 8:
                    result = detect_outliers(s, method="iqr")
                    outlier_ratio = result["count"] / max(len(non_null), 1)
                    if outlier_ratio >= 0.01:
                        skew = float(s.skew()) if len(non_null) > 2 else 0.0
                        add(
                            "medium" if outlier_ratio >= 0.05 else "low", "outliers",
                            f"'{col}' has {result['count']} IQR outliers ({outlier_ratio:.1%}); investigate whether they are errors or legitimate extremes.",
                            "inspect_outliers", "high", "high",
                            {"count": result["count"], "ratio": outlier_ratio, "skewness": skew},
                            column=col, count=result["count"], ratio=outlier_ratio,
                        )

            # Datetime detection.
            if col in object_cols and not is_target:
                parsed = pd.to_datetime(s, errors="coerce", format="mixed")
                parse_ratio = float(parsed.notna().mean()) if rows else 0.0
                if parse_ratio >= 0.90 and unique > 1:
                    add(
                        "low", "datetime", f"'{col}' appears to contain datetime values; extracting temporal features may be useful.",
                        "add_datetime_features", "high", "low",
                        {"parse_ratio": parse_ratio}, column=col, parse_ratio=parse_ratio,
                    )

        # ML-specific reasoning -------------------------------------------------
        if profile == "ml" and target in df.columns:
            feature_cols = [c for c in cols if c != target]
            if feature_cols:
                non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
                if non_numeric:
                    add(
                        "low", "modeling", f"{len(non_numeric)} feature(s) are non-numeric and will require encoding for many ML estimators.",
                        "encode_categorical", "high", "low", {"columns": non_numeric}, columns=non_numeric,
                    )
                if numeric_cols:
                    ranges = {c: float(df[c].max() - df[c].min()) for c in numeric_cols if df[c].notna().any()}
                    if len(ranges) >= 2 and max(ranges.values()) > 0:
                        spread = max(ranges.values()) / max(min(v for v in ranges.values() if v > 0), 1e-12)
                        if spread >= 1000:
                            add(
                                "low", "modeling", "Numeric features have very different scales; scaling may help distance- or gradient-based models.",
                                "scale_features", "high", "low", {"range_ratio": float(spread)}, columns=list(ranges),
                            )

        # Target reasoning ------------------------------------------------------
        inferred_target = target
        if inferred_target is None and cols:
            candidate = cols[-1]
            candidate_values = df[candidate].dropna().nunique()
            # Only infer a target for a clearly small-cardinality final column.
            if 2 <= candidate_values <= 10:
                inferred_target = candidate

        if inferred_target in df.columns:
            y = df[inferred_target]
            if y.isna().any():
                add("high", "target", f"'{inferred_target}' has missing target values; define how those rows should be handled before training.",
                    "inspect_target_missing", "high", "high", {"missing": int(y.isna().sum())}, column=inferred_target)
            counts = y.value_counts(normalize=True, dropna=True)
            if len(counts) >= 2 and float(counts.iloc[0]) >= 0.80:
                add(
                    "medium", "modeling",
                    f"Target '{inferred_target}' is imbalanced: the largest class represents {counts.iloc[0]:.1%} of non-missing targets.",
                    "stratify_target", "high", "medium",
                    {"majority_ratio": float(counts.iloc[0]), "classes": int(len(counts))},
                    column=inferred_target, majority_ratio=float(counts.iloc[0]),
                )

            # Target leakage heuristic: exact/near-exact feature match is a strong warning.
            for col in cols:
                if col == inferred_target or col in object_cols and col == inferred_target:
                    continue
                if len(df[col].dropna()) == 0:
                    continue
                aligned = df[[col, inferred_target]].dropna()
                if len(aligned) < max(10, int(rows * 0.5)):
                    continue
                if aligned[col].nunique() <= 20 and aligned[inferred_target].nunique() <= 20:
                    table = pd.crosstab(aligned[col], aligned[inferred_target], normalize="index")
                    if not table.empty and float(table.max(axis=1).mean()) >= 0.98:
                        add(
                            "high", "modeling",
                            f"'{col}' almost deterministically maps to target '{inferred_target}'. Inspect it for possible target leakage.",
                            "inspect_leakage", "medium", "high",
                            {"mean_dominance": float(table.max(axis=1).mean())},
                            column=col, target=inferred_target,
                        )

        # Numeric multicollinearity signal. We do not recommend dropping a feature automatically.
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr().abs()
            pairs = []
            for i, left in enumerate(numeric_cols):
                for right in numeric_cols[i + 1:]:
                    value = corr.loc[left, right]
                    if pd.notna(value) and value >= 0.95:
                        pairs.append((left, right, float(value)))
            if pairs:
                add(
                    "medium", "statistics",
                    f"{len(pairs)} numeric feature pair(s) have absolute correlation ≥ 0.95; inspect for redundancy or multicollinearity.",
                    "inspect_correlation", "high", "medium",
                    {"pairs": pairs[:20], "threshold": 0.95}, pairs=pairs[:20],
                )

        # Rank by severity, then confidence, then risk. Stable ordering preserves column order.
        confidence_score = {"high": 3, "medium": 2, "low": 1}
        risk_score = {"low": 1, "medium": 2, "high": 3}
        suggestions.sort(key=lambda x: (-x["severity_score"], -confidence_score[x["confidence"]], risk_score[x["risk"]]))
        return suggestions if as_dict else [s["message"] for s in suggestions]

    def explain(self) -> None:
        suggestions = self.suggest(as_dict=True)
        if not suggestions:
            print("No major issues detected. Dataset looks clean 👍")
            return
        print("\nrapidds analysis summary\n" + "-" * 30)
        for level in ["high", "medium", "low"]:
            items = [s for s in suggestions if s["level"] == level]
            if items:
                print(f"\n{level.upper()} PRIORITY:")
                for item in items:
                    print(f" • {item['message']}")
        print("\n(Use ds.clean(...) for explicit actions or ds.auto_clean() for conservative automatic fixes.)")

    # ------------------------------------------------------------------
    # AUTO-CLEAN
    # ------------------------------------------------------------------
    def auto_clean(self, dry_run: bool = False, policy="conservative", target=None,
                   profile="general", drop_high_missing=None, standardize_case=None,
                   convert_numeric=True, drop_duplicates=True, inplace=True):
        """Apply suggestions according to a named safety policy.

        Policies:
        - conservative: only high-confidence, low-risk, reversible-ish fixes.
        - balanced: additionally allows high-confidence column drops and label casing.
        - aggressive: permits medium-confidence, higher-risk cleanup and datetime feature extraction.

        ``dry_run=True`` returns the proposed DataFrame and an audit report without mutating ``self.df``.
        Explicit legacy flags remain supported and can only loosen the policy where requested.
        """
        policy_obj = CleaningPolicy.from_name(policy)
        suggestions = self.suggest(as_dict=True, target=target, profile=profile, aggressiveness=policy)
        work = self.df.copy()
        actions, skipped = [], []

        # Backwards-compatible overrides: explicit True opts in, False opts out.
        if drop_high_missing is True:
            policy_obj = CleaningPolicy(policy_obj.name, policy_obj.max_risk, policy_obj.min_confidence, True,
                                        policy_obj.allow_standardize_case, policy_obj.allow_remove_outliers,
                                        policy_obj.allow_datetime_features)
        if standardize_case is True:
            policy_obj = CleaningPolicy(policy_obj.name, policy_obj.max_risk, policy_obj.min_confidence,
                                        policy_obj.allow_drop_columns, True, policy_obj.allow_remove_outliers,
                                        policy_obj.allow_datetime_features)

        seen = set()
        def record(action, **kwargs):
            entry = {"action": action, **kwargs}
            actions.append(entry)

        for suggestion in suggestions:
            action = suggestion.get("action")
            col = suggestion.get("column")
            key = (action, col)
            if key in seen:
                continue
            seen.add(key)
            if not action_allowed(suggestion, policy_obj):
                skipped.append({"action": action, "column": col, "reason": "policy", "risk": suggestion.get("risk"), "confidence": suggestion.get("confidence")})
                continue
            if action == "drop_duplicates" and drop_duplicates:
                before = len(work); work = work.drop_duplicates().reset_index(drop=True)
                if before != len(work): record(action, rows_removed=before-len(work))
            elif action == "impute" and col in work.columns:
                missing = int(work[col].isna().sum())
                if missing:
                    strategy = suggestion.get("strategy", "median")
                    value = work[col].median() if strategy == "median" else work[col].mean() if strategy == "mean" else (work[col].mode(dropna=True).iloc[0] if not work[col].mode(dropna=True).empty else None)
                    if value is not None and pd.notna(value):
                        work[col] = work[col].fillna(value); record(action, column=col, strategy=strategy, filled=missing, value=value)
            elif action == "strip_whitespace" and col in work.columns:
                before = work[col].copy(); work[col] = work[col].astype("string").str.strip()
                changed = int((before.fillna("<NA>") != work[col].fillna("<NA>")).sum())
                if changed: record(action, column=col, values_changed=changed)
            elif action == "convert_numeric" and convert_numeric and col in work.columns:
                converted = pd.to_numeric(work[col], errors="coerce")
                if converted.notna().sum() == work[col].notna().sum():
                    work[col] = converted; record(action, column=col)
            elif action == "drop_constant" and col in work.columns:
                work = work.drop(columns=[col]); record(action, column=col, reason="constant")
            elif action == "drop_column" and col in work.columns and policy_obj.allow_drop_columns:
                work = work.drop(columns=[col]); record(action, column=col, reason="high_missing_or_empty")
            elif action == "standardize_case" and col in work.columns and standardize_case is not False:
                work[col] = work[col].astype("string").str.strip().str.lower(); record(action, column=col, case="lower")
            elif action == "add_datetime_features" and col in work.columns and policy_obj.allow_datetime_features:
                parsed = pd.to_datetime(work[col], errors="coerce", format="mixed")
                work[f"{col}_year"] = parsed.dt.year
                work[f"{col}_month"] = parsed.dt.month
                work[f"{col}_dayofweek"] = parsed.dt.dayofweek
                work[f"{col}_is_weekend"] = parsed.dt.dayofweek.isin([5, 6])
                record(action, column=col, features=[f"{col}_year", f"{col}_month", f"{col}_dayofweek", f"{col}_is_weekend"])

        report = {
            "dry_run": dry_run,
            "policy": policy_obj.name,
            "profile": profile,
            "target": target,
            "actions": actions,
            "skipped": skipped,
            "action_count": len(actions),
            "skipped_count": len(skipped),
            "suggestions_considered": len(suggestions),
            "rows_before": len(self.df), "rows_after": len(work),
            "columns_before": list(self.df.columns), "columns_after": list(work.columns),
        }
        if not dry_run:
            self.last_cleaning_report = report
            if inplace:
                self.df = work
                return self.df
        return work, report

    # ------------------------------------------------------------------
    # EXPLICIT CLEANING
    # ------------------------------------------------------------------
    def clean(self, num_missing=None, cat_missing=None, drop_duplicates=True):
        df = self.df.copy()
        if drop_duplicates:
            df = df.drop_duplicates()
        if num_missing in {"mean", "median"}:
            cols = df.select_dtypes(include=np.number).columns
            df[cols] = df[cols].fillna(getattr(df[cols], num_missing)())
        if cat_missing == "mode":
            for col in df.select_dtypes(exclude=np.number):
                mode = df[col].mode(dropna=True)
                if not mode.empty: df[col] = df[col].fillna(mode.iloc[0])
        self.df = df
        return df

    def fill_missing(self, numeric="median", categorical="mode"):
        self.df, actions = fill_missing(self.df, numeric, categorical)
        return actions

    def standardize_labels(self, columns, case="lower", strip_whitespace=True):
        self.df, _ = standardize_text(self.df, columns, case, strip_whitespace)
        return self.df

    def convert_numeric(self, columns=None):
        cols = columns or [s["column"] for s in self.suggest(as_dict=True) if s.get("action") == "convert_numeric"]
        for col in cols:
            if col not in self.df.columns: raise ValueError(f"Column '{col}' not found")
            self.df[col] = pd.to_numeric(self.df[col], errors="raise")
        return self.df

    def drop_columns(self, columns):
        self.df = self.df.drop(columns=columns)
        return self.df

    def drop_duplicates(self):
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        return self.df

    # ------------------------------------------------------------------
    # TRANSFORMATION / FEATURES
    # ------------------------------------------------------------------
    def scale(self, columns=None, method="standard"):
        self.df = scale(self.df, columns, method); return self.df

    def encode(self, columns=None, drop_first=False):
        self.df = encode_categorical(self.df, columns, drop_first); return self.df

    def add_datetime_features(self, columns):
        self.df = datetime_features(self.df, columns); return self.df

    def log_transform(self, columns):
        for col in columns:
            if (self.df[col].dropna() < 0).any(): raise ValueError(f"Column '{col}' contains negative values")
            self.df[col] = np.log1p(self.df[col])
        return self.df

    # ------------------------------------------------------------------
    # STATISTICS / EDA
    # ------------------------------------------------------------------
    def describe(self, columns=None):
        from rapidds.analysis import describe
        return describe(self.df, columns)

    def correlation(self, method="pearson"):
        return correlation(self.df, method)

    def covariance(self):
        return covariance(self.df)

    def outliers(self, column, method="iqr", threshold=3.0):
        return detect_outliers(self.df[column], method, threshold)

    # ------------------------------------------------------------------
    # MODELING
    # ------------------------------------------------------------------
    def prepare_for_modeling(self, target):
        if target not in self.df.columns: raise ValueError(f"Target '{target}' not found")
        if self.df[target].isnull().any(): raise ValueError("Target column contains missing values")
        X, y = self.df.drop(columns=[target]), self.df[target]
        warnings = []
        if X.select_dtypes(exclude="number").shape[1] > 0: warnings.append("Categorical features detected — encoding will be required.")
        if X.select_dtypes(include="number").isnull().any().any(): warnings.append("Numeric features contain missing values — imputation may be required.")
        return X, y, warnings

    def split(self, target, test_size=0.2, stratify=False, random_state=42):
        X, y, warnings = self.prepare_for_modeling(target)
        strat = y if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=strat, random_state=random_state)
        return X_train, X_test, y_train, y_test, warnings

    def evaluate_classification(self, y_true, y_pred, y_score=None): return evaluate_classification(y_true, y_pred, y_score)
    def evaluate_regression(self, y_true, y_pred): return evaluate_regression(y_true, y_pred)
    def evaluate_clustering(self, X, labels): return evaluate_clustering(X, labels)

    # ------------------------------------------------------------------
    # LEGACY / REPORTING
    # ------------------------------------------------------------------
    def check_consistency(self):
        issues = []
        for col in self.df.columns:
            types = self.df[col].dropna().map(type).unique()
            if len(types) > 1: issues.append(f"Column '{col}' contains mixed data types.")
        for col in self.df.select_dtypes(include="object"):
            values = self.df[col].dropna().astype(str)
            if values.empty: continue
            numeric_like = pd.to_numeric(values, errors="coerce").notna().mean()
            if numeric_like >= .8: issues.append(f"Column '{col}' appears numeric but is stored as strings.")
            if values.str.strip().nunique() < values.nunique(): issues.append(f"Column '{col}' contains inconsistent whitespace.")
            if values.str.lower().nunique() < values.nunique(): issues.append(f"Column '{col}' contains inconsistent casing.")
        return issues

    def risk_summary(self):
        suggestions = self.suggest(as_dict=True); consistency = self.check_consistency()
        score = sum(s["severity_score"] for s in suggestions)
        return {"overall_risk": "high" if score >= 6 else "medium" if score >= 3 else "low", "issue_count": len(suggestions), "consistency_issues": consistency, "top_concerns": [s["message"] for s in suggestions if s["severity_score"] == 3]}

    def inspect(self):
        analysis = self.analyze(); suggestions = self.suggest(as_dict=True); risk = self.risk_summary()
        return InspectionReport(shape=analysis["shape"], risk=risk, suggestions=suggestions, consistency_issues=self.check_consistency())

    def to_markdown(self):
        return self.inspect().to_markdown()
