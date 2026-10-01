from __future__ import annotations

import copy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from rapidds.analysis import (
    profile_dataframe, quality_checks, correlation, covariance, bootstrap,
    t_test, chi_square, anova, detect_outliers,
    SuggestionPlan, build_context, infer_role, infer_task, rank_suggestions,
)
from rapidds.analysis.intelligence import _safe_skew, _numeric_like_ratio, _datetime_parse_ratio, _score_suggestion, _name_tokens, _LEVEL, LEAKAGE_TOKENS
from rapidds.cleaning import fill_missing, standardize_text, CleaningPolicy, action_allowed
from rapidds.transform import scale, encode_categorical, datetime_features
from rapidds.modeling import evaluate_classification, evaluate_regression, evaluate_clustering
from rapidds.report import InspectionReport
from rapidds.provenance import make_history_entry
from rapidds.export import export_dataset, quality_snapshot

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
        self.history = []
        self._initial_quality_snapshot = quality_snapshot(self.df)
        self._rapidds_version = "0.2.2"

    def _record_change(self, action, details=None, before=None, after=None, source="rapidds"):
        before = self.df if before is None else before
        after = self.df if after is None else after
        self.history.append(make_history_entry(
            action, details=details, source=source,
            rows_before=len(before), rows_after=len(after),
            columns_before=list(before.columns), columns_after=list(after.columns),
        ))

    def history_report(self):
        """Return a copy of the tracked transformation history."""
        return copy.deepcopy(self.history)

    def clear_history(self):
        """Clear transformation history and start a new provenance session."""
        self.history = []
        self._initial_quality_snapshot = quality_snapshot(self.df)
        return self

    def export(self, path, *, include_history=True, include_analysis=True,
               include_suggestions=True, preview_rows=10):
        """Export the current dataset, optionally with provenance and analysis."""
        return export_dataset(
            self, path, include_history=include_history,
            include_analysis=include_analysis, include_suggestions=include_suggestions,
            preview_rows=preview_rows,
        )

    def report(self, path, *, include_history=True, include_analysis=True,
               include_suggestions=True, preview_rows=10):
        """Export a human-readable report. Currently supports HTML and Excel."""
        suffix = str(path).lower()
        if not suffix.endswith((".html", ".htm", ".xlsx", ".xlsm")):
            raise ValueError("Report format must be .html or .xlsx")
        return self.export(path, include_history=include_history,
                           include_analysis=include_analysis,
                           include_suggestions=include_suggestions,
                           preview_rows=preview_rows)

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
    def suggest(self, as_dict: bool = False, target=None, include_low_confidence=True,
                profile="general", aggressiveness="balanced", plan: bool = False):
        """Generate context-aware recommendations using evidence and compound rules.

        ``plan=True`` returns a :class:`SuggestionPlan`. The default structured
        output remains a list for backwards compatibility.
        """
        df = self.df
        profile = str(profile).lower()
        aggressiveness = str(aggressiveness).lower()
        if profile not in {"general", "ml"}:
            raise ValueError("profile must be 'general' or 'ml'")
        if aggressiveness not in {"conservative", "balanced", "aggressive"}:
            raise ValueError("aggressiveness must be conservative, balanced, or aggressive")
        if target is not None and target not in df.columns:
            raise ValueError(f"Target column '{target}' not found")

        context = build_context(df, target=target, profile=profile)
        inferred_target = context.get("target")
        task = context.get("task")
        rows = len(df)
        suggestions = []
        added_keys = set()

        def add(level, category, message, action=None, confidence="medium", risk="medium",
                evidence=None, depends_on=None, priority_bonus=0.0, **extra):
            evidence = evidence or {}
            key = (action, extra.get("column"), extra.get("target"), tuple(extra.get("columns", [])) if isinstance(extra.get("columns"), list) else None)
            if key in added_keys:
                return
            if not include_low_confidence and confidence == "low":
                return
            score = _score_suggestion(level, risk, confidence, evidence)
            score = round(max(0.0, min(1.0, score + priority_bonus)), 3)
            suggestions.append({
                "level": level, "severity_score": _LEVEL[level], "category": category,
                "message": message, "action": action, "confidence": confidence,
                "confidence_score": score, "risk": risk, "evidence": evidence,
                "depends_on": depends_on or [], **extra,
                "approved": False,
            })
            added_keys.add(key)

        # Dataset-level signals.
        duplicate_count = int(df.duplicated().sum())
        if duplicate_count:
            ratio = duplicate_count / max(rows, 1)
            add("medium", "quality", f"{duplicate_count} exact duplicate rows detected ({ratio:.1%} of rows).",
                "drop_duplicates", "high", "low", {"duplicate_count": duplicate_count, "duplicate_ratio": ratio},
                priority_bonus=.08, count=duplicate_count, ratio=ratio)

        # Column signals + semantic roles.
        for col in df.columns:
            s = df[col]
            missing_ratio = float(s.isna().mean()) if rows else 0.0
            unique = int(s.nunique(dropna=True))
            unique_ratio = unique / max(rows, 1)
            numeric = pd.api.types.is_numeric_dtype(s)
            is_target = col == inferred_target
            role = context["roles"][str(col)]
            role_name = role["role"]

            if missing_ratio > 0:
                if is_target:
                    add("high", "target", f"Target '{col}' contains {missing_ratio:.1%} missing values; define how those rows should be handled before training.",
                        "inspect_target_missing", "high", "high", {"missing_ratio": missing_ratio}, column=col, ratio=missing_ratio, target=col)
                elif missing_ratio >= .70:
                    add("high", "missing", f"'{col}' is {missing_ratio:.1%} missing. Inspect whether it should be removed instead of imputed.",
                        "drop_column", "high", "medium" if missing_ratio == 1 else "high", {"missing_ratio": missing_ratio, "threshold": .70}, column=col, ratio=missing_ratio)
                else:
                    skew = _safe_skew(s) if numeric else 0.0
                    strategy = "median" if numeric and abs(skew) > 1 else ("mean" if numeric else "mode")
                    evidence = {"missing_ratio": missing_ratio, "skewness": skew, "strategy": strategy}
                    confidence = "high" if missing_ratio <= .50 else "medium"
                    add("medium" if missing_ratio > .30 else "low", "missing",
                        f"'{col}' has {missing_ratio:.1%} missing values; {strategy} imputation is the context-aware default.",
                        "impute", confidence, "low", evidence, column=col, ratio=missing_ratio, strategy=strategy)
                    if numeric and abs(skew) > 1 and missing_ratio > 0:
                        add("low", "statistics", f"'{col}' is strongly skewed; robust imputation is preferable to mean imputation.",
                            "prefer_robust_imputation", "high", "low", {"skewness": skew, "missing_ratio": missing_ratio},
                            depends_on=[f"impute:{col}"], column=col)

            if unique <= 1 and not is_target and missing_ratio < 1:
                add("medium", "quality", f"'{col}' has no variation and cannot provide useful predictive information.",
                    "drop_constant", "high", "low", {"unique": unique}, column=col)

            if rows >= 20 and unique_ratio < .01 and unique > 1 and not is_target:
                add("low", "quality", f"'{col}' has very low cardinality ({unique} unique values); inspect whether it is informative.",
                    "inspect_cardinality", "medium", "medium", {"unique": unique, "unique_ratio": unique_ratio}, column=col)

            if role_name == "identifier" and not is_target:
                add("medium", "semantic", f"'{col}' appears to be an identifier ({'; '.join(role['reasons'])}); dropping it may be appropriate for modeling, but not for record tracking.",
                    "inspect_identifier", role["confidence"], "medium", {"unique_ratio": unique_ratio, "role_score": role["score"], "reasons": role["reasons"]}, column=col)

            if role_name == "high_cardinality_text" and not is_target:
                add("medium" if unique_ratio > .8 else "low", "modeling",
                    f"'{col}' has high categorical cardinality ({unique} unique values); naive one-hot encoding may create a large feature space.",
                    "inspect_cardinality", "high", "medium", {"unique": unique, "unique_ratio": unique_ratio}, column=col)

            if not numeric and not is_target:
                values = s.dropna().astype(str)
                if not values.empty:
                    numeric_ratio = _numeric_like_ratio(s)
                    if numeric_ratio == 1:
                        add("medium", "types", f"'{col}' contains only numeric-looking values but is stored as text; it can be safely converted.",
                            "convert_numeric", "high", "low", {"numeric_like_ratio": numeric_ratio}, column=col, ratio=numeric_ratio)
                    elif numeric_ratio >= .8:
                        add("medium", "types", f"'{col}' is mostly numeric-looking but contains non-numeric values; inspect before converting.",
                            "inspect_numeric_conversion", "medium", "high", {"numeric_like_ratio": numeric_ratio}, column=col, ratio=numeric_ratio)
                    stripped = values.str.strip()
                    changed = int((stripped != values).sum())
                    if changed:
                        add("low", "formatting", f"'{col}' contains leading or trailing whitespace; trimming is low-risk.",
                            "strip_whitespace", "high", "low", {"affected_values": changed}, column=col)
                    lower_unique = stripped.str.lower().nunique()
                    if lower_unique < stripped.nunique():
                        add("low", "formatting", f"'{col}' contains labels that differ only by casing; standardize only when the semantic equivalence is clear.",
                            "standardize_case", "medium", "medium", {"unique_before": int(stripped.nunique()), "unique_after": int(lower_unique)}, column=col)

                parse_ratio = _datetime_parse_ratio(s)
                if parse_ratio >= .90 and unique > 1:
                    add("low", "datetime", f"'{col}' appears to contain datetime values; temporal features may be useful.",
                        "add_datetime_features", "high", "low", {"parse_ratio": parse_ratio}, column=col, parse_ratio=parse_ratio)

            if numeric and not is_target:
                vals = s.dropna()
                if len(vals) >= 8:
                    q1, q3 = vals.quantile([.25, .75]); iqr = q3 - q1
                    if iqr > 0:
                        mask = (vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)
                        count = int(mask.sum()); ratio = count / len(vals)
                        skew = _safe_skew(s)
                        if ratio >= .01:
                            add("medium" if ratio >= .05 else "low", "outliers",
                                f"'{col}' has {count} IQR outliers ({ratio:.1%}); investigate whether they are errors or legitimate extremes.",
                                "inspect_outliers", "high", "high", {"count": count, "ratio": ratio, "skewness": skew}, column=col, count=count, ratio=ratio)
                        if ratio >= .05 and abs(skew) > 1:
                            add("medium", "statistics", f"'{col}' is both highly skewed and outlier-heavy; inspect its distribution before scaling or trimming.",
                                "inspect_distribution", "high", "medium", {"outlier_ratio": ratio, "skewness": skew},
                                depends_on=[f"inspect_outliers:{col}"], column=col)

        # Task-specific reasoning.
        if inferred_target:
            y = df[inferred_target]
            counts = y.value_counts(normalize=True, dropna=True)
            if len(counts) >= 2 and task and "classification" in task and float(counts.iloc[0]) >= .80:
                add("medium", "modeling", f"Target '{inferred_target}' is imbalanced: the largest class represents {counts.iloc[0]:.1%} of observed targets.",
                    "stratify_target", "high", "medium", {"majority_ratio": float(counts.iloc[0]), "classes": int(len(counts)), "task": task},
                    target=inferred_target)
            if y.isna().any():
                add("high", "target", f"Target '{inferred_target}' has {int(y.isna().sum())} missing values; define target-row handling before training.",
                    "inspect_target_missing", "high", "high", {"missing": int(y.isna().sum())}, target=inferred_target)

            # Target-derived names and deterministic relationships are leakage warnings.
            for col in df.columns:
                if col == inferred_target:
                    continue
                tokens = _name_tokens(col)
                name_signal = bool(tokens & LEAKAGE_TOKENS)
                aligned = df[[col, inferred_target]].dropna()
                if len(aligned) < max(10, int(rows * .5)):
                    continue
                deterministic = False
                dominance = 0.0
                if aligned[col].nunique() <= max(20, int(rows * .02)) and aligned[inferred_target].nunique() <= 20:
                    table = pd.crosstab(aligned[col], aligned[inferred_target], normalize="index")
                    if not table.empty:
                        dominance = float(table.max(axis=1).mean())
                        deterministic = dominance >= .98
                numeric_corr = None
                if pd.api.types.is_numeric_dtype(aligned[col]) and pd.api.types.is_numeric_dtype(aligned[inferred_target]):
                    numeric_corr = float(aligned[col].corr(aligned[inferred_target]))
                if deterministic or name_signal or (numeric_corr is not None and abs(numeric_corr) >= .995):
                    evidence = {"deterministic_mapping": deterministic, "dominance": dominance, "name_signal": name_signal, "target_correlation": numeric_corr}
                    conf = "high" if deterministic or (numeric_corr is not None and abs(numeric_corr) >= .995) else "medium"
                    add("high", "modeling", f"'{col}' may contain target leakage for '{inferred_target}'; inspect whether it is available before the prediction point.",
                        "inspect_leakage", conf, "high", evidence, column=col, target=inferred_target)

            if task == "binary_classification" and rows < 1000 and len(counts) == 2 and float(counts.min()) < .10:
                add("medium", "modeling", "The dataset is relatively small and the minority class is sparse; use stratified validation and inspect class support.",
                    "stratify_and_validate", "high", "medium", {"rows": rows, "minority_ratio": float(counts.min()), "task": task}, target=inferred_target)

        # Cross-feature reasoning.
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != inferred_target]
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr().abs()
            pairs = []
            for i, left in enumerate(numeric_cols):
                for right in numeric_cols[i + 1:]:
                    value = corr.loc[left, right]
                    if pd.notna(value) and value >= .95:
                        pairs.append((left, right, float(value)))
            if pairs:
                add("medium", "statistics", f"{len(pairs)} numeric feature pair(s) have absolute correlation ≥ 0.95; inspect redundancy or multicollinearity.",
                    "inspect_correlation", "high", "medium", {"pairs": pairs[:20], "threshold": .95}, pairs=pairs[:20])

        feature_cols = [c for c in df.columns if c != inferred_target]
        non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
        if profile == "ml" and non_numeric:
            add("low", "modeling", f"{len(non_numeric)} feature(s) are non-numeric and will require encoding for many estimators.",
                "encode_categorical", "high", "low", {"columns": non_numeric, "task": task}, columns=non_numeric)
        if profile == "ml":
            ranges = {c: float(df[c].max() - df[c].min()) for c in numeric_cols if df[c].notna().any()}
            positive = [v for v in ranges.values() if v > 0]
            if len(positive) >= 2:
                spread = max(positive) / min(positive)
                if spread >= 1000:
                    add("low", "modeling", "Numeric features have very different ranges; scaling may help distance- or gradient-based models.",
                        "scale_features", "high", "low", {"range_ratio": spread, "ranges": ranges}, columns=numeric_cols)

        # Compound dataset-level rule: ID + high cardinality + ML context.
        if profile == "ml" and context["identifier_columns"]:
            for col in context["identifier_columns"]:
                if col != inferred_target:
                    add("medium", "modeling", f"'{col}' is identifier-like and is unlikely to be a useful predictive feature; keep it only if it has a documented modeling role.",
                        "inspect_identifier", "high", "medium", {"role": context["roles"][str(col)], "task": task}, column=col, depends_on=[])

        # Ordered workflow suggestions. Dependencies refer to action:column keys.
        action_order = {
            "drop_duplicates": 10, "convert_numeric": 20, "strip_whitespace": 25,
            "drop_constant": 30, "impute": 40, "add_datetime_features": 50,
            "encode_categorical": 60, "scale_features": 70, "inspect_outliers": 80,
            "inspect_leakage": 90, "inspect_correlation": 90, "stratify_target": 75,
        }
        for item in suggestions:
            item["workflow_stage"] = action_order.get(item.get("action"), 100)
            item["priority_score"] = round(item["confidence_score"] * 0.7 + _LEVEL[item["level"]] / 3 * .2 + (1 if item.get("risk") == "low" else 0.5) * .1 + (1 / (item["workflow_stage"] + 1)), 4)
        suggestions = rank_suggestions(suggestions)
        suggestion_plan = SuggestionPlan(suggestions=suggestions, context=context)
        if plan:
            return suggestion_plan
        if as_dict:
            return suggestions
        return [s["message"] for s in suggestions]

    def context(self, target=None, profile="general"):
        """Return dataset-level semantic and modeling context."""
        return build_context(self.df, target=target, profile=profile)

    def infer_task(self, target=None):
        """Infer classification/regression task information for a target."""
        return infer_task(self.df, target=target)

    def roles(self, columns=None, target=None):
        """Infer semantic roles for dataset columns."""
        columns = list(columns) if columns is not None else list(self.df.columns)
        return {str(c): infer_role(self.df, c, target=target) for c in columns}

    def suggestion_plan(self, target=None, profile="general", aggressiveness="balanced"):
        """Return the full actionable SuggestionPlan object."""
        return self.suggest(target=target, profile=profile, aggressiveness=aggressiveness, plan=True)

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
                   convert_numeric=True, drop_duplicates=True, inplace=True, plan=None):
        """Apply suggestions according to a named safety policy.

        Policies:
        - conservative: only high-confidence, low-risk, reversible-ish fixes.
        - balanced: additionally allows high-confidence column drops and label casing.
        - aggressive: permits medium-confidence, higher-risk cleanup and datetime feature extraction.

        ``dry_run=True`` returns the proposed DataFrame and an audit report without mutating ``self.df``.
        Explicit legacy flags remain supported and can only loosen the policy where requested.
        """
        policy_obj = CleaningPolicy.from_name(policy)
        if plan is None:
            suggestions = self.suggest(as_dict=True, target=target, profile=profile, aggressiveness=policy)
        elif isinstance(plan, SuggestionPlan):
            suggestions = plan.as_dict()
        elif isinstance(plan, list):
            suggestions = plan
        else:
            raise TypeError("plan must be a SuggestionPlan, list of suggestions, or None")
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
            before_df = self.df.copy()
            self.last_cleaning_report = report
            if inplace:
                self.df = work
                if actions:
                    self._record_change(
                        "auto_clean",
                        details={"policy": policy_obj.name, "profile": profile, "actions": actions, "skipped": skipped},
                        before=before_df, after=self.df, source="auto_clean",
                    )
                return self.df
        return work, report

    # ------------------------------------------------------------------
    # EXPLICIT CLEANING
    # ------------------------------------------------------------------
    def clean(self, num_missing=None, cat_missing=None, drop_duplicates=True):
        before = self.df.copy()
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
        self._record_change("clean", details={"num_missing": num_missing, "cat_missing": cat_missing, "drop_duplicates": drop_duplicates}, before=before, after=df, source="clean")
        return df

    def fill_missing(self, numeric="median", categorical="mode"):
        before = self.df.copy()
        self.df, actions = fill_missing(self.df, numeric, categorical)
        if actions:
            self._record_change("fill_missing", details={"numeric": numeric, "categorical": categorical, "actions": actions}, before=before, after=self.df, source="fill_missing")
        return actions

    def standardize_labels(self, columns, case="lower", strip_whitespace=True):
        before = self.df.copy()
        self.df, actions = standardize_text(self.df, columns, case, strip_whitespace)
        if actions:
            self._record_change("standardize_labels", details={"columns": list(columns), "case": case, "strip_whitespace": strip_whitespace, "actions": actions}, before=before, after=self.df, source="standardize_labels")
        return self.df

    def convert_numeric(self, columns=None):
        before = self.df.copy()
        cols = columns or [s["column"] for s in self.suggest(as_dict=True) if s.get("action") == "convert_numeric"]
        for col in cols:
            if col not in self.df.columns: raise ValueError(f"Column '{col}' not found")
            self.df[col] = pd.to_numeric(self.df[col], errors="raise")
        if cols:
            self._record_change("convert_numeric", details={"columns": list(cols)}, before=before, after=self.df, source="convert_numeric")
        return self.df

    def drop_columns(self, columns):
        before = self.df.copy()
        self.df = self.df.drop(columns=columns)
        self._record_change("drop_columns", details={"columns": list(columns)}, before=before, after=self.df, source="drop_columns")
        return self.df

    def drop_duplicates(self):
        before = self.df.copy()
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        if len(before) != len(self.df):
            self._record_change("drop_duplicates", details={"rows_removed": len(before) - len(self.df)}, before=before, after=self.df, source="drop_duplicates")
        return self.df

    # ------------------------------------------------------------------
    # TRANSFORMATION / FEATURES
    # ------------------------------------------------------------------
    def scale(self, columns=None, method="standard"):
        before = self.df.copy()
        self.df = scale(self.df, columns, method)
        self._record_change("scale", details={"columns": list(columns) if columns is not None else None, "method": method}, before=before, after=self.df, source="scale")
        return self.df

    def encode(self, columns=None, drop_first=False):
        before = self.df.copy()
        self.df = encode_categorical(self.df, columns, drop_first)
        self._record_change("encode", details={"columns": list(columns) if columns is not None else None, "drop_first": drop_first}, before=before, after=self.df, source="encode")
        return self.df

    def add_datetime_features(self, columns):
        before = self.df.copy()
        self.df = datetime_features(self.df, columns)
        self._record_change("add_datetime_features", details={"columns": list(columns)}, before=before, after=self.df, source="add_datetime_features")
        return self.df

    def log_transform(self, columns):
        before = self.df.copy()
        for col in columns:
            if (self.df[col].dropna() < 0).any(): raise ValueError(f"Column '{col}' contains negative values")
            self.df[col] = np.log1p(self.df[col])
        self._record_change("log_transform", details={"columns": list(columns)}, before=before, after=self.df, source="log_transform")
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
