import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from rapidds.report import InspectionReport


_SEVERITY_MAP = {
    "high": 3,
    "medium": 2,
    "low": 1,
}


class Dataset:
    """
    rapidds Dataset: detect, suggest, explain, and clean — explicitly.
    """

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Dataset expects a pandas DataFrame")

        self.df = df.copy()

    def __repr__(self):
        rows, cols = self.df.shape
        max_missing = self.df.isnull().mean().max() * 100 if rows > 0 else 0

        return (
            f"<rapidds.Dataset | "
            f"rows={rows}, cols={cols}, "
            f"max_missing={max_missing:.1f}%>"
        )

    # --------------------
    # DETECT
    # --------------------
    def analyze(self) -> dict:
        df = self.df

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        missing_ratio = df.isnull().mean()

        return {
            "shape": {
                "rows": df.shape[0],
                "columns": df.shape[1],
            },
            "column_types": {
                "numeric": numeric_cols,
                "categorical": categorical_cols,
            },
            "missing": {
                col: float(ratio)
                for col, ratio in missing_ratio.items()
                if ratio > 0
            },
        }

    # --------------------
    # SUGGEST
    # --------------------
    def suggest(self, as_dict: bool = False):
        df = self.df
        suggestions = []

        missing_ratio = df.isnull().mean()
        numeric_cols = df.select_dtypes(include=np.number).columns

        # Missing value suggestions
        for col, ratio in missing_ratio.items():
            if ratio > 0.7:
                level = "high"
                msg = f"Consider dropping '{col}' ({int(ratio*100)}% missing)."
                cat = "missing"
            elif ratio > 0.3:
                level = "medium"
                msg = f"'{col}' has significant missing values — consider targeted imputation."
                cat = "missing"
            elif ratio > 0:
                level = "low"
                strategy = (
                    "median or model-based imputation"
                    if col in numeric_cols
                    else "mode imputation or treating missing as a category"
                )
                msg = f"'{col}' has some missing values — consider {strategy}."
                cat = "missing"
            else:
                continue

            suggestions.append({
                "level": level,
                "severity_score": _SEVERITY_MAP[level],
                "category": cat,
                "message": msg,
            })

        # Constant columns
        for col in df.columns:
            if df[col].nunique(dropna=False) <= 1:
                suggestions.append({
                    "level": "medium",
                    "severity_score": _SEVERITY_MAP["medium"],
                    "category": "quality",
                    "message": f"'{col}' has a single unique value and may not be useful.",
                })

        # Potential target imbalance heuristic
        last_col = df.columns[-1]
        if df[last_col].nunique(dropna=True) < 10:
            ratio = df[last_col].value_counts(normalize=True)
            if not ratio.empty and ratio.iloc[0] > 0.8:
                suggestions.append({
                    "level": "medium",
                    "severity_score": _SEVERITY_MAP["medium"],
                    "category": "modeling",
                    "message": f"'{last_col}' appears imbalanced — consider stratified sampling or class weights.",
                })

        # Sort by severity (high → low)
        suggestions.sort(
            key=lambda x: x["severity_score"],
            reverse=True
        )

        if as_dict:
            return suggestions

        return [s["message"] for s in suggestions]

    # --------------------
    # EXPLAIN
    # --------------------
    def explain(self) -> None:
        suggestions = self.suggest(as_dict=True)

        if not suggestions:
            print("No major issues detected. Dataset looks clean 👍")
            return

        print("\nrapidds analysis summary")
        print("-" * 30)

        for level in ["high", "medium", "low"]:
            items = [s for s in suggestions if s["level"] == level]
            if items:
                print(f"\n{level.upper()} PRIORITY:")
                for item in items:
                    print(f" • {item['message']}")

        print("\n(Use ds.clean(...) to apply changes explicitly.)")

    # --------------------
    # MARKDOWN EXPORT
    # --------------------
    def to_markdown(self) -> str:
        suggestions = self.suggest(as_dict=True)

        if not suggestions:
            return "### rapidds Analysis\n\nNo major issues detected."

        lines = ["### rapidds Analysis\n"]

        for level in ["high", "medium", "low"]:
            items = [s for s in suggestions if s["level"] == level]
            if items:
                lines.append(f"#### {level.capitalize()} Priority\n")
                for item in items:
                    lines.append(f"- {item['message']}")
                lines.append("")

        return "\n".join(lines)

    # --------------------
    # EXECUTE
    # --------------------
    def clean(
        self,
        num_missing: str | None = None,
        cat_missing: str | None = None,
        drop_duplicates: bool = True,
    ) -> pd.DataFrame:
        df = self.df

        if drop_duplicates:
            df = df.drop_duplicates()

        if num_missing == "mean":
            num_cols = df.select_dtypes(include=np.number).columns
            df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

        elif num_missing == "median":
            num_cols = df.select_dtypes(include=np.number).columns
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        if cat_missing == "mode":
            cat_cols = df.select_dtypes(exclude=np.number).columns
            for col in cat_cols:
                if df[col].isnull().any():
                    mode = df[col].mode(dropna=True)
                    if not mode.empty:
                        df[col] = df[col].fillna(mode.iloc[0])

        self.df = df
        return df

    # --------------------
    # CHECK CONSISTENCY
    # --------------------
    def check_consistency(self) -> list[str]:
        df = self.df
        issues = []

        # Mixed dtypes
        for col in df.columns:
            types = df[col].dropna().map(type).unique()
            if len(types) > 1:
                issues.append(
                    f"Column '{col}' contains mixed data types."
                )

        # Numeric stored as strings
        for col in df.select_dtypes(include="object"):
            sample = df[col].dropna().head(10)
            if not sample.empty:
                try:
                    sample.astype(float)
                    issues.append(
                        f"Column '{col}' appears numeric but is stored as strings."
                    )
                except ValueError:
                    pass

        # Inconsistent casing / whitespace
        for col in df.select_dtypes(include="object"):
            values = df[col].dropna().astype(str)
            if values.str.strip().nunique() < values.nunique():
                issues.append(
                    f"Column '{col}' contains inconsistent whitespace."
                )
            if values.str.lower().nunique() < values.nunique():
                issues.append(
                    f"Column '{col}' contains inconsistent casing."
                )

        return issues

    # --------------------
    # STANDARDIZE LABELS
    # --------------------

    def standardize_labels(
        self,
        columns: list[str],
        case: str = "lower",
        strip_whitespace: bool = True,
    ) -> None:
        df = self.df

        for col in columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found")

            series = df[col].astype(str)

            if strip_whitespace:
                series = series.str.strip()

            if case == "lower":
                series = series.str.lower()
            elif case == "upper":
                series = series.str.upper()
            elif case is not None:
                raise ValueError("case must be 'lower', 'upper', or None")

            df[col] = series

        self.df = df

    # --------------------
    # PREPARE FOR MODELING
    # --------------------

    def prepare_for_modeling(self, target: str):
        df = self.df

        if target not in df.columns:
            raise ValueError(f"Target '{target}' not found")

        if df[target].isnull().any():
            raise ValueError("Target column contains missing values")

        X = df.drop(columns=[target])
        y = df[target]

        warnings = []

        if X.select_dtypes(exclude="number").shape[1] > 0:
            warnings.append(
                "Categorical features detected — encoding will be required."
            )

        if X.select_dtypes(include="number").isnull().any().any():
            warnings.append(
                "Numeric features contain missing values — imputation may be required."
            )

        return X, y, warnings


    # --------------------
    # SPLIT - THIN WRAPPER
    # --------------------
    from sklearn.model_selection import train_test_split

    def split(
        self,
        target: str,
        test_size: float = 0.2,
        stratify: bool = False,
        random_state: int = 42,
    ):
        X, y, warnings = self.prepare_for_modeling(target)

        strat = y if stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=strat,
            random_state=random_state,
        )

        return X_train, X_test, y_train, y_test, warnings


    # --------------------
    # RISK SUMMARY
    # --------------------
    def risk_summary(self) -> dict:
        suggestions = self.suggest(as_dict=True)
        consistency = self.check_consistency()

        risk_score = sum(s["severity_score"] for s in suggestions)

        if risk_score >= 6:
            level = "high"
        elif risk_score >= 3:
            level = "medium"
        else:
            level = "low"

        return {
            "overall_risk": level,
            "issue_count": len(suggestions),
            "consistency_issues": consistency,
            "top_concerns": [
                s["message"] for s in suggestions if s["severity_score"] == 3
            ],
        }

    # --------------------
    # INSPECT
    # --------------------
    def inspect(self):
        """
        Run a full dataset inspection and return an InspectionReport.
        """
        analysis = self.analyze()
        suggestions = self.suggest(as_dict=True)
        risk = self.risk_summary()
        consistency = self.check_consistency()

        return InspectionReport(
            shape=analysis["shape"],
            risk=risk,
            suggestions=suggestions,
            consistency_issues=consistency,
        )
