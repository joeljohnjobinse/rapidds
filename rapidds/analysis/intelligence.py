from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
import re

import numpy as np
import pandas as pd


_LEVEL = {"low": 1, "medium": 2, "high": 3}
_CONFIDENCE = {"low": 1, "medium": 2, "high": 3}
_RISK = {"low": 1, "medium": 2, "high": 3}

ID_TOKENS = {
    "id", "identifier", "uuid", "guid", "key", "customerid", "userid",
    "user_id", "account_id", "transaction_id", "record_id", "row_id"
}
DATE_TOKENS = {"date", "time", "timestamp", "created", "updated", "joined", "dob", "birth"}
TARGET_TOKENS = {"target", "label", "class", "outcome", "response", "y", "fraud", "churn", "default"}
LEAKAGE_TOKENS = {"target", "label", "outcome", "prediction", "predicted", "probability", "score", "future", "post", "review"}


def _name_tokens(name: Any) -> set[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(name)).lower()
    return {p for p in re.split(r"[^a-z0-9]+", text) if p}


def _safe_skew(s: pd.Series) -> float:
    try:
        value = float(s.dropna().skew())
        return 0.0 if np.isnan(value) else value
    except Exception:
        return 0.0


def _numeric_like_ratio(s: pd.Series) -> float:
    values = s.dropna().astype(str)
    if values.empty:
        return 0.0
    return float(pd.to_numeric(values, errors="coerce").notna().mean())


def _datetime_parse_ratio(s: pd.Series) -> float:
    values = s.dropna()
    if values.empty:
        return 0.0
    try:
        try:
            parsed = pd.to_datetime(values, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            parsed = pd.to_datetime(values, errors="coerce")
        return float(parsed.notna().mean())
    except Exception:
        return 0.0


def infer_task(df: pd.DataFrame, target: Optional[str] = None) -> Dict[str, Any]:
    """Infer a conservative ML task from a supplied or likely target."""
    if target is None or target not in df.columns:
        return {"target": None, "task": None, "confidence": "low", "reason": "No explicit target supplied."}
    y = df[target].dropna()
    if y.empty:
        return {"target": target, "task": None, "confidence": "high", "reason": "Target has no observed values."}
    unique = int(y.nunique())
    if pd.api.types.is_numeric_dtype(y) and unique > max(20, int(len(y) * 0.05)):
        task = "regression"
    elif unique == 2:
        task = "binary_classification"
    elif unique <= 20:
        task = "multiclass_classification"
    else:
        task = "regression"
    return {
        "target": target,
        "task": task,
        "confidence": "high",
        "unique_values": unique,
        "observations": int(len(y)),
    }


def infer_role(df: pd.DataFrame, column: str, target: Optional[str] = None) -> Dict[str, Any]:
    """Infer a likely semantic role using names, dtype, cardinality and value patterns."""
    s = df[column]
    rows = len(df)
    non_null = s.dropna()
    unique = int(s.nunique(dropna=True))
    ratio = unique / max(rows, 1)
    tokens = _name_tokens(column)
    name = str(column).lower()
    reasons: List[str] = []

    if column == target:
        return {"role": "target", "confidence": "high", "score": 1.0, "reasons": ["explicit target"]}

    if tokens & ID_TOKENS or (rows >= 20 and ratio >= 0.98):
        score = 0.95 if tokens & ID_TOKENS else 0.78
        reasons.append("identifier-like name" if tokens & ID_TOKENS else "near-unique values")
        return {"role": "identifier", "confidence": "high" if score >= .9 else "medium", "score": score, "reasons": reasons}

    if tokens & DATE_TOKENS:
        parse_ratio = _datetime_parse_ratio(s) if not pd.api.types.is_datetime64_any_dtype(s) else 1.0
        if parse_ratio >= .75 or pd.api.types.is_datetime64_any_dtype(s):
            reasons.append("datetime-like name and values")
            return {"role": "datetime", "confidence": "high", "score": min(1.0, .7 + parse_ratio * .3), "reasons": reasons}

    if pd.api.types.is_datetime64_any_dtype(s):
        return {"role": "datetime", "confidence": "high", "score": 1.0, "reasons": ["datetime dtype"]}

    if pd.api.types.is_numeric_dtype(s):
        if tokens & {"age", "years", "count", "quantity", "amount", "price", "income", "salary", "rate", "score", "percent"}:
            return {"role": "numeric_measure", "confidence": "medium", "score": .75, "reasons": ["numeric semantic name"]}
        return {"role": "numeric_feature", "confidence": "high", "score": .9, "reasons": ["numeric dtype"]}

    if pd.api.types.is_bool_dtype(s):
        return {"role": "boolean", "confidence": "high", "score": .95, "reasons": ["boolean dtype"]}

    if not non_null.empty:
        numeric_ratio = _numeric_like_ratio(s)
        if numeric_ratio >= .9:
            return {"role": "numeric_string", "confidence": "high", "score": numeric_ratio, "reasons": ["numeric-looking text"]}
        if unique <= 50 and ratio <= .5:
            return {"role": "categorical", "confidence": "high", "score": .85, "reasons": ["repeated categorical values"]}
        if "email" in name:
            return {"role": "contact", "confidence": "high", "score": .95, "reasons": ["email-like name"]}
        if "phone" in name or "mobile" in name:
            return {"role": "contact", "confidence": "medium", "score": .85, "reasons": ["phone-like name"]}
        if "url" in name or "link" in name:
            return {"role": "web_identifier", "confidence": "medium", "score": .85, "reasons": ["URL-like name"]}
        if ratio > .5:
            return {"role": "high_cardinality_text", "confidence": "medium", "score": .75, "reasons": ["high cardinality text"]}
    return {"role": "feature", "confidence": "low", "score": .5, "reasons": ["no stronger semantic signal"]}


def _score_suggestion(level: str, risk: str, confidence: str, evidence: Dict[str, Any]) -> float:
    """Calculate a transparent 0-1 recommendation confidence score."""
    base = 0.35 + 0.15 * _LEVEL.get(level, 1) + 0.15 * _CONFIDENCE.get(confidence, 1) - 0.10 * (_RISK.get(risk, 2) - 1)
    # More independent evidence should increase confidence, but with diminishing returns.
    evidence_count = sum(v is not None for v in evidence.values())
    base += min(0.18, evidence_count * 0.025)
    return round(float(max(0.0, min(1.0, base))), 3)


def rank_suggestions(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    confidence = {"high": 3, "medium": 2, "low": 1}
    risk = {"low": 1, "medium": 2, "high": 3}
    out = list(items)
    out.sort(key=lambda x: (
        -float(x.get("priority_score", 0)),
        -_LEVEL.get(x.get("level", "low"), 1),
        -confidence.get(x.get("confidence", "low"), 1),
        risk.get(x.get("risk", "high"), 3),
    ))
    for i, item in enumerate(out, 1):
        item["priority"] = i
    return out


@dataclass
class SuggestionPlan:
    """Actionable, inspectable suggestion plan produced by rapidds."""
    suggestions: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        return iter(self.suggestions)

    def __len__(self):
        return len(self.suggestions)

    def __getitem__(self, item):
        return self.suggestions[item]

    def as_dict(self) -> List[Dict[str, Any]]:
        return self.suggestions

    def summary(self) -> Dict[str, Any]:
        levels = {k: sum(s.get("level") == k for s in self.suggestions) for k in _LEVEL}
        actions = {}
        for s in self.suggestions:
            actions[s.get("action")] = actions.get(s.get("action"), 0) + 1
        return {"context": self.context, "counts": levels, "actions": actions, "total": len(self.suggestions)}

    def show(self) -> str:
        lines = ["rapidds suggestion plan", "=" * 24]
        if self.context:
            lines.append(f"Task: {self.context.get('task') or 'general analysis'}")
            if self.context.get("target"):
                lines.append(f"Target: {self.context['target']}")
        for s in self.suggestions:
            lines.append(f"{s['priority']}. [{s['level'].upper()}] {s['message']}")
            lines.append(f"   action={s.get('action')} confidence={s.get('confidence')} risk={s.get('risk')} score={s.get('confidence_score')}")
            if s.get("depends_on"):
                lines.append(f"   depends_on={', '.join(s['depends_on'])}")
        return "\n".join(lines)

    def approve_all_safe(self) -> List[Dict[str, Any]]:
        return [s for s in self.suggestions if s.get("confidence") == "high" and s.get("risk") == "low"]

    def approve(self, priority: int) -> Dict[str, Any]:
        for s in self.suggestions:
            if s.get("priority") == priority:
                s["approved"] = True
                return s
        raise IndexError(f"No suggestion with priority {priority}")


def build_context(df: pd.DataFrame, target: Optional[str] = None, profile: str = "general") -> Dict[str, Any]:
    rows, cols = df.shape
    inferred = target
    if inferred is None and cols:
        candidates = []
        for c in df.columns:
            vals = df[c].dropna()
            if 2 <= vals.nunique() <= min(10, max(2, int(max(len(vals), 1) * .1))):
                tokens = _name_tokens(c)
                score = 2 if tokens & TARGET_TOKENS else 0
                if c == df.columns[-1]: score += 1
                candidates.append((score, c))
        if candidates:
            candidates.sort(reverse=True)
            if candidates[0][0] >= 2:
                inferred = candidates[0][1]
    task_info = infer_task(df, inferred)
    roles = {str(c): infer_role(df, c, inferred) for c in df.columns}
    numeric = [c for c in df.columns if roles[str(c)]["role"] in {"numeric_feature", "numeric_measure"}]
    categorical = [c for c in df.columns if roles[str(c)]["role"] in {"categorical", "high_cardinality_text"}]
    ids = [c for c in df.columns if roles[str(c)]["role"] == "identifier"]
    dates = [c for c in df.columns if roles[str(c)]["role"] == "datetime"]
    return {
        "rows": int(rows), "columns": int(cols), "target": inferred,
        "task": task_info.get("task"), "task_confidence": task_info.get("confidence"),
        "profile": profile, "numeric_columns": numeric, "categorical_columns": categorical,
        "identifier_columns": ids, "datetime_columns": dates, "roles": roles,
        "duplicate_ratio": float(df.duplicated().mean()) if rows else 0.0,
    }
