"""Export and provenance-aware reporting helpers for rapidds."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (dict, list, tuple, set)) else False:
        return None
    return value


def _history_df(history: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for entry in history:
        row = dict(entry)
        if isinstance(row.get("details"), dict):
            details = row.pop("details")
            row.update({f"detail_{k}": _json_safe(v) for k, v in details.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _suggestions_df(suggestions: Optional[Iterable[Dict[str, Any]]]) -> pd.DataFrame:
    if not suggestions:
        return pd.DataFrame()
    rows = []
    for item in suggestions:
        row = dict(item)
        for key in ("evidence", "depends_on"):
            if key in row:
                row[key] = json.dumps(_json_safe(row[key]), default=str)
        rows.append(row)
    return pd.DataFrame(rows)


def quality_snapshot(df: pd.DataFrame) -> Dict[str, Any]:
    rows, cols = df.shape
    missing = int(df.isna().sum().sum())
    duplicate = int(df.duplicated().sum()) if rows else 0
    return {
        "rows": int(rows),
        "columns": int(cols),
        "missing_cells": missing,
        "missing_ratio": float(missing / max(rows * max(cols, 1), 1)),
        "duplicate_rows": duplicate,
        "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
    }


def build_report_data(dataset, include_analysis: bool = True) -> Dict[str, Any]:
    """Build a serializable report payload from a Dataset instance."""
    suggestions = dataset.suggest(as_dict=True)
    before = dataset._initial_quality_snapshot or quality_snapshot(dataset.df)
    after = quality_snapshot(dataset.df)
    payload = {
        "rapidds_version": getattr(dataset, "_rapidds_version", None),
        "quality_before": before,
        "quality_after": after,
        "history": list(dataset.history),
        "suggestions": suggestions,
        "cleaning_report": dataset.last_cleaning_report,
    }
    if include_analysis:
        payload["analysis"] = dataset.analyze()
    return _json_safe(payload)


def export_dataset(dataset, path, *, include_history=True, include_analysis=True,
                   include_suggestions=True, preview_rows=10):
    """Export data or a provenance-aware report based on file extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    df = dataset.df

    if ext == ".csv":
        df.to_csv(path, index=False)
    elif ext in {".xlsx", ".xlsm"}:
        _export_excel(dataset, path, include_history, include_analysis, include_suggestions, preview_rows)
    elif ext == ".json":
        payload = df.to_dict(orient="records")
        if include_history or include_analysis or include_suggestions:
            payload = {"data": _json_safe(payload)}
            report = build_report_data(dataset, include_analysis=include_analysis)
            if include_history:
                payload["history"] = report["history"]
            if include_suggestions:
                payload["suggestions"] = report["suggestions"]
            payload["quality"] = {"before": report["quality_before"], "after": report["quality_after"]}
        path.write_text(json.dumps(_json_safe(payload), indent=2, default=str), encoding="utf-8")
    elif ext == ".parquet":
        try:
            df.to_parquet(path, index=False)
        except ImportError as exc:
            raise ImportError("Parquet export requires 'pyarrow' or 'fastparquet'. Install one with `pip install pyarrow`.") from exc
    elif ext in {".html", ".htm"}:
        _export_html(dataset, path, include_history, include_analysis, include_suggestions, preview_rows)
    else:
        raise ValueError("Unsupported export format. Use .csv, .xlsx, .json, .parquet, or .html")
    return path


def _export_excel(dataset, path, include_history, include_analysis, include_suggestions, preview_rows):
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:
        raise ImportError("Excel export requires 'openpyxl'. Install it with `pip install openpyxl`.") from exc

    report = build_report_data(dataset, include_analysis=include_analysis)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        dataset.df.to_excel(writer, sheet_name="Cleaned_Data", index=False)
        pd.DataFrame([report["quality_before"], report["quality_after"]], index=["before", "after"]).to_excel(writer, sheet_name="Quality")
        if include_history:
            hist = _history_df(report["history"])
            if hist.empty:
                hist = pd.DataFrame({"message": ["No tracked changes."]})
            hist.to_excel(writer, sheet_name="Changes", index=False)
        if include_suggestions:
            sug = _suggestions_df(report["suggestions"])
            if sug.empty:
                sug = pd.DataFrame({"message": ["No suggestions generated."]})
            sug.to_excel(writer, sheet_name="Suggestions", index=False)
        if include_analysis:
            analysis = report.get("analysis", {})
            rows = []
            for key, value in analysis.items():
                if isinstance(value, (dict, list)):
                    value = json.dumps(_json_safe(value), default=str)
                rows.append({"metric": key, "value": value})
            pd.DataFrame(rows).to_excel(writer, sheet_name="Analysis", index=False)
        metadata = pd.DataFrame([
            {"field": "rapidds_version", "value": report.get("rapidds_version")},
            {"field": "rows", "value": len(dataset.df)},
            {"field": "columns", "value": len(dataset.df.columns)},
            {"field": "preview_rows", "value": preview_rows},
        ])
        metadata.to_excel(writer, sheet_name="Metadata", index=False)

        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            for column_cells in sheet.columns:
                width = min(max(len(str(cell.value or "")) for cell in column_cells) + 2, 45)
                sheet.column_dimensions[column_cells[0].column_letter].width = width


def _export_html(dataset, path, include_history, include_analysis, include_suggestions, preview_rows):
    report = build_report_data(dataset, include_analysis=include_analysis)
    sections = [
        "<h1>rapidds Data Report</h1>",
        f"<p><strong>Rows:</strong> {len(dataset.df)} &nbsp; <strong>Columns:</strong> {len(dataset.df.columns)}</p>",
        "<h2>Quality</h2>",
        pd.DataFrame([report["quality_before"], report["quality_after"]], index=["Before", "After"]).to_html(classes="quality", border=0),
        "<h2>Data Preview</h2>",
        dataset.df.head(preview_rows).to_html(index=False, border=0),
    ]
    if include_history:
        hist = _history_df(report["history"])
        sections += ["<h2>Changes</h2>", (hist.to_html(index=False, border=0) if not hist.empty else "<p>No tracked changes.</p>")]
    if include_suggestions:
        sug = _suggestions_df(report["suggestions"])
        sections += ["<h2>Suggestions</h2>", (sug.to_html(index=False, border=0) if not sug.empty else "<p>No suggestions generated.</p>")]
    if include_analysis:
        sections += ["<h2>Analysis</h2>", f"<pre>{json.dumps(report.get('analysis', {}), indent=2, default=str)}</pre>"]
    html = """<!doctype html><html><head><meta charset='utf-8'><title>rapidds Report</title>
<style>body{font-family:Arial,sans-serif;margin:32px;line-height:1.45}h1,h2{margin-top:28px}table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13px}th,td{border:1px solid #ddd;padding:7px;text-align:left}th{background:#f3f3f3}pre{background:#f6f6f6;padding:16px;overflow:auto}</style></head><body>""" + "".join(sections) + "</body></html>"
    path.write_text(html, encoding="utf-8")
