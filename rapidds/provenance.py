"""Lightweight transformation history for rapidds Dataset."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone


def make_history_entry(action, *, details=None, rows_before=None, rows_after=None,
                       columns_before=None, columns_after=None, source="rapidds"):
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "source": source,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "columns_before": columns_before,
        "columns_after": columns_after,
        "details": deepcopy(details or {}),
    }
