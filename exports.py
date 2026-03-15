from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from events import SCOUTING_FIELDS
from storage import get_all_events

EXPORT_DIR = Path("data/exports")


def _flatten_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for event in events:
        row: Dict[str, Any] = {
            "event_id": event["id"],
            "match_id": event["match_id"],
            "match_number": event["match_number"],
            "team_number": event["team_number"],
            "match_type": event.get("match_type"),
            "timestamp": event["timestamp"],
            "source_input": event.get("source_input"),
            "source_mode": event.get("source_mode"),
            "started_at": event.get("started_at"),
            "ended_at": event.get("ended_at"),
        }

        payload = event["payload"]
        for field in SCOUTING_FIELDS:
            field_data = payload.get(field, {})
            row[f"{field}_value"] = field_data.get("value")
            row[f"{field}_confidence"] = field_data.get("confidence")
        rows.append(row)
    return rows


def _build_export_df() -> pd.DataFrame:
    events = get_all_events()
    flat_rows = _flatten_events(events)
    return pd.DataFrame(flat_rows)


def export_csv() -> Path:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = EXPORT_DIR / f"scouting_export_{ts}.csv"
    _build_export_df().to_csv(output, index=False)
    return output


def export_xlsx() -> Path:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = EXPORT_DIR / f"scouting_export_{ts}.xlsx"
    _build_export_df().to_excel(output, index=False)
    return output
