from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path("data/scouting.db")


def init_storage() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_number INTEGER NOT NULL,
                team_number INTEGER NOT NULL,
                started_at TEXT DEFAULT CURRENT_TIMESTAMP,
                ended_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scouting_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER NOT NULL,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                payload_json TEXT NOT NULL,
                FOREIGN KEY(match_id) REFERENCES matches(id)
            )
            """
        )


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def create_match(match_number: int, team_number: int) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO matches (match_number, team_number) VALUES (?, ?)",
            (match_number, team_number),
        )
        return int(cur.lastrowid)


def end_match(match_id: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE matches SET ended_at = CURRENT_TIMESTAMP WHERE id = ?",
            (match_id,),
        )


def insert_event(match_id: int, payload: Dict[str, Any]) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO scouting_events (match_id, payload_json) VALUES (?, ?)",
            (match_id, json.dumps(payload)),
        )


def get_all_events() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT e.id, e.match_id, e.timestamp, e.payload_json, m.match_number, m.team_number
            FROM scouting_events e
            JOIN matches m ON m.id = e.match_id
            ORDER BY e.id ASC
            """
        ).fetchall()

    events = []
    for row in rows:
        payload = json.loads(row["payload_json"])
        events.append(
            {
                "id": row["id"],
                "match_id": row["match_id"],
                "timestamp": row["timestamp"],
                "match_number": row["match_number"],
                "team_number": row["team_number"],
                "payload": payload,
            }
        )
    return events


def get_match_summary(match_id: int) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM matches WHERE id = ?",
            (match_id,),
        ).fetchone()

    if not row:
        return None

    return {
        "id": row["id"],
        "match_number": row["match_number"],
        "team_number": row["team_number"],
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
    }
