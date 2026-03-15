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
                team_number INTEGER NOT NULL DEFAULT 0,
                source_input TEXT,
                source_mode TEXT DEFAULT 'auto',
                started_at TEXT,
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
        _ensure_matches_columns(conn)


def _ensure_matches_columns(conn: sqlite3.Connection) -> None:
    rows = conn.execute("PRAGMA table_info(matches)").fetchall()
    existing = {row[1] for row in rows}
    if "source_input" not in existing:
        conn.execute("ALTER TABLE matches ADD COLUMN source_input TEXT")
    if "source_mode" not in existing:
        conn.execute("ALTER TABLE matches ADD COLUMN source_mode TEXT DEFAULT 'auto'")
    if "started_at" not in existing:
        conn.execute("ALTER TABLE matches ADD COLUMN started_at TEXT")
    if "ended_at" not in existing:
        conn.execute("ALTER TABLE matches ADD COLUMN ended_at TEXT")


def _has_started_at_default(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='matches'").fetchone()
    if not row or not row[0]:
        return False
    return "started_at TEXT DEFAULT CURRENT_TIMESTAMP" in row[0]


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def create_match(match_number: int, team_number: int = 0) -> int:
    with get_conn() as conn:
        if _has_started_at_default(conn):
            cur = conn.execute(
                "INSERT INTO matches (match_number, team_number, started_at) VALUES (?, ?, NULL)",
                (match_number, team_number),
            )
        else:
            cur = conn.execute(
                "INSERT INTO matches (match_number, team_number) VALUES (?, ?)",
                (match_number, team_number),
            )
        return int(cur.lastrowid)


def update_match_source(match_id: int, source_input: str, source_mode: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE matches SET source_input = ?, source_mode = ? WHERE id = ?",
            (source_input, source_mode, match_id),
        )


def mark_match_started(match_id: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE matches SET started_at = COALESCE(started_at, CURRENT_TIMESTAMP) WHERE id = ?",
            (match_id,),
        )


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
            SELECT
                e.id,
                e.match_id,
                e.timestamp,
                e.payload_json,
                m.match_number,
                m.team_number,
                m.source_input,
                m.source_mode,
                m.started_at,
                m.ended_at
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
                "source_input": row["source_input"],
                "source_mode": row["source_mode"],
                "started_at": row["started_at"],
                "ended_at": row["ended_at"],
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
        "source_input": row["source_input"],
        "source_mode": row["source_mode"],
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
    }
