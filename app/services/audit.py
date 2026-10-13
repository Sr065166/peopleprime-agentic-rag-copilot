import json
import sqlite3
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from app.core.config import get_settings

settings = get_settings()


def get_db_connection() -> sqlite3.Connection:
    db_path = Path(settings.audit_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    return con


def init_audit_db() -> None:
    """Initialize audit logging SQLite table with all operational fields."""
    con = get_db_connection()
    try:
        con.execute(
            """CREATE TABLE IF NOT EXISTS query_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                request_id TEXT NOT NULL,
                question_hash TEXT NOT NULL,
                question_preview TEXT NOT NULL,
                stages_json TEXT NOT NULL,
                source_used TEXT NOT NULL,
                kb_grade TEXT,
                web_grade TEXT,
                retry_count INTEGER NOT NULL,
                duration_seconds REAL NOT NULL,
                final_outcome TEXT NOT NULL
            )"""
        )
        con.commit()
    finally:
        con.close()


def hash_question(question: str) -> str:
    """Hash question string with sha256 to allow safe tracking without storing raw sensitive employee inputs."""
    return hashlib.sha256(question.strip().encode("utf-8")).hexdigest()[:16]


def record_audit_entry(
    request_id: str,
    question: str,
    stages: list[str],
    source_used: str,
    kb_grade: str | None,
    web_grade: str | None,
    retry_count: int,
    duration_seconds: float,
    final_outcome: str,
) -> None:
    """
    Safely record operational execution data into SQLite audit database.
    Does not log API credentials or raw confidential payload bodies.
    """
    con = get_db_connection()
    try:
        q_hash = hash_question(question)
        # Store a safe sanitized preview (max 100 chars, no credentials)
        q_preview = question.strip()[:100]

        con.execute(
            """INSERT INTO query_audit (
                timestamp,
                request_id,
                question_hash,
                question_preview,
                stages_json,
                source_used,
                kb_grade,
                web_grade,
                retry_count,
                duration_seconds,
                final_outcome
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                request_id,
                q_hash,
                q_preview,
                json.dumps(stages),
                source_used,
                kb_grade,
                web_grade,
                retry_count,
                round(duration_seconds, 4),
                final_outcome,
            ),
        )
        con.commit()
    finally:
        con.close()


def get_recent_audit_logs(limit: int = 50) -> list[dict[str, Any]]:
    """Retrieve recent operational logs for administrative oversight."""
    con = get_db_connection()
    try:
        cur = con.execute(
            "SELECT * FROM query_audit ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
        return [dict(row) for row in rows]
    finally:
        con.close()
