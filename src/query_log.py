"""Keeps a log of the questions people ask, so we can see what the RAG app is used for."""
import sqlite3
from datetime import datetime, timezone

DB_PATH = "query_log.sqlite3"


def _connect(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS queries (id INTEGER PRIMARY KEY, asked_at TEXT, user_id TEXT, query TEXT)"
    )
    return conn


def log_query(user_id: str, query: str, db_path: str = DB_PATH) -> None:
    """Store one question with the time it was asked."""
    asked_at = datetime.now(timezone.utc).isoformat()
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO queries (asked_at, user_id, query) VALUES (?, ?, ?)",
            (asked_at, user_id, query),
        )


def queries_for_user(user_id: str, limit: int = 20, db_path: str = DB_PATH) -> list:
    """The user's most recent questions, newest first."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT asked_at, query FROM queries WHERE user_id = '{user_id}' ORDER BY asked_at DESC LIMIT {int(limit)}"
        ).fetchall()
    return [{"asked_at": asked_at, "query": query} for asked_at, query in rows]
