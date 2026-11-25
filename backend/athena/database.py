"""SQLite: saved memories, conversations, and a log of every answered query.

One small local file (ATHENA_DB_PATH, default backend/athena.db). The query
log is what the /api/metrics endpoint and the UI's stats panel read: latency,
token counts and how often Athena abstained, measured on real use.
"""

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .config import BACKEND_DIR

DB_PATH = Path(os.getenv("ATHENA_DB_PATH", BACKEND_DIR / "athena.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT UNIQUE,
    value TEXT
);
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created REAL NOT NULL,
    updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,              -- 'user' | 'assistant'
    content TEXT NOT NULL,
    meta TEXT,                       -- JSON: sources, grounding, timings
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_by_conversation ON messages(conversation_id, id);
CREATE TABLE IF NOT EXISTS query_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created REAL NOT NULL,
    found INTEGER NOT NULL,
    abstained_early INTEGER NOT NULL,
    grounded INTEGER,                -- NULL when not answered
    retrieval_ms INTEGER,
    generation_ms INTEGER,
    total_ms INTEGER,
    prompt_tokens INTEGER,
    output_tokens INTEGER,
    top_score REAL
);
"""


def get_connection(path: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(path or DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def connect():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def initialize_database() -> None:
    # The folder may not exist yet (e.g. a fresh container without a volume).
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(SCHEMA)


# ------------------------------------------------------------ conversations

def create_conversation(title: str) -> str:
    conversation_id = uuid.uuid4().hex[:12]
    now = time.time()
    with connect() as conn:
        conn.execute("INSERT INTO conversations VALUES (?, ?, ?, ?)",
                     (conversation_id, title[:80] or "New chat", now, now))
    return conversation_id


def add_message(conversation_id: str, role: str, content: str, meta: dict | None = None) -> None:
    now = time.time()
    with connect() as conn:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, meta, created) VALUES (?, ?, ?, ?, ?)",
            (conversation_id, role, content, json.dumps(meta) if meta else None, now),
        )
        conn.execute("UPDATE conversations SET updated = ? WHERE id = ?", (now, conversation_id))


def conversation_exists(conversation_id: str) -> bool:
    with connect() as conn:
        return conn.execute("SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)).fetchone() is not None


def list_conversations(limit: int = 50) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, title, created, updated FROM conversations ORDER BY updated DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_messages(conversation_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT role, content, meta, created FROM messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"], "created": r["created"],
             "meta": json.loads(r["meta"]) if r["meta"] else None} for r in rows]


def history_pairs(conversation_id: str, limit: int = 3) -> list[tuple[str, str]]:
    """The last (question, answer) pairs, for rewriting follow-up questions."""
    messages = get_messages(conversation_id)
    pairs, question = [], None
    for m in messages:
        if m["role"] == "user":
            question = m["content"]
        elif question is not None:
            pairs.append((question, m["content"]))
            question = None
    return pairs[-limit:]


def delete_conversation(conversation_id: str) -> bool:
    with connect() as conn:
        return conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,)).rowcount > 0


# ---------------------------------------------------------------- metrics

def log_query(found: bool, abstained_early: bool, grounded: bool | None, retrieval_ms: int,
              generation_ms: int, total_ms: int, prompt_tokens: int | None, output_tokens: int | None,
              top_score: float | None) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO query_log (created, found, abstained_early, grounded, retrieval_ms, generation_ms,"
            " total_ms, prompt_tokens, output_tokens, top_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (time.time(), int(found), int(abstained_early), None if grounded is None else int(grounded),
             retrieval_ms, generation_ms, total_ms, prompt_tokens, output_tokens, top_score),
        )


def _percentile(values: list[int], p: float) -> int | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p * (len(values) - 1))))]


def query_metrics() -> dict:
    with connect() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM query_log").fetchall()]
    answered = [r for r in rows if r["found"]]
    total = [r["total_ms"] for r in rows if r["total_ms"] is not None]
    first = [r["generation_ms"] for r in answered if r["generation_ms"]]
    return {
        "queries": len(rows),
        "answered": len(answered),
        "not_found": len(rows) - len(answered),
        "abstained_without_llm": sum(r["abstained_early"] for r in rows),
        "grounded_answers": sum(1 for r in answered if r["grounded"] == 1),
        "flagged_answers": sum(1 for r in answered if r["grounded"] == 0),
        "latency_ms": {"p50": _percentile(total, 0.5), "p95": _percentile(total, 0.95)},
        "generation_ms": {"p50": _percentile(first, 0.5)},
        "tokens": {
            "prompt": sum(r["prompt_tokens"] or 0 for r in rows),
            "output": sum(r["output_tokens"] or 0 for r in rows),
        },
    }
