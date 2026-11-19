"""SQLite database management, schema initialization, and transaction handling.

Ensures the effect ledger persists independently of agent memory or checkpoint files.
Uses Write-Ahead Logging (WAL) and foreign key constraints for safe concurrent operations.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
from pathlib import Path
from typing import Generator, Optional, Union

DEFAULT_DB_PATH = Path("data") / "recovery_ledger.db"

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;

CREATE TABLE IF NOT EXISTS effect_records (
    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
    action_id TEXT UNIQUE NOT NULL,
    task_id TEXT NOT NULL,
    operation_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    status TEXT NOT NULL,
    params_json TEXT NOT NULL DEFAULT '{}',
    result_json TEXT,
    error_message TEXT,
    checkpoint_id TEXT,
    evidence_json TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_effect_operation_id ON effect_records (operation_id);
CREATE INDEX IF NOT EXISTS idx_effect_task_id ON effect_records (task_id);
CREATE INDEX IF NOT EXISTS idx_effect_action_id ON effect_records (action_id);
CREATE INDEX IF NOT EXISTS idx_effect_status ON effect_records (status);

CREATE TABLE IF NOT EXISTS recovery_decisions (
    decision_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    action_id TEXT NOT NULL,
    operation_id TEXT NOT NULL,
    checkpoint_id TEXT,
    outcome TEXT NOT NULL,
    is_safe_to_execute INTEGER NOT NULL,
    rationale TEXT NOT NULL,
    timestamp REAL NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '[]',
    invariants_json TEXT NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_decision_operation_id ON recovery_decisions (operation_id);
CREATE INDEX IF NOT EXISTS idx_decision_task_id ON recovery_decisions (task_id);
"""


class DatabaseManager:
    """Manages SQLite connections and transactions for the effect ledger."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        if db_path is None:
            self.db_path = DEFAULT_DB_PATH
        elif str(db_path) == ":memory:":
            self.db_path = Path(":memory:")
        else:
            self.db_path = Path(db_path)

        self._is_memory = str(self.db_path) == ":memory:"
        self._shared_memory_conn: Optional[sqlite3.Connection] = None

        if not self._is_memory:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # For in-memory databases, keep a persistent connection so tables persist during lifetime
            self._shared_memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._shared_memory_conn.row_factory = sqlite3.Row

        self.initialize_schema()

    def get_connection(self) -> sqlite3.Connection:
        """Returns a configured SQLite connection."""
        if self._is_memory and self._shared_memory_conn is not None:
            return self._shared_memory_conn

        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        return conn

    def initialize_schema(self) -> None:
        """Applies the database schema and indexes."""
        conn = self.get_connection()
        try:
            conn.executescript(SCHEMA_SQL)
            conn.commit()
        finally:
            if not self._is_memory:
                conn.close()

    @contextlib.contextmanager
    def transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        """Provides a transactional cursor block.

        Automatically commits when exiting normally or rolls back on exception.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()
            if not self._is_memory:
                conn.close()

    def close(self) -> None:
        """Closes any persistent memory connection."""
        if self._shared_memory_conn:
            self._shared_memory_conn.close()
            self._shared_memory_conn = None
