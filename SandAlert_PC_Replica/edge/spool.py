from __future__ import annotations

import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path


class SpoolFullError(RuntimeError):
    pass


@dataclass(frozen=True)
class SpoolItem:
    id: int
    session_id: str
    sequence: int
    kind: str
    payload_path: Path
    size_bytes: int


class DurableSpool:
    def __init__(self, root: str | Path, max_gib: float = 20.0):
        self.root = Path(root)
        self.payload_dir = self.root / "payloads"
        self.payload_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_gib * 1024 ** 3)
        self.db_path = self.root / "spool.sqlite3"
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS spool_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                kind TEXT NOT NULL,
                payload_path TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                UNIQUE(session_id, sequence, kind)
            )
            """
        )
        self._connection.commit()

    def total_bytes(self) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COALESCE(SUM(size_bytes), 0) FROM spool_items"
            ).fetchone()
        return int(row[0])

    def enqueue(self, session_id: str, sequence: int, kind: str, payload: bytes) -> None:
        with self._lock:
            exists = self._connection.execute(
                "SELECT 1 FROM spool_items WHERE session_id=? AND sequence=? AND kind=?",
                (session_id, sequence, kind),
            ).fetchone()
            if exists:
                return
            current = self._connection.execute(
                "SELECT COALESCE(SUM(size_bytes), 0) FROM spool_items"
            ).fetchone()[0]
            if current + len(payload) > self.max_bytes:
                raise SpoolFullError(
                    f"edge spool reached {self.max_bytes / 1024 ** 3:.1f} GiB"
                )
            safe_session = session_id.replace("/", "_")
            final_path = self.payload_dir / f"{safe_session}-{kind}-{sequence:020d}.bin"
            temp_path = final_path.with_suffix(".tmp")
            with temp_path.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temp_path.replace(final_path)
            self._connection.execute(
                "INSERT INTO spool_items(session_id,sequence,kind,payload_path,size_bytes) "
                "VALUES(?,?,?,?,?)",
                (session_id, sequence, kind, str(final_path), len(payload)),
            )
            self._connection.commit()

    def first_pending(self) -> SpoolItem | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT id,session_id,sequence,kind,payload_path,size_bytes "
                "FROM spool_items ORDER BY id LIMIT 1"
            ).fetchone()
        return SpoolItem(*row[:-2], Path(row[-2]), row[-1]) if row else None

    def read(self, item: SpoolItem) -> bytes:
        return item.payload_path.read_bytes()

    def acknowledge(self, session_id: str, sequence: int, kind: str) -> None:
        with self._lock:
            row = self._connection.execute(
                "SELECT id,payload_path FROM spool_items "
                "WHERE session_id=? AND sequence=? AND kind=?",
                (session_id, sequence, kind),
            ).fetchone()
            if row is None:
                return
            Path(row[1]).unlink(missing_ok=True)
            self._connection.execute("DELETE FROM spool_items WHERE id=?", (row[0],))
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()
