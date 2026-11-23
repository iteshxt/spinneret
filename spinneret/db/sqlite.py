"""Read-only SQLite access with a wall-clock timeout (specs §7).

This is the low-level layer; the statement guard and API-facing executor build
on it in week 2.
"""

from __future__ import annotations

import sqlite3
import time
import urllib.parse
from pathlib import Path
from typing import Any


class QueryTimeout(Exception):
    """Raised when a query runs longer than its time budget."""


class ResultTooLarge(Exception):
    """Raised when the fetched rows exceed their byte budget."""


def _row_bytes(row: tuple[Any, ...]) -> int:
    return sum(len(v) if isinstance(v, str | bytes) else 8 for v in row)


def _decode(raw: bytes) -> str:
    # Some Spider databases contain non-UTF-8 bytes.
    return raw.decode("utf-8", errors="replace")


def connect_readonly(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(path)
    uri = f"file:{urllib.parse.quote(str(path.resolve()))}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.text_factory = _decode
    conn.execute("PRAGMA query_only = ON")
    return conn


def execute(
    conn: sqlite3.Connection,
    sql: str,
    *,
    timeout_s: float | None = 5.0,
    max_rows: int | None = None,
    max_bytes: int | None = None,
) -> tuple[list[str], list[tuple[Any, ...]]]:
    """Run ``sql`` and return ``(column_names, rows)``.

    Raises ``QueryTimeout`` after ``timeout_s`` seconds and ``sqlite3.Error`` on SQL errors.
    With ``max_rows``, at most ``max_rows + 1`` rows are fetched (so callers can tell
    the result was truncated). With ``max_bytes``, rows are fetched one at a time and
    ``ResultTooLarge`` is raised once their text and blobs pass that many bytes.
    """
    timed_out = False
    if timeout_s is not None:
        deadline = time.monotonic() + timeout_s

        def handler() -> int:
            nonlocal timed_out
            if time.monotonic() > deadline:
                timed_out = True
                return 1
            return 0

        conn.set_progress_handler(handler, 10_000)
    try:
        cursor = conn.execute(sql)
        if max_bytes is None:
            rows = cursor.fetchall() if max_rows is None else cursor.fetchmany(max_rows + 1)
        else:
            rows, size = [], 0
            for row in cursor:
                size += _row_bytes(row)
                if size > max_bytes:
                    raise ResultTooLarge(f"the result is larger than {max_bytes:,} bytes")
                rows.append(row)
                if max_rows is not None and len(rows) > max_rows:
                    break
        columns = [d[0] for d in cursor.description or []]
        return columns, rows
    except sqlite3.OperationalError as exc:
        if timed_out:
            raise QueryTimeout(f"query exceeded {timeout_s}s") from exc
        raise
    finally:
        conn.set_progress_handler(None, 0)
