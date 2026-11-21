"""Safe query execution for prediction, evaluation and the API (specs §7)."""

from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from spinneret.db.guard import GuardError, check_sql
from spinneret.db.sqlite import QueryTimeout, connect_readonly, execute

Status = Literal["ok", "error", "timeout", "rejected"]


@dataclass
class ExecutionResult:
    status: Status
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def to_json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return f"<blob {len(value)} bytes>"
    return value


class Executor:
    """Guarded, read-only, time- and row-limited execution against Spider databases."""

    def __init__(
        self, db_path: Callable[[str], Path], *, timeout_s: float = 5.0, max_rows: int = 500
    ) -> None:
        self._db_path = db_path
        self.timeout_s = timeout_s
        self.max_rows = max_rows

    def run(
        self,
        db_id: str,
        sql: str,
        *,
        timeout_s: float | None = None,
        max_rows: int | None = None,
    ) -> ExecutionResult:
        try:
            query = check_sql(sql)
        except GuardError as exc:
            return ExecutionResult(status="rejected", error=str(exc))

        limit = self.max_rows if max_rows is None else max_rows
        conn = connect_readonly(self._db_path(db_id))
        try:
            columns, rows = execute(
                conn,
                query,
                timeout_s=self.timeout_s if timeout_s is None else timeout_s,
                max_rows=limit,
            )
        except QueryTimeout as exc:
            return ExecutionResult(status="timeout", error=str(exc))
        except (sqlite3.Error, sqlite3.Warning, ValueError) as exc:
            return ExecutionResult(status="error", error=str(exc))
        finally:
            conn.close()

        truncated = len(rows) > limit
        rows = rows[:limit]
        return ExecutionResult(
            status="ok",
            columns=columns,
            rows=[[to_json_value(v) for v in row] for row in rows],
            row_count=len(rows),
            truncated=truncated,
        )


def results_equal(a: list[Any], b: list[Any], *, order_matters: bool) -> bool:
    """Compare two result sets as multisets of rows (or as sequences if order matters)."""
    rows_a = [tuple(r) for r in a]
    rows_b = [tuple(r) for r in b]
    if order_matters:
        return rows_a == rows_b
    return Counter(rows_a) == Counter(rows_b)
