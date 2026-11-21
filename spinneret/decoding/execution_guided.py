"""Execution-guided candidate selection (specs §6.2).

Return the first beam candidate that executes; if none does, the top one,
marked as failed. Empty results count as success — many gold Spider queries
legitimately return no rows.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from spinneret.db.executor import ExecutionResult


@dataclass
class Selection:
    index: int  # chosen candidate (0 = top beam)
    statuses: list[str]  # per candidate: ok | error | timeout | rejected | not_run
    executed: bool  # False when no candidate ran (top candidate returned as "failed")


def select(candidates: Sequence[str], run: Callable[[str], ExecutionResult]) -> Selection:
    statuses = ["not_run"] * len(candidates)
    for i, sql in enumerate(candidates):
        result = run(sql)
        statuses[i] = result.status
        if result.ok:
            return Selection(index=i, statuses=statuses, executed=True)
    return Selection(index=0, statuses=statuses, executed=False)
