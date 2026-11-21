"""SQL statement guard (specs §7).

Defence in depth on top of the read-only SQLite connection: gives a clear
message for anything that is not a single read-only query.
"""

from __future__ import annotations

import sqlparse
from sqlparse import tokens as T

FORBIDDEN = frozenset(
    "INSERT UPDATE DELETE DROP ALTER CREATE ATTACH DETACH PRAGMA VACUUM REINDEX TRUNCATE".split()
)


class GuardError(ValueError):
    """The SQL is not a single read-only SELECT/WITH statement."""


def check_sql(sql: str) -> str:
    """Return ``sql`` without a trailing semicolon, or raise ``GuardError``."""
    text = sql.strip()
    if not text:
        raise GuardError("empty query")
    statements = [s for s in sqlparse.parse(text) if str(s).strip().strip(";").strip()]
    if len(statements) != 1:
        raise GuardError("only one statement can run at a time")
    statement = statements[0]

    first = statement.token_first(skip_cm=True, skip_ws=True)
    if first is None or first.normalized not in ("SELECT", "WITH"):
        raise GuardError("only read-only SELECT queries can run here")

    flat = [t for t in statement.flatten() if not t.is_whitespace and t.ttype not in T.Comment]
    for i, tok in enumerate(flat):
        if tok.ttype not in T.Keyword:
            continue
        word = tok.normalized.upper()
        if word in FORBIDDEN:
            raise GuardError(f"only read-only SELECT queries can run here ({word} is not allowed)")
        # REPLACE is also a string function; only REPLACE INTO is a write.
        if word == "REPLACE" and i + 1 < len(flat) and flat[i + 1].normalized.upper() == "INTO":
            raise GuardError("only read-only SELECT queries can run here (REPLACE is not allowed)")
    return str(statement).strip().rstrip(";").strip()
