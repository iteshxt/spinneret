"""SQL skeletons and the ``skeleton | sql`` target format (specs §5.7).

A skeleton keeps keywords, operators, aggregates and parentheses and replaces
every table, column, alias, ``*`` and literal with ``_``; then ``_ as _`` and runs
of ``_ , _`` collapse to ``_``::

    select t2.name from concert as t1 join stadium as t2 on t1.stadium_id = t2.stadium_id
    → select _ from _ join _ on _ = _
"""

from __future__ import annotations

import re

SEPARATOR = " | "

KEYWORDS = frozenset(
    """
    select from where group by order having limit offset join on as and or not in like glob
    between is null exists distinct all asc desc intersect union except inner left right outer
    cross natural using case when then else end cast count sum avg min max abs round length
    lower upper substr coalesce ifnull
    """.split()
)
OPERATORS = frozenset(
    {"=", "!=", "<>", ">", "<", ">=", "<=", "+", "-", "/", "%", "||", "(", ")", ","}
)
_TOKEN_RE = re.compile(r"'(?:[^']|'')*'|\S+")


def sql_skeleton(normalized_sql: str) -> str:
    """Skeleton of an already-normalized SQL string (see ``normalize_sql``)."""
    out: list[str] = []
    for tok in _TOKEN_RE.findall(normalized_sql):
        out.append(tok if tok in KEYWORDS or tok in OPERATORS else "_")

    changed = True
    while changed:
        changed = False
        for i in range(len(out) - 2):
            if out[i] == "_" and out[i + 2] == "_" and out[i + 1] in ("as", ","):
                del out[i + 1 : i + 3]
                changed = True
                break
    return " ".join(out)


def skeleton_target(normalized_sql: str) -> str:
    return f"{sql_skeleton(normalized_sql)}{SEPARATOR}{normalized_sql}"


def split_output(text: str) -> tuple[str | None, str]:
    """Split a model output into ``(skeleton, sql)``; ``skeleton`` is None if absent."""
    if SEPARATOR in text:
        skeleton, sql = text.split(SEPARATOR, 1)
        return skeleton.strip(), sql.strip()
    return None, text.strip()
