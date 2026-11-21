"""Question + schema → one model input string (specs §4.4, §5.6).

Full schema (T1)::

    <question> | <db_id> | <table> : <col> , <col> ( <value> , <value> ) | <table> : …

Ranked schema (T2, T3) keeps the ranker's top tables/columns in score order and
appends the foreign keys between kept tables::

    … | <table> : <col> , … | fk : <t.c> = <t.c> , …
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from spinneret.data.schema import Schema

TokenCounter = Callable[[str], int]
_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass
class SchemaView:
    """An ordered selection of tables and columns to render."""

    tables: list[tuple[str, list[str]]]  # (table name, column names) — original names
    foreign_keys: list[str] = field(default_factory=list)


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower().replace("_", " ")))


def render(
    question: str, db_id: str, view: SchemaView, values: dict[str, list[str]] | None = None
) -> str:
    values = values or {}
    parts = [" ".join(question.split()), db_id]
    for table, columns in view.tables:
        cols = []
        for col in columns:
            key = f"{table.lower()}.{col.lower()}"
            text = col.lower()
            if values.get(key):
                text += " ( " + " , ".join(values[key]) + " )"
            cols.append(text)
        parts.append(f"{table.lower()} : " + " , ".join(cols))
    if view.foreign_keys:
        parts.append("fk : " + " , ".join(view.foreign_keys))
    return " | ".join(parts)


def full_view(schema: Schema) -> SchemaView:
    return SchemaView(tables=[(t.name, [c.name for c in t.columns]) for t in schema.tables])


def relevance(question: str, table: str, columns: Iterable[str]) -> int:
    q = _words(question)
    score = len(q & _words(table))
    for col in columns:
        score += len(q & _words(col))
    return score


def serialize_full(
    question: str,
    schema: Schema,
    values: dict[str, list[str]] | None = None,
    *,
    count_tokens: TokenCounter | None = None,
    max_tokens: int = 512,
) -> tuple[str, bool]:
    """Full-schema input; returns ``(text, truncated)``.

    Over budget: re-order tables by relevance, drop values, then drop trailing
    columns of the least relevant tables (removing emptied tables) until it fits.
    """
    view = full_view(schema)
    text = render(question, schema.db_id, view, values)
    if count_tokens is None or count_tokens(text) <= max_tokens:
        return text, False

    view.tables.sort(key=lambda tc: -relevance(question, tc[0], tc[1]))  # stable
    text = render(question, schema.db_id, view, values)
    if count_tokens(text) <= max_tokens:
        return text, True
    text = render(question, schema.db_id, view, None)
    while count_tokens(text) > max_tokens and view.tables:
        table, columns = view.tables[-1]
        if len(columns) > 1:
            view.tables[-1] = (table, columns[:-1])
        else:
            view.tables.pop()
        text = render(question, schema.db_id, view, None)
    return text, True


@dataclass
class RankedSchema:
    """Ranker scores keyed by lower-cased ``table`` and ``table.column``."""

    tables: dict[str, float]
    columns: dict[str, float]


def ranked_view(
    schema: Schema,
    ranked: RankedSchema,
    *,
    top_k_tables: int = 4,
    top_k_columns: int = 5,
    include_foreign_keys: bool = True,
    force_tables: Iterable[str] = (),
    force_columns: dict[str, list[str]] | None = None,
) -> SchemaView:
    """Top-k selection in score order (specs §5.6).

    ``force_*`` (gold items, training only) are always kept at their scored position.
    Key columns linking two kept tables are kept so JOIN conditions stay writable.
    """
    force_columns = force_columns or {}
    t_score = lambda name: ranked.tables.get(name.lower(), 0.0)  # noqa: E731
    c_score = lambda t, c: ranked.columns.get(f"{t.lower()}.{c.lower()}", 0.0)  # noqa: E731

    by_score = sorted(schema.tables, key=lambda t: -t_score(t.name))
    kept = {t.name.lower() for t in by_score[:top_k_tables]}
    kept.update(name.lower() for name in force_tables)
    kept.update(name.lower() for name in force_columns)

    fk_cols: dict[str, set[str]] = {}
    fks: list[str] = []
    for fk in schema.foreign_keys:
        a, b = fk.src.table.lower(), fk.dst.table.lower()
        if a in kept and b in kept and a != b:
            fk_cols.setdefault(a, set()).add(fk.src.name.lower())
            fk_cols.setdefault(b, set()).add(fk.dst.name.lower())
            fks.append(fk.render())

    tables: list[tuple[str, list[str]]] = []
    for table in sorted(
        (t for t in schema.tables if t.name.lower() in kept), key=lambda t: -t_score(t.name)
    ):
        cols = sorted(table.columns, key=lambda c: -c_score(table.name, c.name))
        chosen = {c.name.lower() for c in cols[:top_k_columns]}
        chosen.update(n.lower() for n in force_columns.get(table.name.lower(), []))
        chosen.update(fk_cols.get(table.name.lower(), set()))
        tables.append((table.name, [c.name for c in cols if c.name.lower() in chosen]))
    return SchemaView(tables=tables, foreign_keys=fks if include_foreign_keys else [])


def serialize_ranked(
    question: str,
    schema: Schema,
    ranked: RankedSchema,
    values: dict[str, list[str]] | None = None,
    **kwargs: object,
) -> str:
    return render(question, schema.db_id, ranked_view(schema, ranked, **kwargs), values)  # type: ignore[arg-type]
