"""Ranker inputs: question + schema text pairs with token spans per schema item (specs §5.5).

The schema side reuses the generator's serialization (lower-cased names, matched
values in parentheses) so the ranker sees what the generator sees. Character spans
of every table and column name are mapped to token positions through the fast
tokenizer's offset mapping. Schemas that do not fit in 512 tokens next to the
question are split into chunks of whole tables, each scored separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from spinneret.data.schema import Schema, Table


@dataclass
class Item:
    kind: str  # "table" | "column"
    key: str  # "table" or "table.column" (lower-cased)
    char_start: int
    char_end: int
    table_key: str
    label: int = 0
    tok_start: int = -1
    tok_end: int = -1  # exclusive


@dataclass
class Chunk:
    example_id: str
    input_ids: list[int]
    items: list[Item] = field(default_factory=list)


def _table_text(table: Table, values: dict[str, list[str]], offset: int) -> tuple[str, list[Item]]:
    tkey = table.name.lower()
    items = [Item("table", tkey, offset, offset + len(tkey), tkey)]
    text = f"{tkey} : "
    for i, col in enumerate(table.columns):
        if i:
            text += " , "
        start = offset + len(text)
        name = col.name.lower()
        text += name
        items.append(Item("column", f"{tkey}.{name}", start, start + len(name), tkey))
        if values.get(col.key):
            text += " ( " + " , ".join(values[col.key]) + " )"
    return text, items


def schema_text(tables: list[Table], values: dict[str, list[str]]) -> tuple[str, list[Item]]:
    text, items = "", []
    for i, table in enumerate(tables):
        if i:
            text += " | "
        piece, piece_items = _table_text(table, values, len(text))
        text += piece
        items.extend(piece_items)
    return text, items


def _assign_tokens(items: list[Item], offsets: list[tuple[int, int]], seq_ids: list[Any]) -> None:
    for item in items:
        toks = [
            i
            for i, ((s, e), sid) in enumerate(zip(offsets, seq_ids, strict=True))
            if sid == 1 and e > s and s < item.char_end and e > item.char_start
        ]
        if toks:
            item.tok_start, item.tok_end = toks[0], toks[-1] + 1


def build_chunks(
    tokenizer: Any,
    example_id: str,
    question: str,
    schema: Schema,
    values: dict[str, list[str]],
    *,
    gold_tables: list[str] | None = None,
    gold_columns: dict[str, list[str]] | None = None,
    max_length: int = 512,
) -> list[Chunk]:
    """Pack whole tables into chunks that fit next to the question."""
    gold_t = set(gold_tables or [])
    gold_c = {f"{t}.{c}" for t, cols in (gold_columns or {}).items() for c in cols}
    q_len = len(tokenizer(question, add_special_tokens=False).input_ids)
    budget = max_length - q_len - 4  # <s> q </s></s> schema </s>

    groups: list[list[Table]] = [[]]
    used = 0
    for table in schema.tables:
        piece, _ = _table_text(table, values, 0)
        cost = len(tokenizer(" | " + piece, add_special_tokens=False).input_ids)
        if groups[-1] and used + cost > budget:
            groups.append([])
            used = 0
        groups[-1].append(table)
        used += cost

    chunks = []
    for tables in groups:
        text, items = schema_text(tables, values)
        enc = tokenizer(
            question,
            text,
            truncation="only_second",
            max_length=max_length,
            return_offsets_mapping=True,
        )
        _assign_tokens(items, enc["offset_mapping"], enc.sequence_ids())
        kept = [it for it in items if it.tok_start >= 0]
        for it in kept:
            it.label = int(it.key in (gold_t if it.kind == "table" else gold_c))
        chunks.append(Chunk(example_id=example_id, input_ids=enc["input_ids"], items=kept))
    return chunks
