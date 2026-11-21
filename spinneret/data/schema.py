"""Typed view of one Spider database schema (an entry of ``tables.json``).

Spinneret always works with the ``*_original`` names because those are what SQL
refers to. Column indices follow ``column_names_original``: index 0 is ``*``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Column:
    index: int  # index into column_names_original (0 is "*")
    table: str  # original table name
    name: str  # original column name
    type: str  # Spider column type: text | number | time | boolean | others
    is_pk: bool = False

    @property
    def key(self) -> str:
        """Lower-cased ``table.column`` identifier used across the pipeline."""
        return f"{self.table.lower()}.{self.name.lower()}"


@dataclass(frozen=True)
class Table:
    index: int
    name: str
    columns: tuple[Column, ...]


@dataclass(frozen=True)
class ForeignKey:
    src: Column
    dst: Column

    def render(self) -> str:
        return f"{self.src.key} = {self.dst.key}"


@dataclass(frozen=True)
class Schema:
    db_id: str
    tables: tuple[Table, ...]
    foreign_keys: tuple[ForeignKey, ...]
    _by_index: dict[int, Column] = field(default_factory=dict, repr=False, compare=False)

    def column(self, index: int) -> Column | None:
        """Column by ``column_names_original`` index; ``None`` for ``*`` (index 0)."""
        return self._by_index.get(index)

    def table(self, name: str) -> Table:
        lowered = name.lower()
        for table in self.tables:
            if table.name.lower() == lowered:
                return table
        raise KeyError(f"{self.db_id}: no table named {name!r}")

    @property
    def n_columns(self) -> int:
        return sum(len(t.columns) for t in self.tables)

    @classmethod
    def from_spider(cls, entry: dict[str, Any]) -> Schema:
        table_names = entry["table_names_original"]
        pk_indices: set[int] = set()
        for pk in entry.get("primary_keys", []):
            # Newer tables.json files list composite keys as nested lists.
            if isinstance(pk, list):
                pk_indices.update(pk)
            else:
                pk_indices.add(pk)

        per_table: dict[int, list[Column]] = {i: [] for i in range(len(table_names))}
        by_index: dict[int, Column] = {}
        for idx, ((t_idx, col_name), col_type) in enumerate(
            zip(entry["column_names_original"], entry["column_types"], strict=True)
        ):
            if t_idx < 0:  # the "*" pseudo-column
                continue
            col = Column(
                index=idx,
                table=table_names[t_idx],
                name=col_name,
                type=col_type,
                is_pk=idx in pk_indices,
            )
            per_table[t_idx].append(col)
            by_index[idx] = col

        tables = tuple(
            Table(index=i, name=name, columns=tuple(per_table[i]))
            for i, name in enumerate(table_names)
        )
        fks = tuple(
            ForeignKey(src=by_index[a], dst=by_index[b])
            for a, b in entry.get("foreign_keys", [])
            if a in by_index and b in by_index
        )
        return cls(db_id=entry["db_id"], tables=tables, foreign_keys=fks, _by_index=by_index)
