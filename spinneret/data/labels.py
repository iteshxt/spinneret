"""Gold schema items (tables and columns) used by a Spider query.

Reads the parsed ``sql`` dict that ships with every Spider example, so the gold
SQL never has to be re-parsed. Structure (lists in JSON, tuples in the original
parser):

- ``select``: ``[is_distinct, [[agg_id, val_unit], ...]]``
- ``from``: ``{"table_units": [[type, table_index | sql], ...], "conds": condition}``
- ``where`` / ``having``: condition = ``[cond_unit, "and"|"or", cond_unit, ...]``
- cond_unit: ``[not_op, op_id, val_unit, val1, val2]`` — val may be a sub-query dict,
  a col_unit list, or a literal
- ``groupBy``: ``[col_unit, ...]``; ``orderBy``: ``[]`` or ``[direction, [val_unit, ...]]``
- val_unit: ``[unit_op, col_unit, col_unit | None]``; col_unit: ``[agg_id, col_index, distinct]``
- ``intersect`` / ``union`` / ``except``: nested sql or ``None``
"""

from __future__ import annotations

from typing import Any

from spinneret.data.schema import Schema


def _walk(sql: dict[str, Any], tables: set[int], columns: set[int]) -> None:
    for table_type, item in sql["from"]["table_units"]:
        if table_type == "table_unit":
            tables.add(item)
        elif isinstance(item, dict):
            _walk(item, tables, columns)

    def col_unit(unit: Any) -> None:
        if isinstance(unit, list) and len(unit) == 3 and isinstance(unit[1], int):
            columns.add(unit[1])

    def val_unit(unit: Any) -> None:
        if isinstance(unit, list) and len(unit) == 3:
            col_unit(unit[1])
            col_unit(unit[2])

    def value(val: Any) -> None:
        if isinstance(val, dict):
            _walk(val, tables, columns)
        elif isinstance(val, list):
            col_unit(val)

    def condition(cond: list[Any]) -> None:
        for unit in cond[::2]:
            val_unit(unit[2])
            value(unit[3])
            value(unit[4])

    for _agg, unit in sql["select"][1]:
        val_unit(unit)
    condition(sql["from"]["conds"])
    condition(sql["where"])
    condition(sql["having"])
    for unit in sql["groupBy"]:
        col_unit(unit)
    if sql["orderBy"]:
        for unit in sql["orderBy"][1]:
            val_unit(unit)
    for key in ("intersect", "union", "except"):
        if sql.get(key):
            _walk(sql[key], tables, columns)


def gold_items(sql: dict[str, Any], schema: Schema) -> tuple[list[str], dict[str, list[str]]]:
    """Return ``(tables, {table: [columns]})`` used by the query, lower-cased, in schema order.

    ``*`` is ignored. A column's table is always included among the tables.
    """
    table_ids: set[int] = set()
    column_ids: set[int] = set()
    _walk(sql, table_ids, column_ids)

    used_columns = [schema.column(i) for i in sorted(column_ids)]
    table_names = {schema.tables[i].name.lower() for i in table_ids if 0 <= i < len(schema.tables)}
    table_names.update(c.table.lower() for c in used_columns if c is not None)

    tables = [t.name.lower() for t in schema.tables if t.name.lower() in table_names]
    columns: dict[str, list[str]] = {}
    for table in schema.tables:
        names = [c.name.lower() for c in table.columns if c.index in column_ids]
        if names:
            columns[table.name.lower()] = names
    return tables, columns
