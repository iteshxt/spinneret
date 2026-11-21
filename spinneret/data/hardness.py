"""Spider difficulty levels (easy / medium / hard / extra).

A faithful port of ``Evaluator.eval_hardness`` and its helpers from the official
evaluation script (taoyds/spider, taoyds/test-suite-sql-eval; Apache-2.0),
including its quirks (e.g. ``count_agg`` over WHERE/HAVING condition lists). A test
checks this port against the vendored official script on every Spider example.
"""

from __future__ import annotations

from typing import Any

WHERE_OPS = ("not", "between", "=", ">", "<", ">=", "<=", "!=", "in", "like", "is", "exists")
LIKE_OP = WHERE_OPS.index("like")
AGG_NONE = 0

LEVELS = ("easy", "medium", "hard", "extra")


def _has_agg(unit: Any) -> bool:
    return unit[0] != AGG_NONE


def _count_agg(units: list[Any]) -> int:
    return sum(1 for unit in units if _has_agg(unit))


def _nested(sql: dict[str, Any]) -> list[dict[str, Any]]:
    nested = []
    for cond_unit in sql["from"]["conds"][::2] + sql["where"][::2] + sql["having"][::2]:
        if isinstance(cond_unit[3], dict):
            nested.append(cond_unit[3])
        if isinstance(cond_unit[4], dict):
            nested.append(cond_unit[4])
    for key in ("intersect", "except", "union"):
        if sql[key] is not None:
            nested.append(sql[key])
    return nested


def count_component1(sql: dict[str, Any]) -> int:
    count = 0
    if len(sql["where"]) > 0:
        count += 1
    if len(sql["groupBy"]) > 0:
        count += 1
    if len(sql["orderBy"]) > 0:
        count += 1
    if sql["limit"] is not None:
        count += 1
    if len(sql["from"]["table_units"]) > 0:  # JOIN
        count += len(sql["from"]["table_units"]) - 1

    connectors = sql["from"]["conds"][1::2] + sql["where"][1::2] + sql["having"][1::2]
    count += sum(1 for token in connectors if token == "or")
    cond_units = sql["from"]["conds"][::2] + sql["where"][::2] + sql["having"][::2]
    count += sum(1 for cond_unit in cond_units if cond_unit[1] == LIKE_OP)
    return count


def count_component2(sql: dict[str, Any]) -> int:
    return len(_nested(sql))


def count_others(sql: dict[str, Any]) -> int:
    count = 0
    agg_count = _count_agg(sql["select"][1])
    agg_count += _count_agg(sql["where"][::2])
    agg_count += _count_agg(sql["groupBy"])
    if len(sql["orderBy"]) > 0:
        order_units = sql["orderBy"][1]
        agg_count += _count_agg(
            [unit[1] for unit in order_units if unit[1]]
            + [unit[2] for unit in order_units if unit[2]]
        )
    agg_count += _count_agg(sql["having"])
    if agg_count > 1:
        count += 1
    if len(sql["select"][1]) > 1:
        count += 1
    if len(sql["where"]) > 1:
        count += 1
    if len(sql["groupBy"]) > 1:
        count += 1
    return count


def hardness(sql: dict[str, Any]) -> str:
    c1 = count_component1(sql)
    c2 = count_component2(sql)
    others = count_others(sql)
    if c1 <= 1 and others == 0 and c2 == 0:
        return "easy"
    if (others <= 2 and c1 <= 1 and c2 == 0) or (c1 <= 2 and others < 2 and c2 == 0):
        return "medium"
    if (
        (others > 2 and c1 <= 2 and c2 == 0)
        or (2 < c1 <= 3 and others <= 2 and c2 == 0)
        or (c1 <= 1 and others == 0 and c2 <= 1)
    ):
        return "hard"
    return "extra"
