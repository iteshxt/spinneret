"""Question types and the failure taxonomy for a predictions run (specs §8.4–8.5).

Both work on SQL parsed by Spider's own ``process_sql.get_sql`` (the same parser the
official EM uses), so "same columns" or "same tables" means what EM means by it.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from spinneret.data.hardness import LEVELS
from spinneret.data.prepare import read_jsonl
from spinneret.eval.official import official_modules

SQL = dict[str, Any]
SET_OPS = ("intersect", "union", "except")
CATEGORIES = (
    "invalid_sql", "wrong_table", "join_error", "nesting_setop_error", "wrong_column",
    "aggregation_error", "group_by_error", "condition_error", "order_limit_error",
    "value_error", "other",
)  # fmt: skip
RANKER_CATEGORIES = ("wrong_table", "wrong_column", "join_error")
QUESTION_TYPES = ("simple lookup", "aggregation", "two-table join", "multi-join", "nested / set-op")


# --- parsed-SQL helpers ---------------------------------------------------------------


def _col_units(val_unit: Any) -> list[Any]:
    """Column units of a val_unit ``(unit_op, col_unit1, col_unit2)``."""
    return [cu for cu in val_unit[1:] if cu is not None]


def _conds(conds: list[Any]) -> list[Any]:
    return conds[::2]


def _connectors(conds: list[Any]) -> list[str]:
    return conds[1::2]


def _subqueries(sql: SQL) -> list[SQL]:
    """Nested queries inside FROM / WHERE / HAVING (not set operations)."""
    out = [tu[1] for tu in sql["from"]["table_units"] if tu[0] == "sql"]
    for cond in _conds(sql["where"]) + _conds(sql["having"]):
        out.extend(v for v in cond[3:5] if isinstance(v, dict))
    return out


def tables(sql: SQL) -> set[str]:
    return {tu[1] for tu in sql["from"]["table_units"] if tu[0] == "table_unit"}


def all_tables(sql: SQL) -> set[str]:
    """Tables anywhere in the query, nested queries and set operations included."""
    found = set(tables(sql))
    for sub in _subqueries(sql) + [sql[k] for k in SET_OPS if sql[k]]:
        found |= all_tables(sub)
    return found


def join_conds(sql: SQL) -> set[frozenset[str]]:
    pairs = set()
    for cond in _conds(sql["from"]["conds"]):
        left = cond[2][1][1]
        right = cond[3][1] if isinstance(cond[3], tuple | list) else str(cond[3])
        pairs.add(frozenset((left, right)))
    return pairs


def select_columns(sql: SQL) -> set[str]:
    return {cu[1] for _, vu in sql["select"][1] for cu in _col_units(vu)}


def where_columns(sql: SQL) -> set[str]:
    return {cu[1] for cond in _conds(sql["where"]) for cu in _col_units(cond[2])}


def select_aggs(sql: SQL) -> Counter[int]:
    aggs: Counter[int] = Counter()
    for agg, vu in sql["select"][1]:
        aggs[agg] += 1
        for cu in _col_units(vu):
            aggs[cu[0]] += 1
    aggs.pop(0, None)
    return aggs


def having_signature(sql: SQL) -> set[tuple[Any, ...]]:
    return {
        (cond[0], cond[1], tuple(cu[:2] for cu in _col_units(cond[2])))
        for cond in _conds(sql["having"])
    }


def where_ops(sql: SQL) -> tuple[Counter[tuple[bool, int]], list[str]]:
    return Counter((c[0], c[1]) for c in _conds(sql["where"])), _connectors(sql["where"])


def order_signature(sql: SQL) -> tuple[Any, ...]:
    if not sql["orderBy"]:
        return (None, frozenset(), sql["limit"] is not None)
    direction, units = sql["orderBy"]
    cols = frozenset((cu[0], cu[1]) for vu in units for cu in _col_units(vu))
    return (direction, cols, sql["limit"] is not None)


def nesting_signature(sql: SQL) -> tuple[Any, ...]:
    return tuple(sql[k] is not None for k in SET_OPS) + (len(_subqueries(sql)),)


def has_aggregate(sql: SQL) -> bool:
    if select_aggs(sql):
        return True
    if any(cu[0] for cond in _conds(sql["having"]) for cu in _col_units(cond[2])):
        return True
    if sql["orderBy"]:
        return any(cu[0] for vu in sql["orderBy"][1] for cu in _col_units(vu))
    return False


def is_nested(sql: SQL) -> bool:
    return any(sql[k] for k in SET_OPS) or bool(_subqueries(sql))


# --- question types (specs §8.5) --------------------------------------------------------


def question_features(gold: SQL) -> dict[str, Any]:
    n = len(all_tables(gold))
    return {
        "tables": "1" if n <= 1 else ("2" if n == 2 else "3+"),
        "aggregate": has_aggregate(gold),
        "group_by": bool(gold["groupBy"]),
        "order_limit": bool(gold["orderBy"]) or gold["limit"] is not None,
        "nested": is_nested(gold),
    }


def question_type(f: dict[str, Any]) -> str:
    if f["nested"]:
        return "nested / set-op"
    if f["tables"] == "3+":
        return "multi-join"
    if f["tables"] == "2":
        return "two-table join"
    if f["aggregate"] or f["group_by"]:
        return "aggregation"
    return "simple lookup"


# --- failure taxonomy (specs §8.4) ------------------------------------------------------


def compare(pred: SQL, gold: SQL) -> list[str]:
    """Structural differences between two parsed queries, in taxonomy order."""
    labels = []
    pt, gt = tables(pred), tables(gold)
    if len(pt) == len(gt) and pt != gt:
        labels.append("wrong_table")
    if len(pt) != len(gt) or (pt == gt and join_conds(pred) != join_conds(gold)):
        labels.append("join_error")
    nested_same = nesting_signature(pred) == nesting_signature(gold)
    if not nested_same:
        labels.append("nesting_setop_error")
    if select_columns(pred) != select_columns(gold) or where_columns(pred) != where_columns(gold):
        labels.append("wrong_column")
    if select_aggs(pred) != select_aggs(gold):
        labels.append("aggregation_error")
    if {cu[1] for cu in pred["groupBy"]} != {cu[1] for cu in gold["groupBy"]} or (
        having_signature(pred) != having_signature(gold)
    ):
        labels.append("group_by_error")
    if where_columns(pred) == where_columns(gold) and where_ops(pred) != where_ops(gold):
        labels.append("condition_error")
    if order_signature(pred) != order_signature(gold):
        labels.append("order_limit_error")
    if nested_same:  # same shape: look inside the nested parts too
        pairs = list(zip(_subqueries(pred), _subqueries(gold), strict=True))
        pairs += [(pred[k], gold[k]) for k in SET_OPS if pred[k] and gold[k]]
        for p_sub, g_sub in pairs:
            labels.extend(lab for lab in compare(p_sub, g_sub) if lab not in labels)
    return [c for c in CATEGORIES if c in labels]


class Parser:
    """Parse SQL with the official process_sql, one cached schema per database."""

    def __init__(self, spider_dir: Path) -> None:
        self._ps = official_modules()["process_sql"]
        self.db_dir = spider_dir / "database"
        self._schemas: dict[str, Any] = {}

    def __call__(self, sql: str, db_id: str) -> SQL | None:
        if db_id not in self._schemas:
            path = self.db_dir / db_id / f"{db_id}.sqlite"
            self._schemas[db_id] = self._ps.Schema(self._ps.get_schema(str(path)))
        try:
            return self._ps.get_sql(self._schemas[db_id], sql.replace("value", "1"))
        except Exception:  # noqa: BLE001 — the official parser raises anything on bad SQL
            return None


def label(
    record: dict[str, Any], parse: Parser, runs: Callable[[str, str], bool]
) -> dict[str, Any] | None:
    """Taxonomy row for one scored prediction, or None if it is fully correct."""
    em, ex = record.get("em"), record.get("ex")
    if ex is None or (em == 1 and ex == 1):
        return None
    row = {"id": record["id"], "db_id": record["db_id"], "hardness": record["hardness"],
           "question": record["question"], "gold_sql": record["gold_sql"],
           "pred_sql": record["pred_sql"]}  # fmt: skip
    if ex == 1:
        return {**row, "primary": "semantically_equivalent", "labels": ["semantically_equivalent"]}
    gold = parse(record["gold_sql"], record["db_id"])
    pred = parse(record["pred_sql"], record["db_id"])
    if pred is None or not runs(record["db_id"], record["pred_sql"]):
        labels = ["invalid_sql"]
    elif gold is None:
        labels = ["other"]
    elif em == 1:
        # EM ignores which columns a join uses (it normalizes foreign keys), so look anyway
        labels = compare(pred, gold) or ["value_error"]
    else:
        labels = compare(pred, gold) or ["other"]
    return {**row, "primary": labels[0], "labels": labels}


def taxonomy(
    records: list[dict[str, Any]],
    parse: Parser,
    runs: Callable[[str, str], bool],
    ranked: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for r in records:
        row = label(r, parse, runs)
        if row is None:
            continue
        miss = False
        if ranked is not None and r["id"] in ranked:
            miss = not ranked[r["id"]]["ranker_covered"] and any(
                lab in RANKER_CATEGORIES for lab in row["labels"]
            )
        row["ranker_miss"] = miss
        rows.append(row)
    return rows


def summarize_taxonomy(rows: list[dict[str, Any]], total: int, seed: int = 0) -> dict[str, Any]:
    errors = [r for r in rows if r["primary"] != "semantically_equivalent"]
    rng = random.Random(seed)
    by_primary: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_primary[r["primary"]].append(r)
    cats = CATEGORIES + ("semantically_equivalent",)
    summary: dict[str, Any] = {
        "examples": total,
        "ex_wrong": len(errors),
        "primary": {c: len(by_primary[c]) for c in cats},
        "any_label": {c: sum(c in r["labels"] for r in rows) for c in cats},
        "primary_by_hardness": {
            lvl: {c: sum(r["hardness"] == lvl for r in by_primary[c]) for c in cats}
            for lvl in LEVELS
        },
        "ranker_miss": sum(r["ranker_miss"] for r in errors),
        "examples_per_category": {},
    }
    for c in cats:
        picks = rng.sample(by_primary[c], min(2, len(by_primary[c])))
        summary["examples_per_category"][c] = [
            {k: p[k] for k in ("id", "question", "gold_sql", "pred_sql")} for p in picks
        ]
    return summary


def write_taxonomy(run_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    with (run_dir / "taxonomy.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "db_id", "hardness", "primary", "labels", "ranker_miss",
                    "question", "gold_sql", "pred_sql"])  # fmt: skip
        for r in rows:
            keys = ("id", "db_id", "hardness", "primary")
            tail = (r["question"], r["gold_sql"], r["pred_sql"])
            w.writerow([*(r[k] for k in keys), ";".join(r["labels"]), int(r["ranker_miss"]), *tail])
    (run_dir / "taxonomy_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def question_type_table(
    records: list[dict[str, Any]], features: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """EX per named question type and per feature bucket."""

    def rate(items: list[dict[str, Any]]) -> dict[str, Any]:
        scored = [r["ex"] for r in items if r.get("ex") is not None]
        return {"count": len(scored), "ex": sum(scored) / len(scored) if scored else None}

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        f = features[r["id"]]
        by_type[question_type(f)].append(r)
        buckets[f"tables={f['tables']}"].append(r)
        for key in ("aggregate", "group_by", "order_limit", "nested"):
            buckets[f"{key}={'yes' if f[key] else 'no'}"].append(r)
    return {
        "types": {t: rate(by_type[t]) for t in QUESTION_TYPES},
        "buckets": {b: rate(items) for b, items in sorted(buckets.items())},
    }


def manual_review_sample(
    rows: list[dict[str, Any]], n: int = 100, seed: int = 0
) -> list[dict[str, Any]]:
    """Seeded random sample of EX-wrong predictions for hand labelling (specs §8.4)."""
    errors = [r for r in rows if r["primary"] != "semantically_equivalent"]
    picks = random.Random(seed).sample(errors, min(n, len(errors)))
    return sorted(picks, key=lambda r: r["id"])


def load_records(run_dir: Path) -> list[dict[str, Any]]:
    records = read_jsonl(run_dir / "predictions.jsonl")
    if not records or "ex" not in records[0]:
        raise ValueError(f"{run_dir} is not scored yet — run `spinneret evaluate --run {run_dir}`")
    return records


def ranked_inputs_for(run_dir: Path) -> dict[str, dict[str, Any]] | None:
    """Ranked dev records (with ``ranker_covered``) when the run's model used the ranker."""
    from spinneret.config import load_config
    from spinneret.hub import resolve_checkpoint

    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    cfg = load_config(resolve_checkpoint(meta["checkpoint"]) / "config.yaml")
    if cfg.data.schema_input != "ranked":
        return None
    inputs = Path(meta.get("inputs_dir") or cfg.data.inputs_dir())
    return {r["id"]: r for r in read_jsonl(inputs / f"{meta['split']}.jsonl")}


def analyze_run(run_dir: Path, spider_dir: Path, *, review: int = 100) -> dict[str, Any]:
    """Write taxonomy.csv, taxonomy_summary.json, question_types.json (+ manual_review.csv)."""
    from spinneret.data.spider import database_path
    from spinneret.db.executor import Executor

    records = load_records(run_dir)
    executor = Executor(lambda db_id: database_path(spider_dir, db_id), timeout_s=5.0)
    parse = Parser(spider_dir)
    rows = taxonomy(
        records,
        parse,
        lambda db, sql: executor.run(db, sql, max_rows=1).ok,
        ranked_inputs_for(run_dir),
    )
    summary = summarize_taxonomy(rows, len(records))
    write_taxonomy(run_dir, rows, summary)

    features = {}
    for r in records:
        gold = parse(r["gold_sql"], r["db_id"])
        if gold is not None:
            features[r["id"]] = question_features(gold)
    types = question_type_table([r for r in records if r["id"] in features], features)
    (run_dir / "question_types.json").write_text(json.dumps(types, indent=2), encoding="utf-8")

    if review:
        path = run_dir / "manual_review.csv"
        if path.is_file():  # never overwrite hand labels
            path = run_dir / "manual_review.new.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "hardness", "question", "gold_sql", "pred_sql", "auto_primary",
                        "auto_labels", "manual_primary", "notes"])  # fmt: skip
            for r in manual_review_sample(rows, review):
                w.writerow([r["id"], r["hardness"], r["question"], r["gold_sql"], r["pred_sql"],
                            r["primary"], ";".join(r["labels"]), "", ""])  # fmt: skip
    return {"taxonomy": summary, "question_types": types}


def review_agreement(path: Path, column: str = "manual_primary") -> dict[str, Any]:
    """Agreement between hand labels (``column``) and automatic primary labels."""
    with path.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if (r.get(column) or "").strip()]
    agree = sum(r[column].strip() == r["auto_primary"] for r in rows)
    confusion = Counter(
        (r["auto_primary"], r[column].strip())
        for r in rows
        if r[column].strip() != r["auto_primary"]
    )
    return {
        "labelled": len(rows),
        "agreement": agree / len(rows) if rows else None,
        "disagreements": {f"{a} -> {m}": n for (a, m), n in confusion.most_common()},
    }
