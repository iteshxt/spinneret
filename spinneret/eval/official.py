"""Structured wrapper around the official Spider evaluation (specs §8.1).

The vendored script (``third_party/test-suite-sql-eval``, Apache-2.0) only prints
its scores. ``OfficialEvaluator.score`` runs the same per-example steps as its
``evaluate()`` loop — including its quirks, e.g. replacing ``value`` with ``1`` in
predictions — and ``aggregate`` reproduces its averaging, so the numbers match the
script exactly (a test checks this).
"""

from __future__ import annotations

import contextlib
import importlib
import io
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from spinneret.db.sqlite import QueryTimeout, connect_readonly, execute

OFFICIAL_DIR = Path(__file__).resolve().parents[2] / "third_party" / "test-suite-sql-eval"
LEVELS = ("easy", "medium", "hard", "extra", "all")
PARTIAL_TYPES = (
    "select", "select(no AGG)", "where", "where(no OP)", "group(no Having)",
    "group", "order", "and/or", "IUEN", "keywords",
)  # fmt: skip
EMPTY_SQL: dict[str, Any] = {
    "except": None,
    "from": {"conds": [], "table_units": []},
    "groupBy": [],
    "having": [],
    "intersect": None,
    "limit": None,
    "orderBy": [],
    "select": [False, []],
    "union": None,
    "where": [],
}

_MODULES: dict[str, ModuleType] = {}


def ensure_nltk() -> None:
    """process_sql tokenizes with nltk.word_tokenize, which needs the punkt tables."""
    import nltk

    for resource in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
            return
        except LookupError:
            continue
    nltk.download("punkt_tab", quiet=True)


def official_modules() -> dict[str, ModuleType]:
    if not _MODULES:
        if not (OFFICIAL_DIR / "evaluation.py").is_file():
            raise FileNotFoundError(f"{OFFICIAL_DIR} is empty — run `git submodule update --init`")
        ensure_nltk()
        sys.path.insert(0, str(OFFICIAL_DIR))
        try:
            for name in ("process_sql", "exec_eval", "evaluation"):
                _MODULES[name] = importlib.import_module(name)
        finally:
            sys.path.remove(str(OFFICIAL_DIR))
    return _MODULES


@dataclass
class ExampleScore:
    hardness: str
    exact: int | None  # None when EM was not computed
    exec: int | None  # None when EX was not computed (or the gold query cannot run)
    partial: dict[str, dict[str, float]] | None = None


class OfficialEvaluator:
    """Score predictions against gold SQL exactly like the official script."""

    def __init__(
        self,
        db_dir: Path,
        tables_path: Path,
        *,
        etype: str = "all",
        plug_value: bool = False,
        keep_distinct: bool = False,
    ) -> None:
        mods = official_modules()
        self._ev = mods["evaluation"]
        self._ps = mods["process_sql"]
        self._ex = mods["exec_eval"]
        self.db_dir = db_dir
        self.etype = etype
        self.plug_value = plug_value
        self.keep_distinct = keep_distinct
        self.kmaps = self._ev.build_foreign_key_map_from_json(str(tables_path))
        self._schemas: dict[str, Any] = {}
        self._evaluator = self._ev.Evaluator()

    def db_path(self, db_id: str) -> Path:
        return self.db_dir / db_id / f"{db_id}.sqlite"

    def _schema(self, db_id: str) -> Any:
        if db_id not in self._schemas:
            self._schemas[db_id] = self._ps.Schema(self._ps.get_schema(str(self.db_path(db_id))))
        return self._schemas[db_id]

    def score(self, pred: str, gold: str, db_id: str) -> ExampleScore:
        p_str = pred.replace("value", "1")  # as the official script does
        schema = self._schema(db_id)
        g_sql = self._ps.get_sql(schema, gold)
        hardness = self._evaluator.eval_hardness(g_sql)
        try:
            p_sql = self._ps.get_sql(schema, p_str)
        except Exception:  # noqa: BLE001 — the official script also swallows everything
            p_sql = EMPTY_SQL

        exec_score: int | None = None
        if self.etype in ("all", "exec"):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exec_score = int(
                        self._ex.eval_exec_match(
                            db=str(self.db_path(db_id)),
                            p_str=p_str,
                            g_str=gold,
                            plug_value=self.plug_value,
                            keep_distinct=self.keep_distinct,
                            progress_bar_for_each_datapoint=False,
                        )
                    )
            except AssertionError:  # the gold query itself fails (a few Spider queries do)
                exec_score = None

        exact: int | None = None
        partial = None
        if self.etype in ("all", "match"):
            kmap = self.kmaps[db_id]
            g_units = self._ev.build_valid_col_units(g_sql["from"]["table_units"], schema)
            g_sql = self._ev.rebuild_sql_col(g_units, self._ev.rebuild_sql_val(g_sql), kmap)
            p_units = self._ev.build_valid_col_units(p_sql["from"]["table_units"], schema)
            p_sql = self._ev.rebuild_sql_col(p_units, self._ev.rebuild_sql_val(p_sql), kmap)
            exact = int(self._evaluator.eval_exact_match(p_sql, g_sql))
            partial = self._evaluator.partial_scores
        return ExampleScore(hardness=hardness, exact=exact, exec=exec_score, partial=partial)


def aggregate(items: Iterable[ExampleScore]) -> dict[str, Any]:
    """Average per-example scores per difficulty level exactly like the official script."""
    scores: dict[str, Any] = {}
    for level in LEVELS:
        scores[level] = {"count": 0, "exact": 0.0, "exec": 0, "exec_count": 0, "partial": {}}
        for t in PARTIAL_TYPES:
            scores[level]["partial"][t] = {
                "acc": 0.0, "rec": 0.0, "f1": 0.0, "acc_count": 0, "rec_count": 0,
            }  # fmt: skip

    for item in items:
        for level in (item.hardness, "all"):
            s = scores[level]
            s["count"] += 1
            if item.exec is not None:
                s["exec"] += item.exec
                s["exec_count"] += 1
            if item.exact is not None:
                s["exact"] += item.exact
            for t, p in (item.partial or {}).items():
                acc = s["partial"][t]
                if p["pred_total"] > 0:
                    acc["acc"] += p["acc"]
                    acc["acc_count"] += 1
                if p["label_total"] > 0:
                    acc["rec"] += p["rec"]
                    acc["rec_count"] += 1
                acc["f1"] += p["f1"]

    for level in LEVELS:
        s = scores[level]
        if s["count"] == 0:
            continue
        s["exact"] /= s["count"]
        s["exec"] = s["exec"] / s["exec_count"] if s["exec_count"] else None
        for t in PARTIAL_TYPES:
            p = s["partial"][t]
            p["acc"] = p["acc"] / p["acc_count"] if p["acc_count"] else 0
            p["rec"] = p["rec"] / p["rec_count"] if p["rec_count"] else 0
            if p["acc"] == 0 and p["rec"] == 0:
                p["f1"] = 1
            else:
                p["f1"] = 2.0 * p["acc"] * p["rec"] / (p["rec"] + p["acc"])
            del p["acc_count"], p["rec_count"]
    return scores


class FastExecMatch:
    """Single-database execution match for validation during training.

    Same post-processing and result comparison as the official ``eval_exec_match``,
    but with our read-only executor, a short timeout, and cached gold results.
    """

    def __init__(self, db_dir: Path, *, timeout_s: float = 5.0) -> None:
        self._ex = official_modules()["exec_eval"]
        self.db_dir = db_dir
        self.timeout_s = timeout_s
        self._conns: dict[str, Any] = {}
        self._gold: dict[tuple[str, str], Any] = {}

    def _run(self, db_id: str, sql: str) -> Any:
        if db_id not in self._conns:
            self._conns[db_id] = connect_readonly(self.db_dir / db_id / f"{db_id}.sqlite")
        _, rows = execute(self._conns[db_id], sql, timeout_s=self.timeout_s)
        return rows

    def __call__(self, pred: str, gold: str, db_id: str) -> int | None:
        p_str = self._ex.remove_distinct(self._ex.postprocess(pred))
        g_str = self._ex.remove_distinct(self._ex.postprocess(gold))
        key = (db_id, g_str)
        if key not in self._gold:
            try:
                self._gold[key] = self._run(db_id, g_str)
            except Exception:  # noqa: BLE001
                self._gold[key] = None
        gold_rows = self._gold[key]
        if gold_rows is None:
            return None
        try:
            pred_rows = self._run(db_id, p_str)
        except (QueryTimeout, Exception):  # noqa: BLE001
            return 0
        order_matters = "order by" in g_str.lower()
        return int(self._ex.result_eq(gold_rows, pred_rows, order_matters=order_matters))

    def close(self) -> None:
        for conn in self._conns.values():
            conn.close()
        self._conns.clear()
