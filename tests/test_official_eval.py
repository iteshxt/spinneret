from __future__ import annotations

import contextlib
import io
import json
import re

import pytest

from spinneret.data.normalize import normalize_sql
from spinneret.data.spider import load_examples
from spinneret.eval.official import FastExecMatch, OfficialEvaluator, aggregate, official_modules


def _perturb(i: int, query: str) -> str:
    """Make a mix of right, wrong and broken predictions."""
    if i % 4 == 1:
        return normalize_sql(query)  # right, different surface form
    if i % 4 == 2:
        return query.replace("SELECT", "SELECT DISTINCT", 1).replace(" DESC", " ASC")
    if i % 4 == 3:
        return "SELECT nonsense FROM nowhere"
    return query


def _parse_official(stdout: str, row: str) -> list[float]:
    for line in stdout.splitlines():
        if line.startswith(row):
            return [float(x) for x in re.findall(r"\d+\.\d+", line[20:])]
    raise AssertionError(f"row {row!r} not in official output")


@pytest.mark.data
def test_wrapper_matches_official_script(spider_dir, tmp_path):
    """Our per-example wrapper + aggregate must print the same numbers as evaluate()."""
    dev = load_examples(spider_dir, "dev")[:60]
    preds = [_perturb(i, ex["query"]) for i, ex in enumerate(dev)]
    gold_file, pred_file = tmp_path / "gold.sql", tmp_path / "pred.sql"
    gold_file.write_text("".join(f"{ex['query']}\t{ex['db_id']}\n" for ex in dev))
    pred_file.write_text("".join(p + "\n" for p in preds))

    ev = official_modules()["evaluation"]
    kmaps = ev.build_foreign_key_map_from_json(str(spider_dir / "tables.json"))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ev.evaluate(str(gold_file), str(pred_file), str(spider_dir / "database"), "all", kmaps,
                    False, False, False)  # fmt: skip
    official_exec = _parse_official(out.getvalue(), "execution")
    official_exact = _parse_official(out.getvalue(), "exact match")

    evaluator = OfficialEvaluator(spider_dir / "database", spider_dir / "tables.json")
    ours = aggregate(
        evaluator.score(p, ex["query"], ex["db_id"]) for p, ex in zip(preds, dev, strict=True)
    )
    levels = ["easy", "medium", "hard", "extra", "all"]
    present = [lvl for lvl in levels if ours[lvl]["count"]]
    assert [round(ours[lvl]["exec"], 3) for lvl in levels if lvl in present] == [
        v for lvl, v in zip(levels, official_exec, strict=False) if lvl in present
    ]
    assert [round(ours[lvl]["exact"], 3) for lvl in levels if lvl in present] == [
        v for lvl, v in zip(levels, official_exact, strict=False) if lvl in present
    ]
    assert 0 < ours["all"]["exact"] < 1


@pytest.mark.data
def test_fast_exec_match_agrees_with_official(spider_dir):
    dev = load_examples(spider_dir, "dev")[:40]
    official = OfficialEvaluator(spider_dir / "database", spider_dir / "tables.json", etype="exec")
    fast = FastExecMatch(spider_dir / "database")
    for i, ex in enumerate(dev):
        pred = _perturb(i, ex["query"])
        assert (
            fast(pred, ex["query"], ex["db_id"])
            == official.score(pred, ex["query"], ex["db_id"]).exec
        ), pred
    fast.close()


@pytest.mark.data
@pytest.mark.slow
def test_normalized_gold_scores_100_percent_on_dev(spider_dir):
    """Gold vs normalized gold on all of dev: 100% EM and EX (sanity for both pieces)."""
    dev = load_examples(spider_dir, "dev")
    evaluator = OfficialEvaluator(spider_dir / "database", spider_dir / "tables.json")
    scores = aggregate(
        evaluator.score(normalize_sql(ex["query"]), ex["query"], ex["db_id"]) for ex in dev
    )
    print(json.dumps({k: (v["count"], v["exact"], v["exec"]) for k, v in scores.items()}))
    assert scores["all"]["count"] == 1034
    assert scores["all"]["exact"] == 1.0
    assert scores["all"]["exec"] == 1.0
