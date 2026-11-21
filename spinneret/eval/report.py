"""Score a predictions run with the official metrics and write metrics.json (specs §8)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from spinneret.data.prepare import read_jsonl
from spinneret.eval.official import LEVELS, OfficialEvaluator, aggregate


def evaluate_run(
    run_dir: Path, spider_dir: Path, *, etype: str = "all", progress: bool = True
) -> dict[str, Any]:
    records = read_jsonl(run_dir / "predictions.jsonl")
    evaluator = OfficialEvaluator(spider_dir / "database", spider_dir / "tables.json", etype=etype)
    scores = []
    for r in tqdm(records, desc="evaluate", disable=not progress):
        s = evaluator.score(r["pred_sql"], r["gold_sql"], r["db_id"])
        r["em"], r["ex"] = s.exact, s.exec
        scores.append(s)

    official = aggregate(scores)
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    executed = [r["executed"] for r in records if r.get("executed") is not None]
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    metrics = {
        "run_id": meta.get("run_id", run_dir.name),
        "meta": meta,
        "count": official["all"]["count"],
        "exact_match": {lvl: official[lvl]["exact"] for lvl in LEVELS},
        "execution": {lvl: official[lvl]["exec"] for lvl in LEVELS},
        "counts": {lvl: official[lvl]["count"] for lvl in LEVELS},
        "partial_f1": {t: v["f1"] for t, v in official["all"]["partial"].items()},
        "valid_sql_rate": sum(executed) / len(executed) if executed else None,
        "official": official,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def format_metrics(metrics: dict[str, Any]) -> str:
    rows = [f"{'':10s}" + "".join(f"{lvl:>9s}" for lvl in LEVELS)]
    rows.append(f"{'count':10s}" + "".join(f"{metrics['counts'][lvl]:>9d}" for lvl in LEVELS))
    for name, key in (("EM", "exact_match"), ("EX", "execution")):
        vals = metrics[key]
        rows.append(f"{name:10s}" + "".join(f"{(vals[lvl] or 0) * 100:>8.1f}%" for lvl in LEVELS))
    if metrics.get("valid_sql_rate") is not None:
        rows.append(f"valid SQL: {metrics['valid_sql_rate'] * 100:.1f}%")
    return "\n".join(rows)
