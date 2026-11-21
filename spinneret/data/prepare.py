"""Build processed train / valid / dev JSONL files and dataset statistics (specs §4)."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tqdm import tqdm

from spinneret.config import ExperimentConfig
from spinneret.data.hardness import LEVELS, hardness
from spinneret.data.labels import gold_items
from spinneret.data.normalize import normalize_sql
from spinneret.data.schema import Schema
from spinneret.data.serialize import serialize_full
from spinneret.data.skeleton import skeleton_target, sql_skeleton
from spinneret.data.spider import database_path, load_examples, load_schemas, missing_files
from spinneret.data.split import validation_databases
from spinneret.data.values import ValueMatcher

TokenCounter = Callable[[str], int]


def build_record(
    ex: dict[str, Any],
    *,
    record_id: str,
    split: str,
    source_file: str,
    schema: Schema,
    matcher: ValueMatcher | None,
    count_tokens: TokenCounter | None,
    max_source_len: int,
) -> dict[str, Any]:
    question = " ".join(ex["question"].split())
    target = normalize_sql(ex["query"])
    tables, columns = gold_items(ex["sql"], schema)
    matched = matcher.match(question, schema.db_id) if matcher else {}
    source, truncated = serialize_full(
        question, schema, matched, count_tokens=count_tokens, max_tokens=max_source_len
    )
    record: dict[str, Any] = {
        "id": record_id,
        "split": split,
        "source_file": source_file,
        "db_id": schema.db_id,
        "question": question,
        "source": source,
        "target": target,
        "skeleton": sql_skeleton(target),
        "gold_sql": ex["query"],
        "hardness": hardness(ex["sql"]),
        "gold_tables": tables,
        "gold_columns": columns,
        "n_tables": len(tables),
        "truncated": truncated,
        "matched_values": matched,
    }
    if count_tokens is not None:
        record["source_tokens"] = count_tokens(source)
        record["target_tokens"] = count_tokens(target)
        record["skeleton_target_tokens"] = count_tokens(skeleton_target(target))
    return record


def _percentiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    qs = statistics.quantiles(values, n=100, method="inclusive")
    return {
        "mean": round(statistics.fmean(values), 1),
        "p50": qs[49],
        "p90": qs[89],
        "p99": qs[98],
        "max": max(values),
    }


def split_stats(records: list[dict[str, Any]], max_target_len: int) -> dict[str, Any]:
    n = len(records)
    stats: dict[str, Any] = {
        "questions": n,
        "databases": len({r["db_id"] for r in records}),
        "by_source_file": dict(Counter(r["source_file"] for r in records)),
        "hardness": {level: sum(r["hardness"] == level for r in records) for level in LEVELS},
        "truncated": sum(r["truncated"] for r in records),
        "with_matched_values": sum(bool(r["matched_values"]) for r in records),
        "tables_per_query": dict(sorted(Counter(r["n_tables"] for r in records).items())),
    }
    if records and "source_tokens" in records[0]:
        for key in ("source_tokens", "target_tokens", "skeleton_target_tokens"):
            stats[key] = _percentiles([r[key] for r in records])
        stats["targets_over_max"] = sum(r["target_tokens"] > max_target_len for r in records)
        stats["skeleton_targets_over_max"] = sum(
            r["skeleton_target_tokens"] > max_target_len for r in records
        )
    return stats


def prepare(
    cfg: ExperimentConfig,
    *,
    count_tokens: TokenCounter | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    data = cfg.data
    spider_dir = data.spider_dir
    missing = missing_files(spider_dir)
    if missing:
        raise FileNotFoundError(
            f"Spider not found in {spider_dir} (missing {', '.join(missing)}). "
            "Run `spinneret download` first."
        )

    schemas = load_schemas(spider_dir)
    matcher = (
        ValueMatcher(
            lambda db_id: database_path(spider_dir, db_id),
            schemas,
            data.cache_dir / "values",
            max_per_column=data.max_values_per_column,
        )
        if data.db_content
        else None
    )

    train_spider = load_examples(spider_dir, "train_spider")
    valid_dbs = validation_databases(train_spider, data.validation_fraction, cfg.seed)

    plan: dict[str, list[tuple[str, dict[str, Any]]]] = {"train": [], "valid": [], "dev": []}
    for ex in train_spider:
        plan["valid" if ex["db_id"] in valid_dbs else "train"].append(("train_spider.json", ex))
    if data.include_train_others:
        others = load_examples(spider_dir, "train_others")
        plan["train"].extend(("train_others.json", ex) for ex in others)
    plan["dev"] = [("dev.json", ex) for ex in load_examples(spider_dir, "dev")]

    data.processed_dir.mkdir(parents=True, exist_ok=True)
    all_stats: dict[str, Any] = {
        "config": {
            "seed": cfg.seed,
            "validation_fraction": data.validation_fraction,
            "include_train_others": data.include_train_others,
            "db_content": data.db_content,
            "tokenizer": cfg.model.name if count_tokens else None,
        },
        "validation_databases": sorted(valid_dbs),
        "splits": {},
    }
    for split, items in plan.items():
        records = []
        for i, (source_file, ex) in enumerate(
            tqdm(items, desc=f"prepare {split}", disable=not progress)
        ):
            records.append(
                build_record(
                    ex,
                    record_id=f"{split}-{i:05d}",
                    split=split,
                    source_file=source_file,
                    schema=schemas[ex["db_id"]],
                    matcher=matcher,
                    count_tokens=count_tokens,
                    max_source_len=data.max_source_len,
                )
            )
        out = data.processed_dir / f"{split}.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        all_stats["splits"][split] = split_stats(records, data.max_target_len)

    (data.processed_dir / "stats.json").write_text(
        json.dumps(all_stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return all_stats


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
