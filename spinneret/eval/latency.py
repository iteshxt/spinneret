"""Per-question latency of a tier, one question at a time (specs §8.1, §12).

Each question goes through the full serving path: value matching, the schema stage
(ranking for T2/T3, plus rendering the input), generation (batch 1, the tier's beam)
and execution-guided selection. Stages are timed separately; p50 / p95 go to ``latency.json``.
"""

from __future__ import annotations

import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

from spinneret.config import ExperimentConfig, load_config
from spinneret.data.serialize import ranked_view, render, serialize_full
from spinneret.data.skeleton import split_output
from spinneret.data.spider import database_path, load_schemas
from spinneret.data.values import ValueMatcher
from spinneret.db.executor import Executor
from spinneret.decoding.execution_guided import select
from spinneret.hub import resolve_checkpoint
from spinneret.model.data import load_split
from spinneret.model.generate import generate
from spinneret.model.tokenizer import load_tokenizer

STAGES = ("values", "schema", "generation", "execution", "total")  # schema = ranking + rendering


def _pcts(ms: list[float]) -> dict[str, float]:
    if not ms:
        return {}
    qs = statistics.quantiles(ms, n=100, method="inclusive") if len(ms) > 1 else [ms[0]] * 99
    return {
        "p50": round(qs[49], 1),
        "p95": round(qs[94], 1),
        "mean": round(statistics.fmean(ms), 1),
    }


def _dir_mb(path: Path, pattern: str) -> float:
    return round(sum(p.stat().st_size for p in path.glob(pattern)) / 2**20, 1)


def benchmark(
    checkpoint: str,
    *,
    split: str = "dev",
    limit: int | None = None,
    seed: int = 0,
    beam_size: int | None = None,
    warmup: int = 5,
    device: str | None = None,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForSeq2SeqLM

    ckpt = resolve_checkpoint(checkpoint)
    cfg: ExperimentConfig = load_config(ckpt / "config.yaml")
    data = cfg.data
    k = beam_size or cfg.decoding.beam_size
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cuda = dev.type == "cuda"

    def sync() -> None:
        if cuda:
            torch.cuda.synchronize()

    tokenizer = load_tokenizer(str(ckpt))
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt).to(dev).eval()
    ranker = None
    if data.schema_input == "ranked":
        from spinneret.ranker.infer import SchemaRanker

        ranker = SchemaRanker(data.ranker_checkpoint or "", device=str(dev))
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    schemas = load_schemas(data.spider_dir)
    matcher = (
        ValueMatcher(
            lambda db_id: database_path(data.spider_dir, db_id),
            schemas,
            data.cache_dir / "values",
            max_per_column=data.max_values_per_column,
        )
        if data.db_content
        else None
    )
    executor = Executor(lambda db_id: database_path(data.spider_dir, db_id), timeout_s=5.0)

    def count_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=True)["input_ids"])

    examples = load_split(data.inputs_dir(), split, data.target_format)
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)  # warm-up questions are not always the same DB
    if limit:
        order = order[:limit]

    times: dict[str, list[float]] = {s: [] for s in STAGES}
    same_source = 0
    warm = order[:warmup]
    for n, i in enumerate([*warm, *order]):
        e = examples[i]
        schema = schemas[e.db_id]
        t0 = time.perf_counter()
        values = matcher.match(e.question, e.db_id) if matcher else {}
        t1 = time.perf_counter()
        if ranker is not None:
            scores = ranker.rank(e.question, schema, values)
            view = ranked_view(
                schema, scores, top_k_tables=data.top_k_tables,
                top_k_columns=data.top_k_columns, include_foreign_keys=data.include_foreign_keys,
            )  # fmt: skip
            source = render(e.question, e.db_id, view, values)
        else:
            source, _ = serialize_full(
                e.question,
                schema,
                values,
                count_tokens=count_tokens,
                max_tokens=data.max_source_len,
            )
        sync()
        t2 = time.perf_counter()
        beams = generate(
            model, tokenizer, [source], device=dev, num_beams=k,
            max_source_len=data.max_source_len, max_new_tokens=data.max_target_len, batch_size=1,
        )[0]  # fmt: skip
        sync()
        t3 = time.perf_counter()
        sqls = [split_output(b.text)[1] for b in beams]
        select(sqls, lambda sql, db=e.db_id: executor.run(db, sql, max_rows=1))
        t4 = time.perf_counter()
        if n < len(warm):
            continue
        same_source += source == e.source
        for stage, ms in zip(
            STAGES, ((t1 - t0), (t2 - t1), (t3 - t2), (t4 - t3), (t4 - t0)), strict=True
        ):
            times[stage].append(ms * 1000)

    weights = sorted(ckpt.glob("*.safetensors")) or sorted(ckpt.glob("*.bin"))
    result: dict[str, Any] = {
        "checkpoint": checkpoint,
        "run": cfg.run_name,
        "model": cfg.model.name,
        "device": torch.cuda.get_device_name(dev) if cuda else "cpu",
        "split": split,
        "questions": len(order),
        "beam_size": k,
        "decoding": "eg",
        "latency_ms": {s: _pcts(times[s]) for s in STAGES},
        "inputs_match_prepared": same_source / max(1, len(order)),
        "generator_params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "generator_weights_mb": sum(round(w.stat().st_size / 2**20, 1) for w in weights),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated(dev) / 2**20, 1)
        if cuda
        else None,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if ranker is not None:
        rdir = resolve_checkpoint(data.ranker_checkpoint or "")
        result["ranker_params_m"] = round(
            sum(p.numel() for p in ranker.model.parameters()) / 1e6, 1
        )
        result["ranker_weights_mb"] = _dir_mb(rdir, "*.safetensors") or _dir_mb(rdir, "*.bin")
    return result


def write_latency(result: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "latency.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path
