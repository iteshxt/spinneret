"""Run a trained generator over a split and write a predictions run (specs §8.2)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from spinneret.config import ExperimentConfig, load_config
from spinneret.data.skeleton import split_output
from spinneret.data.spider import database_path
from spinneret.db.executor import Executor
from spinneret.decoding.execution_guided import select
from spinneret.hub import resolve_checkpoint
from spinneret.model.data import load_split
from spinneret.model.generate import generate
from spinneret.model.tokenizer import load_tokenizer


def predict(
    checkpoint: str,
    *,
    split: str = "dev",
    decoding: str = "eg",
    beam_size: int | None = None,
    out_dir: Path | None = None,
    batch_size: int = 16,
    limit: int | None = None,
) -> Path:
    import torch
    from transformers import AutoModelForSeq2SeqLM

    ckpt = resolve_checkpoint(checkpoint)
    cfg: ExperimentConfig = load_config(ckpt / "config.yaml")
    k = 1 if decoding == "greedy" else (beam_size or cfg.decoding.beam_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(str(ckpt))
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt).to(device)

    examples = load_split(cfg.data.inputs_dir(), split, cfg.data.target_format)
    if limit:
        examples = examples[:limit]
    spider_dir = cfg.data.spider_dir
    executor = Executor(lambda db_id: database_path(spider_dir, db_id), timeout_s=5.0)

    t0 = time.time()
    beams = generate(
        model,
        tokenizer,
        [e.source for e in examples],
        device=device,
        num_beams=k,
        max_source_len=cfg.data.max_source_len,
        max_new_tokens=cfg.data.max_target_len,
        batch_size=batch_size,
    )
    gen_seconds = time.time() - t0

    run_id = f"{cfg.run_name}-{split}-{decoding}{k if k > 1 else ''}"
    out_dir = out_dir or Path("reports") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for e, cands in zip(examples, beams, strict=True):
        parsed = [split_output(c.text) for c in cands]
        sqls = [sql for _, sql in parsed]
        if decoding == "eg":
            chosen = select(sqls, lambda sql, db=e.db_id: executor.run(db, sql, max_rows=1))
            index, statuses, executed = chosen.index, chosen.statuses, chosen.executed
        else:
            index, statuses, executed = 0, ["not_run"] * len(sqls), None
        records.append(
            {
                "id": e.id,
                "db_id": e.db_id,
                "question": e.question,
                "hardness": e.hardness,
                "gold_sql": e.gold_sql,
                "pred_sql": sqls[index],
                "skeleton": parsed[index][0],
                "selected_rank": index,
                "executed": executed,
                "candidates": [
                    {"sql": s, "score": c.score, "status": st}
                    for s, c, st in zip(sqls, cands, statuses, strict=True)
                ],
            }
        )

    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / "pred.txt").write_text("".join(r["pred_sql"] + "\n" for r in records))
    meta = {
        "run_id": run_id,
        "checkpoint": checkpoint,
        "model": cfg.model.name,
        "config_run": cfg.run_name,
        "split": split,
        "decoding": decoding,
        "beam_size": k,
        "examples": len(records),
        "device": str(device),
        "generation_seconds": round(gen_seconds, 1),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir
