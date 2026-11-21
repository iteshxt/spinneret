"""Schema-ranker training and scoring (specs §5.5).

Same conventions as the generator loop: JSON-lines progress, per-epoch
validation (coverage@(k1,k2) + AUC), best/last checkpoints, Hub push every epoch,
resume from the Hub's ``last-checkpoint``.
"""

from __future__ import annotations

import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from spinneret.config import ExperimentConfig, dump_config
from spinneret.data.prepare import read_jsonl
from spinneret.data.schema import Schema
from spinneret.data.spider import load_schemas
from spinneret.eval.coverage import covered, roc_auc
from spinneret.hub import HubSync
from spinneret.model.train import build_scheduler, emit, set_seed
from spinneret.ranker.dataset import Chunk, build_chunks
from spinneret.ranker.model import SchemaItemClassifier, collate_chunks, focal_loss

Scores = tuple[dict[str, float], dict[str, float]]  # (table scores, column scores)


def chunks_for(
    tokenizer: Any, schemas: dict[str, Schema], records: list[dict[str, Any]], max_length: int
) -> list[Chunk]:
    chunks = []
    for r in records:
        chunks.extend(
            build_chunks(
                tokenizer, r["id"], r["question"], schemas[r["db_id"]], r.get("matched_values", {}),
                gold_tables=r.get("gold_tables"), gold_columns=r.get("gold_columns"),
                max_length=max_length,
            )
        )  # fmt: skip
    return chunks


@torch.no_grad()
def score_chunks(
    model: SchemaItemClassifier, chunks: list[Chunk], pad_id: int, device: Any, batch_size: int = 32
) -> dict[str, Scores]:
    """Probability per table / column, keyed by example id."""
    model.eval()
    out: dict[str, Scores] = {}
    use_amp = device.type == "cuda"
    order = sorted(range(len(chunks)), key=lambda i: -len(chunks[i].input_ids))
    for start in range(0, len(order), batch_size):
        part = [chunks[i] for i in order[start : start + batch_size]]
        batch = collate_chunks(part, pad_id)
        tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            t_logits, c_logits = model(tensors)
        t_probs = torch.sigmoid(t_logits.float()).tolist()
        c_probs = torch.sigmoid(c_logits.float()).tolist()
        for (b, key), p in zip(batch["table_refs"], t_probs, strict=True):
            out.setdefault(part[b].example_id, ({}, {}))[0][key] = p
        for (b, key), p in zip(batch["column_refs"], c_probs, strict=True):
            out.setdefault(part[b].example_id, ({}, {}))[1][key] = p
    return out


def complete_scores(scores: Scores, schema: Schema) -> Scores:
    """Give every schema item a score (0.0 for items the ranker never saw)."""
    tables = {t.name.lower(): scores[0].get(t.name.lower(), 0.0) for t in schema.tables}
    columns = {c.key: scores[1].get(c.key, 0.0) for t in schema.tables for c in t.columns}
    return tables, columns


def ranker_metrics(
    scores: dict[str, Scores],
    records: list[dict[str, Any]],
    schemas: dict[str, Schema],
    k_tables: int,
    k_columns: int,
) -> dict[str, Any]:
    hits, t_pos, t_neg, c_pos, c_neg = 0, [], [], [], []
    for r in records:
        t_scores, c_scores = complete_scores(scores.get(r["id"], ({}, {})), schemas[r["db_id"]])
        hits += covered(
            t_scores, c_scores, r["gold_tables"], r["gold_columns"], k_tables, k_columns
        )
        gold_t = set(r["gold_tables"])
        gold_c = {f"{t}.{c}" for t, cols in r["gold_columns"].items() for c in cols}
        for key, s in t_scores.items():
            (t_pos if key in gold_t else t_neg).append(s)
        for key, s in c_scores.items():
            (c_pos if key in gold_c else c_neg).append(s)
    return {
        "valid_coverage": hits / max(1, len(records)),
        "table_auc": roc_auc(t_pos, t_neg),
        "column_auc": roc_auc(c_pos, c_neg),
        "k_tables": k_tables,
        "k_columns": k_columns,
    }


def _save(folder: Path, model: SchemaItemClassifier, tokenizer: Any, cfg: ExperimentConfig,
          state: dict[str, Any], optimizer: Any = None, scheduler: Any = None) -> None:  # fmt: skip
    tmp = folder.with_name(folder.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    model.save(tmp)
    tokenizer.save_pretrained(tmp)
    dump_config(cfg, tmp / "config.yaml")
    (tmp / "state.json").write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    if optimizer is not None:
        torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
        torch.save(scheduler.state_dict(), tmp / "scheduler.pt")
        torch.save({"python": random.getstate(), "torch": torch.get_rng_state()}, tmp / "rng.pt")
    shutil.rmtree(folder, ignore_errors=True)
    tmp.rename(folder)


def train_ranker(
    cfg: ExperimentConfig, *, resume: str | None = None, out_root: Path = Path("checkpoints")
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tc, data = cfg.train, cfg.data
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and tc.precision == "fp16"
    run_dir = out_root / cfg.run_name
    last_dir, best_dir = run_dir / "last", run_dir / "best"
    hub = (
        HubSync(cfg.hub.repo, private=cfg.hub.private_during_training)
        if cfg.hub.repo and cfg.hub.push_during_training
        else None
    )
    resume_dir: Path | None = None
    if resume == "local" and (last_dir / "state.json").is_file():
        resume_dir = last_dir
    elif resume == "hub" and hub:
        resume_dir = hub.download_last(run_dir / "hub")
    emit("start", run=cfg.run_name, device=str(device), resume_from=str(resume_dir or "scratch"))

    source = resume_dir or cfg.model.name
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = (SchemaItemClassifier.load(resume_dir) if resume_dir
             else SchemaItemClassifier.from_encoder(cfg.model.name)).to(device)  # fmt: skip

    schemas = load_schemas(data.spider_dir)
    train_recs = read_jsonl(data.processed_dir / "train.jsonl")
    valid_recs = read_jsonl(data.processed_dir / "valid.jsonl")
    if tc.max_valid_examples:
        valid_recs = random.Random(cfg.seed).sample(
            valid_recs, min(tc.max_valid_examples, len(valid_recs))
        )
    train_chunks = chunks_for(tokenizer, schemas, train_recs, data.max_source_len)
    valid_chunks = chunks_for(tokenizer, schemas, valid_recs, data.max_source_len)
    lengths = [len(c.input_ids) for c in train_chunks]

    steps_per_epoch = math.ceil(
        math.ceil(len(train_chunks) / tc.per_device_batch_size) / tc.gradient_accumulation_steps
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.learning_rate, weight_decay=0.01)
    scheduler = build_scheduler(optimizer, tc, steps_per_epoch * tc.max_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    state: dict[str, Any] = {"epoch": 0, "global_step": 0, "best": None, "best_epoch": None,
                             "bad_evals": 0, "history": []}  # fmt: skip
    if resume_dir:
        state = json.loads((resume_dir / "state.json").read_text(encoding="utf-8"))
        optimizer.load_state_dict(torch.load(resume_dir / "optimizer.pt", weights_only=False))
        scheduler.load_state_dict(torch.load(resume_dir / "scheduler.pt", weights_only=False))
    emit("data", train_examples=len(train_recs), train_chunks=len(train_chunks),
         valid_examples=len(valid_recs), steps_per_epoch=steps_per_epoch,
         params=sum(p.numel() for p in model.parameters()))  # fmt: skip

    from spinneret.model.data import length_grouped_batches

    pad_id = tokenizer.pad_token_id
    for epoch in range(state["epoch"] + 1, tc.max_epochs + 1):
        model.train()
        t0 = time.time()
        batches = length_grouped_batches(lengths, tc.per_device_batch_size,
                                         random.Random(cfg.seed * 1000 + epoch))  # fmt: skip
        running, seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for i, idx in enumerate(batches):
            batch = collate_chunks([train_chunks[j] for j in idx], pad_id)
            tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                t_logits, c_logits = model(tensors)
            loss = focal_loss(t_logits, tensors["table_labels"]) + focal_loss(
                c_logits, tensors["column_labels"]
            )
            scaler.scale(loss / tc.gradient_accumulation_steps).backward()
            running += loss.item()
            seen += 1
            if (i + 1) % tc.gradient_accumulation_steps == 0 or i + 1 == len(batches):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                state["global_step"] += 1
                if state["global_step"] % tc.log_every_steps == 0:
                    emit("step", epoch=epoch, step=state["global_step"],
                         loss=round(running / seen, 5), lr=scheduler.get_last_lr()[0])  # fmt: skip

        entry: dict[str, Any] = {"epoch": epoch, "train_loss": round(running / max(1, seen), 5),
                                 "train_seconds": round(time.time() - t0, 1)}  # fmt: skip
        scores = score_chunks(model, valid_chunks, pad_id, device, tc.eval_batch_size)
        metrics = ranker_metrics(scores, valid_recs, schemas, data.top_k_tables, data.top_k_columns)
        entry.update(metrics)
        best = state["best"]
        if best is None or (metrics["valid_coverage"], metrics["column_auc"] or 0) > (
            best["valid_coverage"],
            best["column_auc"] or 0,
        ):
            state["best"] = metrics
            state["best_epoch"] = epoch
            state["bad_evals"] = 0
            _save(best_dir, model, tokenizer, cfg, {**state, "history": state["history"] + [entry]})
            if hub:
                hub.push_best(
                    best_dir, f"best: epoch {epoch} coverage {metrics['valid_coverage']:.4f}"
                )
            entry["best"] = True
        else:
            state["bad_evals"] += 1
        state["epoch"] = epoch
        state["history"].append(entry)
        _save(last_dir, model, tokenizer, cfg, state, optimizer, scheduler)
        if hub and epoch % cfg.hub.push_last_every_epochs == 0:
            hub.push_last(last_dir, f"last: epoch {epoch}")
        emit("epoch", **entry, best_epoch=state["best_epoch"])
        if state["bad_evals"] >= tc.early_stopping_patience:
            emit("early_stop", epoch=epoch)
            break

    summary = {"run": cfg.run_name, "best": state["best"], "best_epoch": state["best_epoch"],
               "epochs": state["epoch"], "history": state["history"]}  # fmt: skip
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "best_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if hub and last_dir.exists():
        hub.push_last(last_dir, f"final: epoch {state['epoch']}")
    emit("done", **{k: v for k, v in summary.items() if k != "history"})
    return summary
