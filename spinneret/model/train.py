"""Generator fine-tuning loop (specs §5.2).

A small, explicit PyTorch loop rather than ``Seq2SeqTrainer``: per-epoch validation
with real EM/EX (official exact-set-match + fast execution match), best-checkpoint
selection, early stopping, and resumable checkpoints pushed to the Hugging Face
Hub every epoch so Colab sessions can be reclaimed without losing progress.
Progress is printed as JSON lines (``logs.sh`` tails them).
"""

from __future__ import annotations

import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any

from spinneret.config import ExperimentConfig, TrainConfig, dump_config
from spinneret.data.skeleton import split_output
from spinneret.eval.official import FastExecMatch, OfficialEvaluator
from spinneret.hub import HubSync
from spinneret.model.data import Example, collate, encode, length_grouped_batches, load_split
from spinneret.model.generate import generate
from spinneret.model.tokenizer import load_tokenizer, read_added_tokens


def emit(event: str, **fields: Any) -> None:
    record = {"event": event, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **fields}
    print(json.dumps(record, default=str), flush=True)


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def added_tokens_for(cfg: ExperimentConfig) -> list[str]:
    choice = cfg.model.added_tokens
    if choice == "none":
        return []
    if choice == "auto":
        report = cfg.data.processed_dir / "added_tokens.json"
        if report.is_file() and cfg.model.name not in json.loads(report.read_text("utf-8")):
            # A silent empty list would train a model that cannot write "<" (specs §5.2).
            raise RuntimeError(
                f"no tokenizer audit for {cfg.model.name} in {report}; run "
                f"`spinneret audit-tokenizer --model {cfg.model.name}` first"
            )
        return read_added_tokens(cfg.data.processed_dir, cfg.model.name)
    return list(choice)


def build_optimizer(model: Any, tc: TrainConfig) -> Any:
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    if tc.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=tc.learning_rate, weight_decay=0.0)
    try:
        from transformers.optimization import Adafactor

        return Adafactor(
            params,
            lr=tc.learning_rate,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )
    except ImportError:  # transformers without its Adafactor: use PyTorch's
        return torch.optim.Adafactor(params, lr=tc.learning_rate)


def build_scheduler(optimizer: Any, tc: TrainConfig, total_steps: int) -> Any:
    import torch

    warmup = int(tc.warmup_ratio * total_steps)

    def factor(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        if tc.lr_schedule == "linear":
            return max(0.0, (total_steps - step) / max(1, total_steps - warmup))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class Validator:
    """Greedy-decode the validation split; official EM + fast single-DB EX."""

    def __init__(self, cfg: ExperimentConfig, examples: list[Example]) -> None:
        db_dir = cfg.data.spider_dir / "database"
        self.cfg = cfg
        self.examples = examples
        self.em = OfficialEvaluator(db_dir, cfg.data.spider_dir / "tables.json", etype="match")
        self.ex = FastExecMatch(db_dir)

    def __call__(self, model: Any, tokenizer: Any, device: Any) -> dict[str, Any]:
        beams = generate(
            model,
            tokenizer,
            [e.source for e in self.examples],
            device=device,
            num_beams=1,
            max_source_len=self.cfg.data.max_source_len,
            max_new_tokens=self.cfg.data.max_target_len,
            batch_size=self.cfg.train.eval_batch_size,
        )
        em = ex = ex_n = 0
        for e, b in zip(self.examples, beams, strict=True):
            _, sql = split_output(b[0].text)
            em += self.em.score(sql, e.gold_sql, e.db_id).exact or 0
            hit = self.ex(sql, e.gold_sql, e.db_id)
            if hit is not None:
                ex += hit
                ex_n += 1
        n = len(self.examples)
        sample = [
            {"question": e.question, "pred": b[0].text, "gold": e.target}
            for e, b in list(zip(self.examples, beams, strict=True))[:3]
        ]
        return {"valid_em": em / n, "valid_ex": ex / max(1, ex_n), "n": n, "sample": sample}


def _save(
    folder: Path,
    model: Any,
    tokenizer: Any,
    cfg: ExperimentConfig,
    state: dict[str, Any],
    *,
    optimizer: Any = None,
    scheduler: Any = None,
) -> None:
    import torch

    tmp = folder.with_name(folder.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    model.save_pretrained(tmp)
    tokenizer.save_pretrained(tmp)
    dump_config(cfg, tmp / "config.yaml")
    (tmp / "state.json").write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    if optimizer is not None:
        torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
        torch.save(scheduler.state_dict(), tmp / "scheduler.pt")
        rng = {"python": random.getstate(), "torch": torch.get_rng_state()}
        if torch.cuda.is_available():
            rng["cuda"] = torch.cuda.get_rng_state_all()
        torch.save(rng, tmp / "rng.pt")
    shutil.rmtree(folder, ignore_errors=True)
    tmp.rename(folder)


def _better(metrics: dict[str, Any], best: dict[str, Any] | None, key: str) -> bool:
    if best is None:
        return True
    other = "valid_em" if key == "valid_ex" else "valid_ex"
    return (metrics[key], metrics[other]) > (best[key], best[other])


def train(
    cfg: ExperimentConfig, *, resume: str | None = None, out_root: Path = Path("checkpoints")
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForSeq2SeqLM

    tc = cfg.train
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    elif resume == "hub":
        if hub is None:
            raise ValueError("--resume hub needs hub.repo in the config")
        resume_dir = hub.download_last(run_dir / "hub")
    emit("start", run=cfg.run_name, device=str(device), resume_from=str(resume_dir or "scratch"))

    if resume_dir:
        tokenizer = load_tokenizer(str(resume_dir))
        model = AutoModelForSeq2SeqLM.from_pretrained(resume_dir)
    else:
        tokenizer = load_tokenizer(cfg.model.name, added_tokens_for(cfg))
        model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.name)
        model.resize_token_embeddings(len(tokenizer))
    if tc.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)

    inputs_dir = cfg.data.inputs_dir()
    train_ex = load_split(inputs_dir, "train", cfg.data.target_format)
    valid_ex = load_split(inputs_dir, "valid", cfg.data.target_format)
    if tc.max_valid_examples:
        valid_ex = random.Random(cfg.seed).sample(
            valid_ex, min(tc.max_valid_examples, len(valid_ex))
        )
    features = encode(tokenizer, train_ex, cfg.data.max_source_len, cfg.data.max_target_len)
    lengths = [len(f["input_ids"]) for f in features]
    micro_batches = math.ceil(len(features) / tc.per_device_batch_size)
    steps_per_epoch = math.ceil(micro_batches / tc.gradient_accumulation_steps)
    total_steps = steps_per_epoch * tc.max_epochs

    optimizer = build_optimizer(model, tc)
    scheduler = build_scheduler(optimizer, tc, total_steps)
    state: dict[str, Any] = {
        "epoch": 0, "global_step": 0, "best": None, "best_epoch": None,
        "bad_evals": 0, "history": [],
    }  # fmt: skip
    if resume_dir:
        state = json.loads((resume_dir / "state.json").read_text(encoding="utf-8"))
        optimizer.load_state_dict(torch.load(resume_dir / "optimizer.pt", weights_only=False))
        scheduler.load_state_dict(torch.load(resume_dir / "scheduler.pt", weights_only=False))
        rng = torch.load(resume_dir / "rng.pt", weights_only=False)
        random.setstate(rng["python"])
        torch.set_rng_state(rng["torch"])
        if "cuda" in rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        # The best checkpoint so far stays at the Hub repo root; it is re-pushed only
        # when a later epoch beats state["best"].

    validator = Validator(cfg, valid_ex)
    emit(
        "data", train=len(train_ex), valid=len(valid_ex), steps_per_epoch=steps_per_epoch,
        params=sum(p.numel() for p in model.parameters()), vocab=len(tokenizer),
    )  # fmt: skip

    pad_id = tokenizer.pad_token_id
    for epoch in range(state["epoch"] + 1, tc.max_epochs + 1):
        model.train()
        t0 = time.time()
        batches = length_grouped_batches(
            lengths, tc.per_device_batch_size, random.Random(cfg.seed * 1000 + epoch)
        )
        running, seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for i, idx in enumerate(batches):
            batch = {
                k: v.to(device) for k, v in collate([features[j] for j in idx], pad_id).items()
            }
            loss = model(**batch).loss
            (loss / tc.gradient_accumulation_steps).backward()
            running += loss.item()
            seen += 1
            if (i + 1) % tc.gradient_accumulation_steps == 0 or i + 1 == len(batches):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                state["global_step"] += 1
                if state["global_step"] % tc.log_every_steps == 0:
                    loss_avg = round(running / seen, 4)
                    lr = scheduler.get_last_lr()[0]
                    emit("step", epoch=epoch, step=state["global_step"], loss=loss_avg, lr=lr)
        train_loss = running / max(1, seen)
        entry: dict[str, Any] = {"epoch": epoch, "train_loss": round(train_loss, 4),
                                 "train_seconds": round(time.time() - t0, 1)}  # fmt: skip

        if epoch % tc.eval_every_epochs == 0:
            t1 = time.time()
            metrics = validator(model, tokenizer, device)
            entry.update({k: metrics[k] for k in ("valid_em", "valid_ex")})
            entry["valid_seconds"] = round(time.time() - t1, 1)
            if _better(metrics, state["best"], tc.select_best_by):
                state["best"] = {k: metrics[k] for k in ("valid_em", "valid_ex")}
                state["best_epoch"] = epoch
                state["bad_evals"] = 0
                _save(
                    best_dir,
                    model,
                    tokenizer,
                    cfg,
                    {**state, "history": state["history"] + [entry]},
                )
                if hub:
                    hub.push_best(
                        best_dir, f"best: epoch {epoch} valid_ex {metrics['valid_ex']:.4f}"
                    )
                entry["best"] = True
            else:
                state["bad_evals"] += 1
            entry["sample"] = metrics["sample"][0]

        state["epoch"] = epoch
        state["history"].append({k: v for k, v in entry.items() if k != "sample"})
        _save(last_dir, model, tokenizer, cfg, state, optimizer=optimizer, scheduler=scheduler)
        if hub and epoch % cfg.hub.push_last_every_epochs == 0:
            hub.push_last(last_dir, f"last: epoch {epoch}")
        emit("epoch", **entry, best_so_far=state["best"], best_epoch=state["best_epoch"])
        if state["bad_evals"] >= tc.early_stopping_patience:
            emit("early_stop", epoch=epoch, patience=tc.early_stopping_patience)
            break

    validator.ex.close()
    summary = {"run": cfg.run_name, "best": state["best"], "best_epoch": state["best_epoch"],
               "epochs": state["epoch"], "history": state["history"]}  # fmt: skip
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "best_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if hub and last_dir.exists():
        hub.push_last(last_dir, f"final: epoch {state['epoch']}")
    emit("done", **{k: v for k, v in summary.items() if k != "history"})
    return summary
