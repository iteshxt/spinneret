"""Training examples, tokenization, length-grouped batching and collation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spinneret.data.prepare import read_jsonl
from spinneret.data.skeleton import skeleton_target


@dataclass
class Example:
    id: str
    db_id: str
    question: str
    source: str
    target: str  # what the model must generate (SQL, or "skeleton | SQL")
    gold_sql: str  # original Spider SQL, for evaluation
    hardness: str


def load_split(inputs_dir: Path, split: str, target_format: str) -> list[Example]:
    examples = []
    for r in read_jsonl(inputs_dir / f"{split}.jsonl"):
        target = skeleton_target(r["target"]) if target_format == "skeleton_sql" else r["target"]
        examples.append(
            Example(
                id=r["id"],
                db_id=r["db_id"],
                question=r["question"],
                source=r["source"],
                target=target,
                gold_sql=r["gold_sql"],
                hardness=r["hardness"],
            )
        )
    return examples


def encode(
    tokenizer: Any, examples: list[Example], max_source_len: int, max_target_len: int
) -> list[dict[str, list[int]]]:
    sources = tokenizer(
        [e.source for e in examples], max_length=max_source_len, truncation=True
    ).input_ids
    targets = tokenizer(
        [e.target for e in examples], max_length=max_target_len, truncation=True
    ).input_ids
    return [{"input_ids": s, "labels": t} for s, t in zip(sources, targets, strict=True)]


def length_grouped_batches(
    lengths: list[int], batch_size: int, rng: random.Random, mega_factor: int = 50
) -> list[list[int]]:
    """Shuffle, then sort within mega-batches by length to cut padding; shuffle batches."""
    order = list(range(len(lengths)))
    rng.shuffle(order)
    mega = batch_size * mega_factor
    batches = []
    for start in range(0, len(order), mega):
        chunk = sorted(order[start : start + mega], key=lambda i: -lengths[i])
        batches.extend(chunk[i : i + batch_size] for i in range(0, len(chunk), batch_size))
    rng.shuffle(batches)
    return batches


def collate(features: list[dict[str, list[int]]], pad_id: int) -> dict[str, Any]:
    import torch

    max_in = max(len(f["input_ids"]) for f in features)
    max_out = max(len(f["labels"]) for f in features)
    input_ids = torch.full((len(features), max_in), pad_id, dtype=torch.long)
    attention = torch.zeros((len(features), max_in), dtype=torch.long)
    labels = torch.full((len(features), max_out), -100, dtype=torch.long)
    for i, f in enumerate(features):
        input_ids[i, : len(f["input_ids"])] = torch.tensor(f["input_ids"])
        attention[i, : len(f["input_ids"])] = 1
        labels[i, : len(f["labels"])] = torch.tensor(f["labels"])
    return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}
