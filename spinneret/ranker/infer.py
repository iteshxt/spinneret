"""Use a trained ranker: score schemas and build ranked inputs for T2/T3 (specs §5.5–5.6)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from spinneret.config import ExperimentConfig
from spinneret.data.prepare import read_jsonl
from spinneret.data.schema import Schema
from spinneret.data.serialize import RankedSchema, ranked_view, render
from spinneret.data.spider import load_schemas
from spinneret.eval.coverage import covered
from spinneret.hub import resolve_checkpoint
from spinneret.ranker.dataset import build_chunks
from spinneret.ranker.model import SchemaItemClassifier
from spinneret.ranker.train import chunks_for, complete_scores, score_chunks


class SchemaRanker:
    def __init__(self, checkpoint: str | Path, *, device: str | None = None) -> None:
        from transformers import AutoTokenizer

        folder = resolve_checkpoint(checkpoint)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(folder)
        self.model = SchemaItemClassifier.load(folder).to(self.device)
        if self.device.type == "cuda":
            self.model.half()
        self.model.eval()

    def rank(
        self, question: str, schema: Schema, values: dict[str, list[str]] | None = None
    ) -> RankedSchema:
        chunks = build_chunks(self.tokenizer, "q", question, schema, values or {})
        scores = score_chunks(self.model, chunks, self.tokenizer.pad_token_id, self.device)
        tables, columns = complete_scores(scores.get("q", ({}, {})), schema)
        return RankedSchema(tables=tables, columns=columns)

    def rank_records(
        self, records: list[dict[str, Any]], schemas: dict[str, Schema], batch_size: int = 32
    ) -> dict[str, RankedSchema]:
        chunks = chunks_for(self.tokenizer, schemas, records, 512)
        scores = score_chunks(self.model, chunks, self.tokenizer.pad_token_id, self.device,
                              batch_size)  # fmt: skip
        out = {}
        for r in records:
            t, c = complete_scores(scores.get(r["id"], ({}, {})), schemas[r["db_id"]])
            out[r["id"]] = RankedSchema(tables=t, columns=c)
        return out


def build_ranked(
    cfg: ExperimentConfig, *, splits: tuple[str, ...] = ("train", "valid", "dev")
) -> Path:
    """Write ranked inputs for ``cfg`` (schema: ranked) into ``cfg.data.inputs_dir()``."""
    data = cfg.data
    if data.schema_input != "ranked" or not data.ranker_checkpoint:
        raise ValueError("build-ranked needs a config with schema: ranked and ranker_checkpoint")
    ranker = SchemaRanker(data.ranker_checkpoint)
    schemas = load_schemas(data.spider_dir)
    out_dir = data.inputs_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {"ranker": data.ranker_checkpoint, "top_k_tables": data.top_k_tables,
                             "top_k_columns": data.top_k_columns, "splits": {}}  # fmt: skip
    for split in splits:
        records = read_jsonl(data.processed_dir / f"{split}.jsonl")
        ranked = ranker.rank_records(records, schemas)
        hits = 0
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as fh:
            for r in records:
                schema, scores = schemas[r["db_id"]], ranked[r["id"]]
                force = split == "train"  # gold items only ever added to *training* inputs
                view = ranked_view(
                    schema, scores,
                    top_k_tables=data.top_k_tables, top_k_columns=data.top_k_columns,
                    include_foreign_keys=data.include_foreign_keys,
                    force_tables=r["gold_tables"] if force else (),
                    force_columns=r["gold_columns"] if force else None,
                )  # fmt: skip
                ok = covered(scores.tables, scores.columns, r["gold_tables"], r["gold_columns"],
                             data.top_k_tables, data.top_k_columns)  # fmt: skip
                hits += ok
                row = dict(r)
                row["source"] = render(r["question"], r["db_id"], view, r.get("matched_values"))
                row["ranked_tables"] = sorted(scores.tables.items(), key=lambda kv: -kv[1])
                row["ranked_columns"] = sorted(scores.columns.items(), key=lambda kv: -kv[1])[:40]
                row["ranker_covered"] = ok
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        stats["splits"][split] = {"examples": len(records), "coverage": hits / max(1, len(records))}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    # the generator tokenizer audit applies unchanged
    added = data.processed_dir / "added_tokens.json"
    if added.is_file():
        (out_dir / "added_tokens.json").write_text(added.read_text(encoding="utf-8"))
    return out_dir
