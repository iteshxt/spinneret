"""Question → SQL candidates for one served model (specs §6.1, design §5.2).

A ``Translator`` wraps one generator checkpoint and follows the exact input path the
model was trained and evaluated with (its ``config.yaml``): value matching, then either
the full schema (T1) or the ranker's top-k view (T2, T3), then beam search. The ranker
is passed in so one instance can be shared by several translators.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spinneret.config import ExperimentConfig, load_config
from spinneret.data.schema import Schema
from spinneret.data.serialize import ranked_view, render, serialize_full
from spinneret.data.skeleton import split_output
from spinneret.data.spider import database_path
from spinneret.data.values import ValueMatcher
from spinneret.hub import resolve_checkpoint
from spinneret.model.generate import generate
from spinneret.model.tokenizer import load_tokenizer


@dataclass
class Candidate:
    rank: int  # 0 = top beam
    sql: str
    skeleton: str | None
    score: float  # sequence log-probability


@dataclass
class SchemaItem:
    name: str
    score: float
    kept: bool
    table: str | None = None  # columns only


@dataclass
class Translation:
    candidates: list[Candidate]
    matched_values: dict[str, list[str]]
    tables: list[SchemaItem] | None = None  # None without a ranker (T1)
    columns: list[SchemaItem] | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)


class Translator:
    def __init__(
        self,
        checkpoint: str | Path,
        schemas: dict[str, Schema],
        *,
        ranker: Any | None = None,
        device: Any = None,
        beam_size: int | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM

        folder = resolve_checkpoint(checkpoint)
        self.checkpoint = str(checkpoint)
        self.cfg: ExperimentConfig = load_config(folder / "config.yaml")
        data = self.cfg.data
        if data.schema_input == "ranked" and ranker is None:
            raise ValueError(f"{checkpoint} was trained on ranked inputs and needs a ranker")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = load_tokenizer(str(folder))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(folder).to(self.device).eval()
        self.ranker = ranker if data.schema_input == "ranked" else None
        self.beam_size = beam_size or self.cfg.decoding.beam_size
        self.schemas = schemas
        self.matcher = (
            ValueMatcher(
                lambda db_id: database_path(data.spider_dir, db_id),
                schemas,
                data.cache_dir / "values",
                max_per_column=data.max_values_per_column,
            )
            if data.db_content
            else None
        )
        self.params_m = round(sum(p.numel() for p in self.model.parameters()) / 1e6, 1)

    @property
    def uses_skeleton(self) -> bool:
        return self.cfg.data.target_format == "skeleton_sql"

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=True)["input_ids"])

    def translate(self, question: str, db_id: str, k: int | None = None) -> Translation:
        data = self.cfg.data
        schema = self.schemas[db_id]
        sync = self._sync
        t0 = time.perf_counter()
        values = self.matcher.match(question, db_id) if self.matcher else {}
        t1 = time.perf_counter()
        tables = columns = None
        if self.ranker is not None:
            ranked = self.ranker.rank(question, schema, values)
            view = ranked_view(
                schema, ranked, top_k_tables=data.top_k_tables,
                top_k_columns=data.top_k_columns, include_foreign_keys=data.include_foreign_keys,
            )  # fmt: skip
            source = render(question, db_id, view, values)
            tables, columns = _schema_items(schema, ranked, view.tables)
        else:
            source, _ = serialize_full(
                question, schema, values, count_tokens=self._count_tokens,
                max_tokens=data.max_source_len,
            )  # fmt: skip
        sync()
        t2 = time.perf_counter()
        beams = generate(
            self.model, self.tokenizer, [source], device=self.device,
            num_beams=k or self.beam_size, max_source_len=data.max_source_len,
            max_new_tokens=data.max_target_len, batch_size=1,
        )[0]  # fmt: skip
        sync()
        t3 = time.perf_counter()
        candidates = []
        for i, beam in enumerate(beams):
            skeleton, sql = split_output(beam.text)
            candidates.append(Candidate(rank=i, sql=sql, skeleton=skeleton, score=beam.score))
        return Translation(
            candidates=candidates,
            matched_values=values,
            tables=tables,
            columns=columns,
            timings_ms={
                "values": _ms(t1 - t0),
                "ranking": _ms(t2 - t1) if self.ranker is not None else 0.0,
                "inference": _ms(t3 - t2),
            },
        )

    def _sync(self) -> None:
        if self.device.type == "cuda":
            import torch

            torch.cuda.synchronize(self.device)


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 1)


def _schema_items(
    schema: Schema, ranked: Any, kept: list[tuple[str, list[str]]]
) -> tuple[list[SchemaItem], list[SchemaItem]]:
    """Every table and column with its ranker score; ``kept`` = shown to the generator."""
    kept_tables = {t.lower() for t, _ in kept}
    kept_columns = {f"{t.lower()}.{c.lower()}" for t, cols in kept for c in cols}
    tables, columns = [], []
    for table in schema.tables:
        t = table.name.lower()
        tables.append(SchemaItem(table.name, round(ranked.tables.get(t, 0.0), 4),
                                 t in kept_tables))  # fmt: skip
        for col in table.columns:
            key = f"{t}.{col.name.lower()}"
            if col.name == "*":
                continue
            columns.append(SchemaItem(col.name, round(ranked.columns.get(key, 0.0), 4),
                                      key in kept_columns, table=table.name))  # fmt: skip
    tables.sort(key=lambda x: -x.score)
    columns.sort(key=lambda x: -x.score)
    return tables, columns
