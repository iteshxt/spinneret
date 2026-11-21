"""Schema-item classifier: RoBERTa encoder + table and column heads (specs §5.5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from spinneret.ranker.dataset import Chunk

HEADS_FILE = "heads.pt"
META_FILE = "ranker.json"


class SchemaItemClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, hidden: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = encoder
        self.table_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )
        self.column_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    @classmethod
    def from_encoder(cls, name_or_path: str | Path) -> SchemaItemClassifier:
        from transformers import AutoModel

        encoder = AutoModel.from_pretrained(name_or_path, add_pooling_layer=False)
        return cls(encoder, encoder.config.hidden_size)

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(folder)
        heads = {k: v for k, v in self.state_dict().items() if not k.startswith("encoder.")}
        torch.save(heads, folder / HEADS_FILE)
        (folder / META_FILE).write_text(json.dumps({"type": "schema-item-classifier"}))

    @classmethod
    def load(cls, folder: Path) -> SchemaItemClassifier:
        model = cls.from_encoder(folder)
        heads = torch.load(folder / HEADS_FILE, map_location="cpu", weights_only=True)
        model.load_state_dict(heads, strict=False)
        return model

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).last_hidden_state  # (B, L, H)
        csum = torch.cumsum(hidden, dim=1)
        csum = torch.cat([torch.zeros_like(csum[:, :1]), csum], dim=1)  # prefix sums

        def pool(spans: torch.Tensor) -> torch.Tensor:  # (N, 3): batch, start, end
            b, s, e = spans[:, 0], spans[:, 1], spans[:, 2]
            total = csum[b, e] - csum[b, s]
            return total / (e - s).clamp(min=1).unsqueeze(-1).to(total.dtype)

        table_vecs = pool(batch["table_spans"])
        column_vecs = pool(batch["column_spans"])
        column_ctx = torch.cat([column_vecs, table_vecs[batch["column_table"]]], dim=-1)
        return self.table_head(table_vecs).squeeze(-1), self.column_head(column_ctx).squeeze(-1)


def collate_chunks(chunks: list[Chunk], pad_id: int) -> dict[str, Any]:
    """Pad chunks and flatten their items into span tensors."""
    max_len = max(len(c.input_ids) for c in chunks)
    input_ids = torch.full((len(chunks), max_len), pad_id, dtype=torch.long)
    attention = torch.zeros((len(chunks), max_len), dtype=torch.long)
    t_spans, t_labels, c_spans, c_labels, c_table = [], [], [], [], []
    t_refs, c_refs = [], []  # (chunk index, item key) for mapping scores back
    for b, chunk in enumerate(chunks):
        input_ids[b, : len(chunk.input_ids)] = torch.tensor(chunk.input_ids)
        attention[b, : len(chunk.input_ids)] = 1
        table_index: dict[str, int] = {}
        for item in chunk.items:
            if item.kind == "table":
                table_index[item.key] = len(t_spans)
                t_spans.append((b, item.tok_start, item.tok_end))
                t_labels.append(item.label)
                t_refs.append((b, item.key))
        for item in chunk.items:
            if item.kind == "column" and item.table_key in table_index:
                c_spans.append((b, item.tok_start, item.tok_end))
                c_labels.append(item.label)
                c_table.append(table_index[item.table_key])
                c_refs.append((b, item.key))
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "table_spans": torch.tensor(t_spans, dtype=torch.long).reshape(-1, 3),
        "table_labels": torch.tensor(t_labels, dtype=torch.float),
        "column_spans": torch.tensor(c_spans, dtype=torch.long).reshape(-1, 3),
        "column_labels": torch.tensor(c_labels, dtype=torch.float),
        "column_table": torch.tensor(c_table, dtype=torch.long),
        "table_refs": t_refs,
        "column_refs": c_refs,
    }


def focal_loss(
    logits: torch.Tensor, labels: torch.Tensor, gamma: float = 2.0, alpha: float = 0.75
) -> torch.Tensor:
    """Binary focal loss; ``alpha`` weights the (rare) positive class."""
    if logits.numel() == 0:
        return logits.sum()
    logits = logits.float()
    p = torch.sigmoid(logits)
    ce = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    p_t = p * labels + (1 - p) * (1 - labels)
    a_t = alpha * labels + (1 - alpha) * (1 - labels)
    return (a_t * (1 - p_t) ** gamma * ce).mean()
