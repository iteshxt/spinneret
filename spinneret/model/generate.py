"""Batched generation with optional beams (shared by validation and prediction)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Beam:
    text: str
    score: float  # sequence log-probability (length-normalised by HF beam search)


def generate(
    model: Any,
    tokenizer: Any,
    sources: list[str],
    *,
    device: Any,
    num_beams: int = 1,
    max_source_len: int = 512,
    max_new_tokens: int = 256,
    batch_size: int = 32,
) -> list[list[Beam]]:
    """Return ``num_beams`` candidates per source, best first, in input order."""
    import torch

    order = sorted(range(len(sources)), key=lambda i: -len(sources[i]))
    results: list[list[Beam]] = [[] for _ in sources]
    model.eval()
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            enc = tokenizer(
                [sources[i] for i in idx],
                max_length=max_source_len,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            out = model.generate(
                **enc,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
            texts = tokenizer.batch_decode(
                out.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            if num_beams > 1 and getattr(out, "sequences_scores", None) is not None:
                scores = out.sequences_scores.tolist()
            else:
                scores = [0.0] * len(texts)
            for j, i in enumerate(idx):
                chunk = range(j * num_beams, (j + 1) * num_beams)
                results[i] = [Beam(" ".join(texts[k].split()), float(scores[k])) for k in chunk]
    return results
