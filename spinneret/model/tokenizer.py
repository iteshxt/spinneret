"""Tokenizer audit: find characters a tokenizer cannot encode (specs §4.7).

The original T5 SentencePiece vocabulary has no ``<``; every "less than" query
would silently turn into ``<unk>``. The audit finds such characters in the
training targets, decides which tokens to add, and verifies the round trip.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ADDED_TOKENS_FILE = "added_tokens.json"
# Tokens to add for a missing character. A bare added token swallows the space before
# it on decode ("age< 30"), so — like PICARD's T5 checkpoints — we add space-prefixed
# variants, which match the normalized SQL (operators always have spaces around them),
# longest first, plus the bare character as a fallback (e.g. inside string values).
SPECIAL_CASES = {"<": [" <=", " <>", " <", "<"]}


def tokens_for(ch: str) -> list[str]:
    return SPECIAL_CASES.get(ch, [f" {ch}", ch])


def load_tokenizer(name: str, added_tokens: Iterable[str] = ()) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    extra = [t for t in added_tokens if t]
    if extra:
        tokenizer.add_tokens(extra)
    return tokenizer


def roundtrip(tokenizer: Any, text: str) -> str:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def _same(a: str, b: str) -> bool:
    return " ".join(a.split()) == " ".join(b.split())


def missing_characters(tokenizer: Any, texts: Iterable[str]) -> dict[str, int]:
    counts = Counter(ch for text in texts for ch in text if not ch.isspace())
    unk = tokenizer.unk_token_id
    return {
        ch: n
        for ch, n in sorted(counts.items(), key=lambda kv: -kv[1])
        if unk is not None and unk in tokenizer(ch, add_special_tokens=False).input_ids
    }


def _failures(tokenizer: Any, texts: list[str], limit: int = 5) -> tuple[int, list[dict[str, str]]]:
    count, examples = 0, []
    for text in texts:
        back = roundtrip(tokenizer, text)
        if not _same(back, text):
            count += 1
            if len(examples) < limit:
                examples.append({"text": text, "decoded": back})
    return count, examples


def audit(model_name: str, targets: list[str], sources: list[str]) -> dict[str, Any]:
    """Audit ``model_name`` on SQL targets (must round-trip) and inputs (reported only)."""
    tokenizer = load_tokenizer(model_name)
    missing_targets = missing_characters(tokenizer, targets)
    missing_sources = missing_characters(tokenizer, sources)

    to_add: list[str] = []
    for ch in missing_targets:
        to_add.extend(t for t in tokens_for(ch) if t not in to_add)

    before, before_examples = _failures(tokenizer, targets)
    fixed = load_tokenizer(model_name, to_add)
    after, after_examples = _failures(fixed, targets)
    return {
        "model": model_name,
        "n_targets": len(targets),
        "n_sources": len(sources),
        "missing_in_targets": missing_targets,
        "missing_in_sources": missing_sources,
        "added_tokens": to_add,
        "target_roundtrip_failures_before": before,
        "target_roundtrip_failures_after": after,
        "examples_before": before_examples,
        "examples_after": after_examples,
    }


def read_added_tokens(processed_dir: Path, model_name: str) -> list[str]:
    """Tokens recorded by the audit for ``model_name`` (empty if not audited)."""
    path = processed_dir / ADDED_TOKENS_FILE
    if not path.is_file():
        return []
    report = json.loads(path.read_text(encoding="utf-8"))
    return list(report.get(model_name, {}).get("added_tokens", []))
