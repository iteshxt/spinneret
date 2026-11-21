"""Database-value matching, BRIDGE-style (specs §4.5).

For every TEXT column we cache its distinct values; at query time question
n-grams are matched against them so the model sees exact literal spellings,
e.g. ``singer : country ( France )``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.process import cdist

from spinneret.data.schema import Schema
from spinneret.db.sqlite import connect_readonly, execute

_NUMBER_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)$")
_WORD_RE = re.compile(r"\w+(?:['’]\w+)*")
STOPWORDS = frozenset(
    """
    a an the of in on at to for from by with and or not is are was were be been what which who
    whom whose how many much when where list show give find return all each every that this those
    these their its it his her than more most less least do does did have has had any some
    """.split()
)


def is_number(text: str) -> bool:
    return bool(_NUMBER_RE.match(text.strip()))


def _norm(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def build_value_index(
    db_path: Path, schema: Schema, *, max_value_len: int = 64, max_distinct: int = 5000
) -> dict[str, list[str]]:
    """Distinct values of every TEXT column, keyed by lower-cased ``table.column``."""
    index: dict[str, list[str]] = {}
    conn = connect_readonly(db_path)
    try:
        for table in schema.tables:
            for col in table.columns:
                if col.type != "text":
                    continue
                sql = (
                    f"SELECT DISTINCT {_quote_ident(col.name)} FROM {_quote_ident(table.name)} "
                    f"LIMIT {max_distinct + 1}"
                )
                try:
                    _, rows = execute(conn, sql, timeout_s=10.0)
                except Exception:  # noqa: BLE001 — broken columns are simply skipped
                    continue
                if len(rows) > max_distinct:
                    continue
                seen: dict[str, None] = {}
                for (raw,) in rows:
                    if not isinstance(raw, str):
                        continue
                    value = raw.strip()
                    if 0 < len(value) <= max_value_len and not is_number(value):
                        seen.setdefault(value, None)
                if seen:
                    index[col.key] = list(seen)
    finally:
        conn.close()
    return index


def question_ngrams(question: str, max_n: int = 6) -> list[str]:
    words = _WORD_RE.findall(question.lower())
    grams: dict[str, None] = {}
    for n in range(1, max_n + 1):
        for i in range(len(words) - n + 1):
            gram_words = words[i : i + n]
            if all(w in STOPWORDS for w in gram_words):
                continue
            grams.setdefault(" ".join(gram_words), None)
    return list(grams)


@dataclass
class _DbValues:
    keys: list[str]  # column key per value
    originals: list[str]
    normalized: list[str]


class ValueMatcher:
    """Match question n-grams to cached column values for one or more databases."""

    def __init__(
        self,
        db_path: Callable[[str], Path],
        schemas: dict[str, Schema],
        cache_dir: Path,
        *,
        max_per_column: int = 2,
        threshold: float = 85.0,
        min_fuzzy_len: int = 3,
    ) -> None:
        self._db_path = db_path
        self._schemas = schemas
        self._cache_dir = cache_dir
        self.max_per_column = max_per_column
        self.threshold = threshold
        self.min_fuzzy_len = min_fuzzy_len
        self._loaded: dict[str, _DbValues] = {}

    def index(self, db_id: str) -> dict[str, list[str]]:
        path = self._cache_dir / f"{db_id}.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        index = build_value_index(self._db_path(db_id), self._schemas[db_id])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
        return index

    def _values(self, db_id: str) -> _DbValues:
        if db_id not in self._loaded:
            keys, originals, normalized = [], [], []
            for key, values in self.index(db_id).items():
                for value in values:
                    norm = _norm(value)
                    if norm:
                        keys.append(key)
                        originals.append(value)
                        normalized.append(norm)
            self._loaded[db_id] = _DbValues(keys, originals, normalized)
        return self._loaded[db_id]

    def match(self, question: str, db_id: str) -> dict[str, list[str]]:
        """``{table.column: [values]}`` for values mentioned in ``question``."""
        values = self._values(db_id)
        grams = question_ngrams(question)
        if not grams or not values.originals:
            return {}

        scores = cdist(
            grams,
            values.normalized,
            scorer=fuzz.ratio,
            score_cutoff=self.threshold,
            dtype=np.uint8,
            workers=1,
        )
        best = scores.max(axis=0)
        gram_set = set(grams)
        candidates: dict[str, list[tuple[float, int, str]]] = {}
        for j in np.nonzero(best)[0]:
            norm = values.normalized[j]
            exact = norm in gram_set
            if not exact and len(norm) < self.min_fuzzy_len:
                continue
            score = 101.0 if exact else float(best[j])
            original = values.originals[j]
            candidates.setdefault(values.keys[j], []).append((score, len(original), original))

        matched: dict[str, list[str]] = {}
        for key, items in candidates.items():
            items.sort(key=lambda t: (-t[0], -t[1], t[2]))
            picked: list[str] = []
            for _, _, original in items:
                if original not in picked:
                    picked.append(original)
                if len(picked) == self.max_per_column:
                    break
            matched[key] = picked
        return matched
