"""Train / validation split by database, so validation stays cross-domain like dev."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from typing import Any


def validation_databases(
    examples: Sequence[dict[str, Any]], fraction: float, seed: int
) -> set[str]:
    """Pick whole databases until they cover at least ``fraction`` of ``examples``.

    Databases are shuffled from a sorted list with a seeded RNG, so the result is
    deterministic for a given (examples, fraction, seed).
    """
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between 0 and 1")
    counts = Counter(ex["db_id"] for ex in examples)
    db_ids = sorted(counts)
    random.Random(seed).shuffle(db_ids)

    target = fraction * len(examples)
    chosen: set[str] = set()
    covered = 0
    for db_id in db_ids:
        if covered >= target:
            break
        chosen.add(db_id)
        covered += counts[db_id]
    return chosen
