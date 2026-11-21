"""Ranker metrics: coverage@(k1, k2) and ROC-AUC (specs §8.1)."""

from __future__ import annotations

from collections.abc import Iterable


def covered(
    table_scores: dict[str, float],
    column_scores: dict[str, float],
    gold_tables: Iterable[str],
    gold_columns: dict[str, list[str]],
    k_tables: int,
    k_columns: int,
) -> bool:
    """True if every gold table is in the top-k1 tables and every gold column is in
    the top-k2 columns of its table — i.e. the ranked input contains all it needs."""
    top_tables = set(sorted(table_scores, key=lambda t: -table_scores[t])[:k_tables])
    if any(t not in top_tables for t in gold_tables):
        return False
    for table, cols in gold_columns.items():
        in_table = [k for k in column_scores if k.split(".", 1)[0] == table]
        top = set(sorted(in_table, key=lambda k: -column_scores[k])[:k_columns])
        if any(f"{table}.{c}" not in top for c in cols):
            return False
    return True


def roc_auc(positives: list[float], negatives: list[float]) -> float | None:
    """Mann–Whitney estimate of ROC-AUC (ties count half); None if a class is empty."""
    if not positives or not negatives:
        return None
    pooled = sorted([(s, 1) for s in positives] + [(s, 0) for s in negatives])
    rank_sum, i, n = 0.0, 0, len(pooled)
    while i < n:
        j = i
        while j + 1 < n and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        rank_sum += avg_rank * sum(1 for k in range(i, j + 1) if pooled[k][1] == 1)
        i = j + 1
    n_pos, n_neg = len(positives), len(negatives)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
