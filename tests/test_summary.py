"""spinneret summarize: tables from whatever runs exist, the rest pending."""

from __future__ import annotations

import json
from pathlib import Path

from spinneret.eval.summary import build_summary, render_markdown

LEVELS = ("easy", "medium", "hard", "extra", "all")


def _metrics(em: float, ex: float) -> dict:
    return {
        "exact_match": dict.fromkeys(LEVELS, em),
        "execution": dict.fromkeys(LEVELS, ex),
        "valid_sql_rate": 0.9,
    }


def test_summary_marks_missing_runs_pending(tmp_path: Path) -> None:
    run = tmp_path / "t5-base-ranked-dev-eg8"
    run.mkdir()
    (run / "metrics.json").write_text(json.dumps(_metrics(0.626, 0.688)))
    (run / "latency.json").write_text(json.dumps({
        "generator_params_m": 223.5, "ranker_params_m": 125.0,
        "latency_ms": {s: {"p50": 1200.0, "p95": 2100.0}
                       for s in ("values", "schema", "generation", "execution", "total")},
    }))  # fmt: skip

    s = build_summary(tmp_path)
    t3 = s["tiers"][2]
    assert t3["scores"]["ex"] == 0.688
    assert s["tiers"][0]["scores"] is None
    md = render_markdown(s)
    assert (
        "| T3 Spinneret | T5-base + ranker + skeleton | 62.6 | 68.8 | — | 90.0 | 223.5 + 125.0 |"
        in md
    )
    assert "1.20 s | 2.10 s |" in md
    assert "| without values | pending | pending | pending |" in md
