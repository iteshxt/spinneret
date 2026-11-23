from __future__ import annotations

import json
import re
from pathlib import Path

from spinneret.cards import generator_card, ranker_card
from spinneret.config import ExperimentConfig, ServeModel


def _reports(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    levels = {lv: 0.5 for lv in ("easy", "medium", "hard", "extra", "all")}
    (run / "metrics.json").write_text(json.dumps({
        "exact_match": levels, "execution": {**levels, "all": 0.688}, "valid_sql_rate": 0.917,
        "meta": {"beam_size": 8},
    }))  # fmt: skip
    (run / "latency.json").write_text(json.dumps({
        "generator_params_m": 222.9, "ranker_params_m": 125.8, "peak_gpu_memory_mb": 1700,
        "device": "GPU", "latency_ms": {"total": {"p50": 1290.0, "p95": 2500.0}},
    }))  # fmt: skip
    (run / "question_types.json").write_text(
        json.dumps({"types": {"multi-join": {"count": 43, "ex": 0.279}}})
    )
    return tmp_path


def _cfg(ranked: bool) -> ExperimentConfig:
    data = {"schema": "ranked", "ranker_checkpoint": "hub:u/r", "target_format": "skeleton_sql"}
    return ExperimentConfig.model_validate(
        {"model": {"name": "google-t5/t5-base"}, "data": data if ranked else {}}
    )


def test_generator_card_has_required_fields(tmp_path: Path) -> None:
    m = ServeModel(id="t3", label="Spinneret · T5-base + ranker", generator="hub:u/gen", run="run")
    state = {"best_epoch": 18, "best": {"valid_em": 0.6, "valid_ex": 0.67}}
    card = generator_card(m, _reports(tmp_path), "hub:u/rank", checkpoint=(_cfg(True), state))
    assert card.startswith("---\nlicense: cc-by-sa-4.0\n")
    for needle in ("base_model: google-t5/t5-base", "# Spinneret · T5-base + ranker",
                   "| Execution accuracy % | 50.0 | 50.0 | 50.0 | 50.0 | 68.8 |",
                   "Valid (executable) SQL: 91.7%", "| multi-join | 43 | 27.9 |",
                   "median 1.29 s", "(+ 125.8M ranker)", "u/rank", "fk :", "drop the skeleton",
                   "Best epoch: 18", "CC BY-SA 4.0", "github.com/iteshxt/spinneret"):  # fmt: skip
        assert needle in card, needle
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", card)  # never a Hub token


def test_full_schema_card_has_no_ranker_or_skeleton(tmp_path: Path) -> None:
    m = ServeModel(id="t1", label="Baseline · T5-small", generator="hub:u/small", run="run")
    card = generator_card(m, _reports(tmp_path), None, checkpoint=(_cfg(False), {}))
    assert "# Spinneret Baseline · T5-small" in card
    assert "reads the whole schema" in card and "fk :" not in card and "skeleton" not in card


def test_ranker_card() -> None:
    summary = {"e5_ranker_k": {f"{a}x{b}": {"dev_coverage": 0.985} for a in (3, 4, 5)
                               for b in (4, 5, 6)}}  # fmt: skip
    state = {"best_epoch": 2, "best": {"table_auc": 0.98, "column_auc": 0.97,
                                       "valid_coverage": 0.981}}  # fmt: skip
    card = ranker_card(summary, state)
    assert "base_model: FacebookAI/roberta-base" in card
    assert "Spider dev coverage at (4, 5): 98.5%" in card and "| 4 | 98.5 | 98.5 | 98.5 |" in card
