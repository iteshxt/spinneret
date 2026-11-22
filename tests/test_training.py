"""End-to-end smoke test of train → resume → predict → evaluate with a tiny T5."""

from __future__ import annotations

import json
import shutil

import pytest

from spinneret.config import ExperimentConfig

torch = pytest.importorskip("torch")


@pytest.fixture
def tiny_setup(tmp_path, spider_dir):
    from transformers import T5Config, T5ForConditionalGeneration

    from spinneret.model.tokenizer import load_tokenizer, tokens_for

    processed = tmp_path / "processed"
    processed.mkdir()
    src = spider_dir.parents[1] / "processed"
    if not (src / "train.jsonl").is_file():
        pytest.skip("run `spinneret prepare` first")
    for split, n in (("train", 24), ("valid", 8), ("dev", 6)):
        lines = (src / f"{split}.jsonl").read_text().splitlines()[:n]
        (processed / f"{split}.jsonl").write_text("\n".join(lines) + "\n")

    tokenizer = load_tokenizer("google-t5/t5-small", tokens_for("<"))
    config = T5Config(
        vocab_size=len(tokenizer), d_model=32, d_ff=64, d_kv=16, num_layers=1,
        num_decoder_layers=1, num_heads=2, decoder_start_token_id=0, pad_token_id=0,
        eos_token_id=1,
    )  # fmt: skip
    model_dir = tmp_path / "tiny-t5"
    T5ForConditionalGeneration(config).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    cfg = ExperimentConfig.model_validate(
        {
            "run_name": "tiny",
            "model": {"name": str(model_dir), "added_tokens": "none"},
            "data": {"spider_dir": str(spider_dir), "processed_dir": str(processed),
                     "max_target_len": 64},
            "train": {"max_epochs": 2, "per_device_batch_size": 8, "gradient_accumulation_steps": 1,
                      "eval_batch_size": 8, "log_every_steps": 1, "learning_rate": 1e-3},
            "hub": {"repo": None},
            "decoding": {"beam_size": 2},
        }
    )  # fmt: skip
    return cfg, tmp_path


@pytest.mark.data
@pytest.mark.slow
def test_train_resume_predict_evaluate(tiny_setup, capsys):
    from spinneret.eval.predict import predict
    from spinneret.eval.report import evaluate_run
    from spinneret.model.train import train

    cfg, tmp = tiny_setup
    out_root = tmp / "checkpoints"
    summary = train(cfg, out_root=out_root)
    assert summary["epochs"] == 2 and summary["best_epoch"] in (1, 2)
    run = out_root / "tiny"
    for name in ("config.yaml", "state.json", "model.safetensors"):
        assert (run / "best" / name).is_file()
    assert (run / "last" / "optimizer.pt").is_file()
    assert not (run / "best" / "optimizer.pt").exists()
    events = [json.loads(line)["event"] for line in capsys.readouterr().out.splitlines()
              if line.startswith("{")]  # fmt: skip
    assert events[0] == "start" and "epoch" in events and events[-1] == "done"

    cfg.train.max_epochs = 3
    resumed = train(cfg, resume="local", out_root=out_root)
    assert resumed["epochs"] == 3 and len(resumed["history"]) == 3

    report = predict(str(run / "best"), split="dev", decoding="eg", beam_size=2,
                     out_dir=tmp / "report", batch_size=4)  # fmt: skip
    preds = [json.loads(line) for line in (report / "predictions.jsonl").read_text().splitlines()]
    assert len(preds) == 6 and all(len(p["candidates"]) == 2 for p in preds)
    metrics = evaluate_run(report, cfg.data.spider_dir, progress=False)
    assert metrics["count"] == 6
    assert 0.0 <= metrics["execution"]["all"] <= 1.0
    shutil.rmtree(out_root)


def test_auto_added_tokens_require_an_audit_for_the_model(tmp_path):
    from spinneret.model.train import added_tokens_for

    report = {"google-t5/t5-small": {"added_tokens": [" <=", " <>", " <", "<"]}}
    (tmp_path / "added_tokens.json").write_text(json.dumps(report))

    def cfg(model):
        return ExperimentConfig.model_validate(
            {
                "run_name": "x",
                "model": {"name": model, "added_tokens": "auto"},
                "data": {"processed_dir": str(tmp_path)},
            }
        )

    assert added_tokens_for(cfg("google-t5/t5-small")) == [" <=", " <>", " <", "<"]
    with pytest.raises(RuntimeError, match="audit-tokenizer"):
        added_tokens_for(cfg("google-t5/t5-base"))
