from __future__ import annotations

import json

import pytest

from spinneret.config import ExperimentConfig
from spinneret.eval.coverage import covered, roc_auc


def test_coverage_rule():
    tables = {"singer": 0.9, "concert": 0.8, "stadium": 0.1}
    columns = {"singer.name": 0.9, "singer.age": 0.7, "singer.country": 0.1, "concert.year": 0.6}
    gold_c = {"singer": ["name", "age"]}
    assert covered(tables, columns, ["singer"], gold_c, k_tables=1, k_columns=2)
    assert not covered(tables, columns, ["singer"], gold_c, k_tables=1, k_columns=1)
    assert not covered(tables, columns, ["stadium"], {}, k_tables=2, k_columns=2)


def test_roc_auc():
    assert roc_auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert roc_auc([0.1], [0.9]) == 0.0
    assert roc_auc([0.5], [0.5]) == 0.5
    assert roc_auc([], [0.1]) is None


torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def roberta_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("FacebookAI/roberta-base")


@pytest.mark.network
def test_chunk_spans_align_with_names(roberta_tokenizer, concert_schema):
    from spinneret.ranker.dataset import build_chunks

    chunks = build_chunks(
        roberta_tokenizer, "x", "Which singers are from France?", concert_schema,
        {"singer.country": ["France"]}, gold_tables=["singer"],
        gold_columns={"singer": ["name", "country"]},
    )  # fmt: skip
    assert len(chunks) == 1
    items = {it.key: it for it in chunks[0].items}
    for key in ("singer", "singer.country", "singer_in_concert.singer_id"):
        it = items[key]
        decoded = roberta_tokenizer.decode(chunks[0].input_ids[it.tok_start : it.tok_end]).strip()
        assert decoded == key.split(".")[-1]
    assert items["singer"].label == 1 and items["stadium"].label == 0
    assert items["singer.country"].label == 1 and items["singer.age"].label == 0


@pytest.mark.network
def test_long_schemas_are_chunked_by_whole_tables(roberta_tokenizer, concert_schema):
    from spinneret.ranker.dataset import build_chunks

    chunks = build_chunks(roberta_tokenizer, "x", "How many singers?", concert_schema, {},
                          max_length=48)  # fmt: skip
    assert len(chunks) > 1
    seen_tables = [it.key for c in chunks for it in c.items if it.kind == "table"]
    assert sorted(seen_tables) == sorted(t.name.lower() for t in concert_schema.tables)
    for c in chunks:
        tables_here = {it.key for it in c.items if it.kind == "table"}
        assert all(it.table_key in tables_here for it in c.items)
        assert len(c.input_ids) <= 48


@pytest.fixture
def tiny_ranker_cfg(tmp_path, spider_dir, roberta_tokenizer):
    from transformers import RobertaConfig, RobertaModel

    src = spider_dir.parents[1] / "processed"
    if not (src / "train.jsonl").is_file():
        pytest.skip("run `spinneret prepare` first")
    processed = tmp_path / "processed"
    processed.mkdir()
    for split, n in (("train", 32), ("valid", 12), ("dev", 8)):
        lines = (src / f"{split}.jsonl").read_text().splitlines()[:n]
        (processed / f"{split}.jsonl").write_text("\n".join(lines) + "\n")
    enc_dir = tmp_path / "tiny-roberta"
    RobertaModel(
        RobertaConfig(
            vocab_size=roberta_tokenizer.vocab_size,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            max_position_embeddings=520,
            pad_token_id=roberta_tokenizer.pad_token_id,
        ),  # fmt: skip
        add_pooling_layer=False,
    ).save_pretrained(enc_dir)
    roberta_tokenizer.save_pretrained(enc_dir)
    return ExperimentConfig.model_validate(
        {
            "run_name": "tiny-ranker",
            "model": {"name": str(enc_dir), "added_tokens": "none"},
            "data": {"spider_dir": str(spider_dir), "processed_dir": str(processed)},
            "train": {"optimizer": "adamw", "learning_rate": 1e-3, "max_epochs": 2,
                      "per_device_batch_size": 8, "eval_batch_size": 8, "log_every_steps": 1,
                      "precision": "fp32"},
            "hub": {"repo": None},
        }
    )  # fmt: skip


@pytest.mark.data
@pytest.mark.network
@pytest.mark.slow
def test_train_ranker_then_build_ranked(tiny_ranker_cfg, tmp_path):
    from spinneret.ranker.infer import build_ranked
    from spinneret.ranker.train import train_ranker

    out_root = tmp_path / "checkpoints"
    summary = train_ranker(tiny_ranker_cfg, out_root=out_root)
    assert summary["epochs"] == 2
    best = summary["best"]
    assert 0 <= best["valid_coverage"] <= 1 and best["column_auc"] is not None
    ckpt = out_root / "tiny-ranker" / "best"
    assert (ckpt / "heads.pt").is_file()

    gen_cfg = tiny_ranker_cfg.model_copy(deep=True)
    gen_cfg.data.schema_input = "ranked"
    gen_cfg.data.ranker_checkpoint = str(ckpt)
    gen_cfg.data.top_k_tables, gen_cfg.data.top_k_columns = 2, 3
    out_dir = build_ranked(gen_cfg)
    assert out_dir == gen_cfg.data.processed_dir / "ranked" / "best"
    train_rows = [json.loads(x) for x in (out_dir / "train.jsonl").read_text().splitlines()]
    dev_rows = [json.loads(x) for x in (out_dir / "dev.jsonl").read_text().splitlines()]
    # training inputs always contain the gold tables (force-included)
    for row in train_rows:
        for table in row["gold_tables"]:
            assert f"| {table} :" in row["source"]
    assert all("ranker_covered" in r for r in dev_rows)
    stats = json.loads((out_dir / "stats.json").read_text())
    assert set(stats["splits"]) == {"train", "valid", "dev"}
