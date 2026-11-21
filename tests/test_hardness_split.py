from __future__ import annotations

import importlib
import sys
from collections import Counter

import pytest

from spinneret.data.hardness import hardness
from spinneret.data.spider import load_examples
from spinneret.data.split import validation_databases

from .conftest import OFFICIAL_EVAL_DIR


def test_hardness_easy_count():
    sql = {
        "select": [False, [[3, [0, [0, 0, False], None]]]],
        "from": {"table_units": [["table_unit", 1]], "conds": []},
        "where": [],
        "groupBy": [],
        "having": [],
        "orderBy": [],
        "limit": None,
        "intersect": None,
        "union": None,
        "except": None,
    }
    assert hardness(sql) == "easy"


@pytest.fixture(scope="module")
def official_evaluator():
    sys.path.insert(0, str(OFFICIAL_EVAL_DIR))
    try:
        evaluation = importlib.import_module("evaluation")
    finally:
        sys.path.remove(str(OFFICIAL_EVAL_DIR))
    return evaluation.Evaluator


@pytest.mark.data
def test_hardness_matches_official_script(spider_dir, official_evaluator):
    """The port must agree with the official eval_hardness on every Spider example."""
    evaluator = official_evaluator.__new__(official_evaluator)  # eval_hardness needs no state
    dev_levels = Counter()
    for source in ("train_spider", "train_others", "dev"):
        for ex in load_examples(spider_dir, source):
            ours = hardness(ex["sql"])
            assert ours == evaluator.eval_hardness(ex["sql"]), ex["query"]
            if source == "dev":
                dev_levels[ours] += 1
    print("dev hardness:", dict(dev_levels))
    assert sum(dev_levels.values()) == 1034


def test_validation_split_is_deterministic_and_by_database():
    examples = [{"db_id": f"db{i % 20}"} for i in range(1000)]
    a = validation_databases(examples, 0.10, seed=42)
    b = validation_databases(examples, 0.10, seed=42)
    assert a == b
    covered = sum(ex["db_id"] in a for ex in examples)
    assert 100 <= covered < 150  # at least 10%, whole databases only
    assert validation_databases(examples, 0.10, seed=7) != a


@pytest.mark.data
def test_validation_split_on_spider(spider_dir):
    train = load_examples(spider_dir, "train_spider")
    valid = validation_databases(train, 0.10, seed=42)
    n_valid = sum(ex["db_id"] in valid for ex in train)
    assert 700 <= n_valid < 1000
    others = {ex["db_id"] for ex in load_examples(spider_dir, "train_others")}
    dev = {ex["db_id"] for ex in load_examples(spider_dir, "dev")}
    assert not valid & others and not valid & dev
