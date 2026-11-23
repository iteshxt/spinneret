"""Failure taxonomy and question types (specs §8.4–8.5) on in-memory schemas."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from spinneret.eval.analysis import (
    compare,
    label,
    question_features,
    question_type,
    review_agreement,
    summarize_taxonomy,
)
from spinneret.eval.official import official_modules

SCHEMA = {
    "singer": ["singer_id", "name", "country", "age"],
    "concert": ["concert_id", "name", "stadium_id", "year"],
    "singer_in_concert": ["singer_id", "concert_id"],
    "stadium": ["stadium_id", "name", "capacity"],
}


@pytest.fixture(scope="module")
def parse():  # noqa: ANN201
    ps = official_modules()["process_sql"]
    schema = ps.Schema(SCHEMA)

    def run(sql: str, db_id: str = "concert_singer") -> dict | None:
        try:  # like analysis.Parser: unparsable SQL becomes None
            return ps.get_sql(schema, sql)
        except Exception:  # noqa: BLE001
            return None

    return run


JOIN = (
    "SELECT T1.name FROM singer AS T1 JOIN singer_in_concert AS T2 ON T1.singer_id = T2.singer_id"
)


@pytest.mark.parametrize(
    ("pred", "gold", "primary"),
    [
        ("SELECT name FROM stadium", "SELECT name FROM singer", "wrong_table"),
        ("SELECT name FROM singer", JOIN, "join_error"),
        (
            "SELECT name FROM singer WHERE age > 30",
            "SELECT name FROM singer WHERE age > (SELECT avg(age) FROM singer)",
            "nesting_setop_error",
        ),
        ("SELECT country FROM singer", "SELECT name FROM singer", "wrong_column"),
        ("SELECT max(age) FROM singer", "SELECT avg(age) FROM singer", "aggregation_error"),
        (
            "SELECT country, count(*) FROM singer GROUP BY name",
            "SELECT country, count(*) FROM singer GROUP BY country",
            "group_by_error",
        ),
        (
            "SELECT name FROM singer WHERE age < 30",
            "SELECT name FROM singer WHERE age > 30",
            "condition_error",
        ),
        (
            "SELECT name FROM singer ORDER BY age ASC LIMIT 1",
            "SELECT name FROM singer ORDER BY age DESC LIMIT 1",
            "order_limit_error",
        ),
    ],
)
def test_compare_finds_the_primary_category(parse, pred: str, gold: str, primary: str) -> None:
    labels = compare(parse(pred), parse(gold))
    assert labels[0] == primary


def test_compare_looks_inside_nested_queries(parse) -> None:  # noqa: ANN001
    gold = "SELECT name FROM singer WHERE age > (SELECT avg(age) FROM singer)"
    pred = "SELECT name FROM singer WHERE age > (SELECT max(age) FROM singer)"
    assert compare(parse(pred), parse(gold)) == ["aggregation_error"]


def test_identical_queries_have_no_differences(parse) -> None:  # noqa: ANN001
    assert compare(parse(JOIN), parse(JOIN)) == []


def test_label_uses_scores_and_execution(parse) -> None:  # noqa: ANN001
    base = {"id": "dev-1", "db_id": "x", "hardness": "easy", "question": "q",
            "gold_sql": "SELECT name FROM singer"}  # fmt: skip

    def runs(db: str, sql: str) -> bool:
        return "bogus" not in sql

    def lab(pred: str, em: int, ex: int) -> str | None:
        row = label({**base, "pred_sql": pred, "em": em, "ex": ex}, parse, runs)
        return row and row["primary"]

    assert lab("SELECT name FROM singer", 1, 1) is None
    assert lab("SELECT name FROM singer", 0, 1) == "semantically_equivalent"
    assert lab("SELECT name FROM singer WHERE name = 'bogus'", 1, 0) == "invalid_sql"
    gold_where = "SELECT name FROM singer WHERE country = 'France'"
    row = label({**base, "gold_sql": gold_where, "pred_sql": gold_where.replace("France", "france"),
                 "em": 1, "ex": 0}, parse, runs)  # fmt: skip
    assert row["primary"] == "value_error"
    assert lab("SELECT nonsense FROM nowhere", 0, 0) == "invalid_sql"
    wrong_join = JOIN.replace("T1.singer_id = T2.singer_id", "T1.singer_id = T2.concert_id")
    row = label({**base, "gold_sql": JOIN, "pred_sql": wrong_join, "em": 1, "ex": 0}, parse, runs)
    assert row["primary"] == "join_error"  # EM passes, but the join columns differ


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT name FROM singer", "simple lookup"),
        ("SELECT count(*) FROM singer", "aggregation"),
        ("SELECT country FROM singer GROUP BY country", "aggregation"),
        (JOIN, "two-table join"),
        (
            JOIN + " JOIN concert AS T3 ON T2.concert_id = T3.concert_id",
            "multi-join",
        ),
        ("SELECT name FROM singer UNION SELECT name FROM stadium", "nested / set-op"),
    ],
)
def test_question_type(parse, sql: str, expected: str) -> None:  # noqa: ANN001
    assert question_type(question_features(parse(sql))) == expected


def test_summary_counts_and_ranker_miss() -> None:
    rows = [
        {"id": "a", "hardness": "easy", "primary": "wrong_table", "labels": ["wrong_table"],
         "ranker_miss": True, "question": "", "gold_sql": "", "pred_sql": ""},
        {"id": "b", "hardness": "hard", "primary": "semantically_equivalent",
         "labels": ["semantically_equivalent"], "ranker_miss": False, "question": "",
         "gold_sql": "", "pred_sql": ""},
    ]  # fmt: skip
    s = summarize_taxonomy(rows, total=10)
    assert s["ex_wrong"] == 1
    assert s["primary"]["wrong_table"] == 1
    assert s["primary"]["semantically_equivalent"] == 1
    assert s["ranker_miss"] == 1
    assert s["primary_by_hardness"]["easy"]["wrong_table"] == 1


def test_review_agreement(tmp_path: Path) -> None:
    path = tmp_path / "manual_review.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "auto_primary", "manual_primary", "second_primary"])
        w.writerows([["a", "wrong_table", "wrong_table", "wrong_table"],
                     ["b", "other", "join_error", "other"],
                     ["c", "other", "", "other"]])  # fmt: skip
    out = review_agreement(path)
    assert out["labelled"] == 2
    assert out["agreement"] == 0.5
    assert out["disagreements"] == {"other -> join_error": 1}
    second = review_agreement(path, column="second_primary")
    assert (second["labelled"], second["agreement"], second["disagreements"]) == (3, 1.0, {})
