from __future__ import annotations

import pytest

from spinneret.data.labels import gold_items
from spinneret.data.spider import load_examples, load_schemas


def _sql(select, table_units, *, conds=None, where=None, **extra):
    base = {
        "select": select,
        "from": {"table_units": table_units, "conds": conds or []},
        "where": where or [],
        "groupBy": [],
        "having": [],
        "orderBy": [],
        "limit": None,
        "intersect": None,
        "union": None,
        "except": None,
    }
    base.update(extra)
    return base


def test_schema_from_spider(concert_schema):
    assert [t.name for t in concert_schema.tables] == [
        "stadium",
        "singer",
        "concert",
        "singer_in_concert",
    ]
    assert concert_schema.n_columns == 21
    assert concert_schema.column(0) is None  # "*"
    assert concert_schema.column(9).key == "singer.name"
    # composite primary key given as a nested list
    assert concert_schema.column(20).is_pk and concert_schema.column(21).is_pk
    assert [fk.render() for fk in concert_schema.foreign_keys] == [
        "concert.stadium_id = stadium.stadium_id",
        "singer_in_concert.singer_id = singer.singer_id",
        "singer_in_concert.concert_id = concert.concert_id",
    ]


def test_labels_ignore_star(concert_schema):
    # SELECT count(*) FROM singer
    sql = _sql([False, [[3, [0, [0, 0, False], None]]]], [["table_unit", 1]])
    assert gold_items(sql, concert_schema) == (["singer"], {})


def test_labels_join_and_subquery(concert_schema):
    # SELECT T2.Name FROM singer_in_concert AS T1 JOIN singer AS T2 ON T1.Singer_ID = T2.Singer_ID
    # WHERE T2.Age > (SELECT avg(Age) FROM singer)
    sub = _sql([False, [[5, [0, [0, 13, False], None]]]], [["table_unit", 1]])
    sql = _sql(
        [False, [[0, [0, [0, 9, False], None]]]],
        [["table_unit", 3], ["table_unit", 1]],
        conds=[[False, 2, [0, [0, 21, False], None], [0, 8, False], None]],
        where=[[False, 3, [0, [0, 13, False], None], sub, None]],
    )
    tables, columns = gold_items(sql, concert_schema)
    assert tables == ["singer", "singer_in_concert"]
    assert columns == {"singer": ["singer_id", "name", "age"], "singer_in_concert": ["singer_id"]}


def test_labels_set_operation(concert_schema):
    # SELECT Name FROM stadium EXCEPT SELECT T2.Name FROM concert AS T1 JOIN stadium AS T2 ...
    other = _sql([False, [[0, [0, [0, 3, False], None]]]], [["table_unit", 2], ["table_unit", 0]])
    sql = _sql([False, [[0, [0, [0, 3, False], None]]]], [["table_unit", 0]], **{"except": other})
    tables, _ = gold_items(sql, concert_schema)
    assert tables == ["stadium", "concert"]


@pytest.mark.data
def test_labels_cover_every_spider_example(spider_dir):
    schemas = load_schemas(spider_dir)
    for source in ("train_spider", "train_others", "dev"):
        for ex in load_examples(spider_dir, source):
            tables, columns = gold_items(ex["sql"], schemas[ex["db_id"]])
            assert tables, f"{source}: no tables for {ex['query']}"
            assert set(columns) <= set(tables)
