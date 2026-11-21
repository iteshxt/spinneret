from __future__ import annotations

import sqlite3
from collections import Counter

import pytest

from spinneret.data.normalize import normalize_sql
from spinneret.data.skeleton import skeleton_target, split_output, sql_skeleton
from spinneret.data.spider import database_path, load_examples
from spinneret.db.sqlite import QueryTimeout, connect_readonly, execute


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT count(*) FROM singer", "select count ( * ) from singer"),
        (
            'SELECT Name FROM singer WHERE Country = "France";',
            "select name from singer where country = 'France'",
        ),
        (
            'SELECT name FROM people WHERE name = "O\'Brien"',
            "select name from people where name = 'O''Brien'",
        ),
        (
            "SELECT T2.Name FROM concert AS T1 JOIN stadium AS T2 ON T1.Stadium_ID = T2.Stadium_ID",
            "select t2.name from concert as t1 join stadium as t2 on t1.stadium_id = t2.stadium_id",
        ),
        ("SELECT a FROM t WHERE b != 1 AND c >= 2.5", "select a from t where b != 1 and c >= 2.5"),
        ("SELECT a FROM t WHERE b LIKE '%Smith%'", "select a from t where b like '%Smith%'"),
        ("select  a\n from t   where b < 3", "select a from t where b < 3"),
    ],
)
def test_normalize(sql, expected):
    assert normalize_sql(sql) == expected


@pytest.mark.parametrize(
    ("sql", "skeleton"),
    [
        ("select count ( * ) from singer", "select count ( _ ) from _"),
        ("select name from singer where country = 'France'", "select _ from _ where _ = _"),
        (
            "select t2.name from concert as t1 join stadium as t2 on t1.stadium_id = t2.stadium_id "
            "group by t1.stadium_id order by count ( * ) desc limit 1",
            "select _ from _ join _ on _ = _ group by _ order by count ( _ ) desc limit _",
        ),
        ("select name , age , country from singer", "select _ from _"),
        (
            "select name from singer where age > ( select avg ( age ) from singer )",
            "select _ from _ where _ > ( select avg ( _ ) from _ )",
        ),
        ("select a from t where b = 'x y z'", "select _ from _ where _ = _"),
    ],
)
def test_skeleton(sql, skeleton):
    assert sql_skeleton(sql) == skeleton


def test_skeleton_target_round_trip():
    sql = "select name from singer where country = 'France'"
    assert split_output(skeleton_target(sql)) == ("select _ from _ where _ = _", sql)
    assert split_output(sql) == (None, sql)


def _same_result(a: list, b: list) -> bool:
    return a == b or Counter(map(repr, a)) == Counter(map(repr, b))


@pytest.mark.data
@pytest.mark.slow
def test_normalization_preserves_execution(spider_dir):
    """Gold and normalized gold must return identical results on every Spider query."""
    connections: dict[str, sqlite3.Connection] = {}
    failures, checked, skipped = [], 0, 0
    for source in ("train_spider", "train_others", "dev"):
        for ex in load_examples(spider_dir, source):
            db_id = ex["db_id"]
            if db_id not in connections:
                connections[db_id] = connect_readonly(database_path(spider_dir, db_id))
            conn = connections[db_id]
            try:
                _, gold_rows = execute(conn, ex["query"], timeout_s=10)
            except (QueryTimeout, sqlite3.Error):
                skipped += 1  # gold itself does not run (a few Spider queries are broken)
                continue
            normalized = normalize_sql(ex["query"])
            try:
                _, rows = execute(conn, normalized, timeout_s=20)
            except (QueryTimeout, sqlite3.Error) as exc:
                failures.append((source, ex["query"], normalized, repr(exc)))
                continue
            checked += 1
            if not _same_result(gold_rows, rows):
                failures.append((source, ex["query"], normalized, "different result"))
    for conn in connections.values():
        conn.close()
    print(f"checked {checked}, gold not runnable {skipped}, failures {len(failures)}")
    assert not failures, "\n".join(map(str, failures[:20]))
