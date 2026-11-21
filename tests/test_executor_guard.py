from __future__ import annotations

import sqlite3

import pytest

from spinneret.db.executor import ExecutionResult, Executor, results_equal
from spinneret.db.guard import GuardError, check_sql
from spinneret.db.sqlite import connect_readonly
from spinneret.decoding.execution_guided import select


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE singer",
        "INSERT INTO singer VALUES (1)",
        "SELECT 1; DELETE FROM singer",
        "ATTACH DATABASE 'x.db' AS x",
        "PRAGMA table_info(singer)",
        "WITH x AS (SELECT 1) DELETE FROM singer",
        "REPLACE INTO singer VALUES (1)",
        "UPDATE singer SET age = 1",
        "",
    ],
)
def test_guard_rejects(sql):
    with pytest.raises(GuardError):
        check_sql(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT name FROM singer;",
        "WITH x AS (SELECT age FROM singer) SELECT max(age) FROM x",
        "SELECT created_at, update_date FROM logs",
        "SELECT replace(name, 'a', 'b') FROM singer",
        "select count ( * ) from singer where name like '%drop table%'",
    ],
)
def test_guard_allows(sql):
    assert check_sql(sql) == sql.rstrip(";")


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "t.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE n (x int, b blob)")
    conn.executemany("INSERT INTO n VALUES (?, ?)", [(i, b"\x00\x01") for i in range(1000)])
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def executor(db):
    return Executor(lambda _db_id: db, timeout_s=1.0, max_rows=500)


def test_executor_row_cap_and_blobs(executor):
    result = executor.run("t", "SELECT x, b FROM n")
    assert result.status == "ok"
    assert result.row_count == 500 and result.truncated
    assert result.rows[0] == [0, "<blob 2 bytes>"]
    small = executor.run("t", "SELECT x FROM n WHERE x < 3")
    assert small.rows == [[0], [1], [2]] and not small.truncated


def test_executor_statuses(executor):
    assert executor.run("t", "SELECT nope FROM n").status == "error"
    assert executor.run("t", "DELETE FROM n").status == "rejected"
    slow = "WITH RECURSIVE c(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM c) SELECT count(*) FROM c"
    assert executor.run("t", slow).status == "timeout"


def test_connection_is_read_only_even_without_guard(db):
    conn = connect_readonly(db)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM n")
    conn.close()


def test_results_equal():
    assert results_equal([[1], [2]], [[2], [1]], order_matters=False)
    assert not results_equal([[1], [2]], [[2], [1]], order_matters=True)
    assert not results_equal([[1], [1]], [[1]], order_matters=False)


def test_execution_guided_selection():
    def run(sql: str) -> ExecutionResult:
        return ExecutionResult(status="ok" if sql == "good" else "error")

    chosen = select(["bad", "good", "also good"], run)
    assert chosen.index == 1 and chosen.executed
    assert chosen.statuses == ["error", "ok", "not_run"]
    none = select(["bad", "worse"], run)
    assert none.index == 0 and not none.executed
