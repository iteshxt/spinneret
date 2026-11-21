from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")


@pytest.fixture(scope="module")
def client(spider_dir):
    from fastapi.testclient import TestClient

    from spinneret.api.main import create_app
    from spinneret.config import ServeConfig

    return TestClient(create_app(ServeConfig(spider_dir=spider_dir)))


@pytest.mark.data
def test_health_and_databases(client):
    assert client.get("/api/health").json()["databases"] == 20
    dbs = client.get("/api/databases").json()
    assert len(dbs) == 20
    concert = next(d for d in dbs if d["db_id"] == "concert_singer")
    assert concert == {"db_id": "concert_singer", "n_tables": 4, "n_columns": 21}


@pytest.mark.data
def test_schema_and_examples(client):
    schema = client.get("/api/databases/concert_singer/schema").json()
    assert [t["name"] for t in schema["tables"]] == [
        "stadium", "singer", "concert", "singer_in_concert",
    ]  # fmt: skip
    assert {"source": "singer_in_concert.Singer_ID", "target": "singer.Singer_ID"} in schema[
        "foreign_keys"
    ]
    examples = client.get("/api/databases/concert_singer/examples?limit=3").json()
    assert len(examples) == 3 and examples[0]["hardness"] in ("easy", "medium", "hard", "extra")


@pytest.mark.data
def test_errors(client):
    r = client.get("/api/databases/nope/schema")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    r = client.get("/api/databases/concert_singer/examples?limit=0")
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_input"
