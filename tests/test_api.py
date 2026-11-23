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


# --- models, query, execute, metrics (stub translators, no GPU) -------------------------


class StubTranslator:
    """Two fixed beams: a broken one first, so execution-guided selection must skip it."""

    params_m = 1.0

    def __init__(self, ranked: bool) -> None:
        self.ranked = ranked
        self.uses_skeleton = ranked

    def translate(self, question, db_id, k=None):
        from spinneret.model.translator import Candidate, SchemaItem, Translation

        beams = ["select nope from singer", "select name from singer where country = 'France'"]
        cands = [Candidate(rank=i, sql=s, skeleton="select _ from _" if self.ranked else None,
                           score=-float(i)) for i, s in enumerate(beams[: k or 2])]  # fmt: skip
        tables = columns = None
        if self.ranked:
            tables = [SchemaItem("singer", 0.99, True), SchemaItem("stadium", 0.02, False)]
            columns = [SchemaItem("Country", 0.98, True, table="singer")]
        timings = {"values": 1.0, "ranking": 2.0, "inference": 3.0}
        return Translation(candidates=cands, matched_values={"singer.country": ["France"]},
                           tables=tables, columns=columns, timings_ms=timings)  # fmt: skip


@pytest.fixture(scope="module")
def served(spider_dir, tmp_path_factory):
    from fastapi.testclient import TestClient

    from spinneret.api.main import create_app
    from spinneret.config import ServeConfig, ServeModel

    reports = tmp_path_factory.mktemp("reports")
    run = reports / "t1-run"
    run.mkdir()
    (run / "metrics.json").write_text(
        '{"exact_match": {"all": 0.5}, "execution": {"all": 0.6}, "valid_sql_rate": 0.7}'
    )
    (run / "latency.json").write_text(
        '{"generator_params_m": 60.5, "latency_ms": {"total": {"p50": 370.0, "p95": 740.0}}}'
    )
    cfg = ServeConfig(
        spider_dir=spider_dir, default_model="t3", reports_dir=reports,
        web_dir=tmp_path_factory.mktemp("no-web"), rankers={"base": "stub"},
        models=[ServeModel(id="t1", label="Baseline", generator="stub", run="t1-run"),
                ServeModel(id="t3", label="Spinneret", generator="stub", ranker="base")],
    )  # fmt: skip
    loader = lambda cfg, schemas: {"t1": StubTranslator(False), "t3": StubTranslator(True)}  # noqa: E731
    with TestClient(create_app(cfg, loader=loader, background=False)) as c:
        yield c


@pytest.mark.data
def test_models(served):
    health = served.get("/api/health").json()
    assert health["status"] == "ok" and health["models_loaded"] == ["t1", "t3"]
    models = {m["id"]: m for m in served.get("/api/models").json()}
    assert models["t3"]["default"] and models["t3"]["skeleton"] is True
    t1 = models["t1"]
    assert (t1["dev_ex"], t1["p50_ms"], t1["params_m"], t1["ranker"]) == (0.6, 370.0, 60.5, None)
    assert models["t3"]["dev_ex"] is None and models["t3"]["ranker"] == "stub"


@pytest.mark.data
def test_cors_only_for_configured_origins(served):
    ok = served.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert ok.headers["access-control-allow-origin"] == "http://localhost:5173"
    other = served.get("/api/health", headers={"Origin": "https://elsewhere.example"})
    assert "access-control-allow-origin" not in other.headers


def test_gpu_queue_limit():
    from spinneret.api.deps import Busy, Registry
    from spinneret.config import ServeConfig

    registry = Registry(ServeConfig(queue_limit=1), {}, lambda cfg, schemas: {})
    with registry.gpu(), pytest.raises(Busy), registry.gpu():
        pass
    with registry.gpu():  # the slot is free again
        pass


@pytest.mark.data
def test_query_busy(served, monkeypatch):
    from contextlib import contextmanager

    from spinneret.api.deps import Busy

    @contextmanager
    def full():
        raise Busy
        yield

    monkeypatch.setattr(served.app.state.registry, "gpu", full)
    r = served.post("/api/query", json={"db_id": "concert_singer", "question": "How many?"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "busy"


@pytest.mark.data
def test_query_execution_guided(served):
    q = {"db_id": "concert_singer", "question": "  Which singers are from France? "}
    r = served.post("/api/query", json=q)
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "t3" and body["selected_rank"] == 1
    assert [c["status"] for c in body["candidates"]] == ["error", "ok"]
    assert body["execution"]["status"] == "ok" and body["execution"]["columns"] == ["Name"]
    assert body["skeleton"] == "select _ from _"
    assert body["schema_items"]["tables"][0] == {"name": "singer", "score": 0.99, "kept": True}
    assert set(body["timings_ms"]) == {"values", "ranking", "inference", "execution", "total"}


@pytest.mark.data
def test_query_t1_and_failures(served):
    q = {"db_id": "concert_singer", "question": "names?"}
    body = served.post("/api/query", json={**q, "options": {"model": "t1"}}).json()
    assert body["schema_items"] is None and body["skeleton"] is None
    # plain beam: the broken top candidate is returned as-is
    body = served.post("/api/query", json={**q, "options": {"decoding": "beam"}}).json()
    assert body["selected_rank"] == 0 and body["execution"]["status"] == "error"
    assert [c["status"] for c in body["candidates"]] == ["error", "not_run"]
    # no candidate runs: top one, marked failed
    body = served.post("/api/query", json={**q, "options": {"beam_size": 1}}).json()
    assert body["execution"]["status"] == "failed" and body["execution"]["error"]


@pytest.mark.data
@pytest.mark.parametrize(
    ("payload", "status"),
    [
        ({"db_id": "concert_singer", "question": "   "}, 400),
        ({"db_id": "concert_singer", "question": "x" * 501}, 400),
        ({"db_id": "concert_singer", "question": "hi", "options": {"model": "t9"}}, 400),
        ({"db_id": "nope", "question": "hi"}, 404),
        ({"db_id": "concert_singer", "question": 5}, 422),
    ],
)
def test_query_errors(served, payload, status):
    r = served.post("/api/query", json=payload)
    assert r.status_code == status
    if status != 422:
        assert r.json()["error"]["code"] == {400: "invalid_input", 404: "not_found"}[status]


@pytest.mark.data
def test_execute(served):
    ok = served.post(
        "/api/execute",
        json={"db_id": "concert_singer", "sql": "select name from singer", "max_rows": 2},
    )
    assert ok.json()["status"] == "ok" and ok.json()["truncated"] and ok.json()["row_count"] == 2
    bad = served.post("/api/execute", json={"db_id": "concert_singer", "sql": "drop table singer"})
    assert bad.json()["status"] == "rejected"
    assert served.post("/api/execute", json={"db_id": "concert_singer",
                                             "sql": "x" * 5001}).status_code == 400  # fmt: skip


@pytest.mark.data
def test_metrics(served):
    runs = served.get("/api/metrics").json()
    assert runs == [{"run_id": "t1-run", "em": 0.5, "ex": 0.6, "valid_sql": 0.7, "created": None}]
    detail = served.get("/api/metrics/t1-run").json()
    assert detail["latency"]["generator_params_m"] == 60.5 and detail["taxonomy"] is None
    assert served.get("/api/metrics/../../etc").status_code == 404
    assert served.get("/api/summary").status_code == 404


@pytest.mark.data
def test_query_while_loading(spider_dir):
    import threading

    from fastapi.testclient import TestClient

    from spinneret.api.main import create_app
    from spinneret.config import ServeConfig, ServeModel

    gate = threading.Event()

    def slow(cfg, schemas):
        gate.wait(5)
        return {"t1": StubTranslator(False)}

    cfg = ServeConfig(spider_dir=spider_dir, models=[ServeModel(id="t1", label="B", generator="s")])
    with TestClient(create_app(cfg, loader=slow)) as c:
        r = c.post("/api/query", json={"db_id": "concert_singer", "question": "hi"})
        assert r.status_code == 503 and r.json()["error"]["code"] == "model_loading"
        assert c.get("/api/health").json()["status"] == "loading"
        gate.set()
