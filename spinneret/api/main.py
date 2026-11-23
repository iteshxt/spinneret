"""FastAPI application (specs §9): databases, models, query, execute, metrics, and the
built frontend at ``/`` when ``web/dist`` exists (design §13)."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from spinneret.api.deps import Busy, Loader, Registry, load_translators
from spinneret.api.models import (
    CandidateOut,
    ColumnInfo,
    ColumnItem,
    DatabaseSummary,
    Example,
    ExecuteRequest,
    ExecutionOut,
    ForeignKeyInfo,
    Health,
    ModelInfo,
    QueryRequest,
    QueryResponse,
    RunDetail,
    RunSummary,
    SchemaInfo,
    SchemaItems,
    TableInfo,
    TableItem,
)
from spinneret.config import ServeConfig
from spinneret.data.hardness import hardness
from spinneret.data.spider import database_path, load_examples, load_schemas
from spinneret.db.executor import ExecutionResult, Executor
from spinneret.decoding.execution_guided import select

MAX_QUESTION, MAX_SQL = 500, 5000


class ApiError(HTTPException):
    def __init__(self, status: int, message: str, code: str | None = None) -> None:
        super().__init__(status_code=status, detail=message)
        self.code = code


def _load_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _execution(result: ExecutionResult, status: str | None = None) -> ExecutionOut:
    return ExecutionOut(
        status=status or result.status, columns=result.columns, rows=result.rows,
        row_count=result.row_count, truncated=result.truncated, error=result.error,
    )  # fmt: skip


def create_app(
    cfg: ServeConfig, *, loader: Loader | None = None, background: bool = True
) -> FastAPI:
    spider_dir = Path(cfg.spider_dir)
    schemas = load_schemas(spider_dir)
    dev = load_examples(spider_dir, "dev")
    if cfg.databases == "dev":
        keep = {ex["db_id"] for ex in dev}
        schemas = {k: v for k, v in schemas.items() if k in keep}
    examples: dict[str, list[Example]] = defaultdict(list)
    for ex in dev:
        if ex["db_id"] in schemas:
            examples[ex["db_id"]].append(
                Example(question=" ".join(ex["question"].split()), hardness=hardness(ex["sql"]))
            )
    limits = cfg.execution
    executor = Executor(lambda db_id: database_path(spider_dir, db_id),
                        timeout_s=limits.timeout_s, max_rows=limits.max_rows,
                        max_value_bytes=limits.max_value_bytes,
                        max_result_bytes=limits.max_result_bytes)  # fmt: skip
    registry = Registry(cfg, schemas, loader or load_translators)
    reports = Path(cfg.reports_dir)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        registry.start(background=background)  # schemas and metrics are served meanwhile
        yield

    app = FastAPI(title="Spinneret", version="0.1.0", lifespan=lifespan)
    app.state.registry = registry
    app.add_middleware(CORSMiddleware, allow_origins=cfg.cors_origins,
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])  # fmt: skip

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        code = getattr(exc, "code", None) or {
            404: "not_found", 400: "invalid_input", 503: "model_loading"
        }.get(exc.status_code, "error")  # fmt: skip
        body = {"error": {"code": code, "message": str(exc.detail)}}
        return JSONResponse(status_code=exc.status_code, content=body)

    def need_db(db_id: str) -> None:
        if db_id not in schemas:
            raise ApiError(404, f"unknown database {db_id!r}")

    # --- status, databases ------------------------------------------------------------

    @app.get("/api/health", response_model=Health)
    def health() -> Health:
        status = "error" if registry.error else ("ok" if registry.ready else "loading")
        return Health(status=status, device=registry.device, databases=len(schemas),
                      models_loaded=sorted(registry.translators), error=registry.error)  # fmt: skip

    @app.get("/api/databases", response_model=list[DatabaseSummary])
    def databases() -> list[DatabaseSummary]:
        return [
            DatabaseSummary(db_id=db_id, n_tables=len(s.tables), n_columns=s.n_columns)
            for db_id, s in sorted(schemas.items())
        ]

    @app.get("/api/databases/{db_id}/schema", response_model=SchemaInfo)
    def schema(db_id: str) -> SchemaInfo:
        need_db(db_id)
        s = schemas[db_id]
        return SchemaInfo(
            db_id=db_id,
            tables=[
                TableInfo(
                    name=t.name,
                    columns=[
                        ColumnInfo(name=c.name, type=c.type, is_pk=c.is_pk) for c in t.columns
                    ],
                )
                for t in s.tables
            ],
            foreign_keys=[
                ForeignKeyInfo(
                    source=f"{fk.src.table}.{fk.src.name}", target=f"{fk.dst.table}.{fk.dst.name}"
                )
                for fk in s.foreign_keys
            ],
        )

    @app.get("/api/databases/{db_id}/examples", response_model=list[Example])
    def db_examples(db_id: str, limit: int = 6) -> list[Example]:
        need_db(db_id)
        if not 1 <= limit <= 50:
            raise ApiError(400, "limit must be between 1 and 50")
        return examples[db_id][:limit]

    # --- models, query, execute -------------------------------------------------------

    @app.get("/api/models", response_model=list[ModelInfo])
    def models() -> list[ModelInfo]:
        out = []
        for m in cfg.models:
            run = reports / m.run if m.run else None
            metrics = _load_json(run / "metrics.json") if run else None
            lat = _load_json(run / "latency.json") if run else None
            total = ((lat or {}).get("latency_ms") or {}).get("total") or {}
            params = None
            if lat and lat.get("generator_params_m") is not None:
                params = round(lat["generator_params_m"] + (lat.get("ranker_params_m") or 0), 1)
            t = registry.translators.get(m.id)
            out.append(ModelInfo(
                id=m.id, label=m.label, generator=m.generator,
                ranker=cfg.rankers.get(m.ranker) if m.ranker else None,
                default=m.id == cfg.default_model, loaded=t is not None,
                skeleton=t.uses_skeleton if t is not None else None, params_m=params,
                dev_em=(metrics or {}).get("exact_match", {}).get("all"),
                dev_ex=(metrics or {}).get("execution", {}).get("all"),
                p50_ms=total.get("p50"), p95_ms=total.get("p95"), run=m.run,
            ))  # fmt: skip
        return out

    @app.post("/api/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        question = " ".join(req.question.split())
        if not question:
            raise ApiError(400, "the question is empty")
        if len(question) > MAX_QUESTION:
            raise ApiError(400, f"the question is longer than {MAX_QUESTION} characters")
        model_id = req.options.model or cfg.default_model or next(iter(registry.models), None)
        if model_id not in registry.models:
            raise ApiError(400, f"unknown model {model_id!r}")
        need_db(req.db_id)
        if registry.error:
            raise ApiError(503, f"models failed to load: {registry.error}")
        if not registry.ready or model_id not in registry.translators:
            raise ApiError(503, "models are still loading")
        translator = registry.translators[model_id]
        max_rows = req.options.max_rows or limits.max_rows

        t0 = time.perf_counter()
        try:
            with registry.gpu():
                tr = translator.translate(question, req.db_id, req.options.beam_size)
        except Busy:
            raise ApiError(503, "the demo is busy answering other questions; try again in a "
                           "few seconds", code="busy") from None  # fmt: skip
        t1 = time.perf_counter()
        results: dict[int, ExecutionResult] = {}

        def run(i: int, sql: str) -> ExecutionResult:
            results[i] = executor.run(req.db_id, sql, max_rows=max_rows)
            return results[i]

        sqls = [c.sql for c in tr.candidates]
        if req.options.decoding == "eg":
            pos = iter(range(len(sqls)))
            sel = select(sqls, lambda sql: run(next(pos), sql))
            chosen, statuses = sel.index, sel.statuses
            execution = _execution(results[chosen], None if sel.executed else "failed")
        else:  # plain beam: the top candidate, whatever happens
            res = run(0, sqls[0])
            chosen, statuses = 0, [res.status] + ["not_run"] * (len(sqls) - 1)
            execution = _execution(res)
        t2 = time.perf_counter()

        timings = {**tr.timings_ms, "execution": round((t2 - t1) * 1000, 1),
                   "total": round((t2 - t0) * 1000, 1)}  # fmt: skip
        items = None
        if tr.tables is not None and tr.columns is not None:
            items = SchemaItems(
                tables=[TableItem(name=t.name, score=t.score, kept=t.kept) for t in tr.tables],
                columns=[ColumnItem(table=c.table or "", name=c.name, score=c.score, kept=c.kept)
                         for c in tr.columns],
            )  # fmt: skip
        best = tr.candidates[chosen]
        return QueryResponse(
            model=model_id, sql=best.sql, skeleton=best.skeleton, selected_rank=chosen,
            candidates=[CandidateOut(rank=c.rank, sql=c.sql, score=round(c.score, 4),
                                     status=statuses[c.rank]) for c in tr.candidates],
            schema_items=items, matched_values=tr.matched_values, execution=execution,
            timings_ms=timings,
        )  # fmt: skip

    @app.post("/api/execute", response_model=ExecutionOut)
    def execute(req: ExecuteRequest) -> ExecutionOut:
        if not req.sql.strip():
            raise ApiError(400, "the SQL is empty")
        if len(req.sql) > MAX_SQL:
            raise ApiError(400, f"the SQL is longer than {MAX_SQL:,} characters")
        need_db(req.db_id)
        return _execution(
            executor.run(req.db_id, req.sql, max_rows=req.max_rows or limits.max_rows)
        )

    # --- evaluation results -----------------------------------------------------------

    def run_dirs() -> list[str]:
        if not reports.is_dir():
            return []
        return sorted(str(p.parent.relative_to(reports)) for p in reports.rglob("metrics.json"))

    @app.get("/api/metrics", response_model=list[RunSummary])
    def metrics_list() -> list[RunSummary]:
        out = []
        for run_id in run_dirs():
            m = _load_json(reports / run_id / "metrics.json") or {}
            out.append(RunSummary(
                run_id=run_id, em=(m.get("exact_match") or {}).get("all"),
                ex=(m.get("execution") or {}).get("all"), valid_sql=m.get("valid_sql_rate"),
                created=(m.get("meta") or {}).get("created"),
            ))  # fmt: skip
        return out

    @app.get("/api/metrics/{run_id:path}", response_model=RunDetail)
    def metrics_detail(run_id: str) -> RunDetail:
        if run_id not in run_dirs():  # only known run folders, never arbitrary paths
            raise ApiError(404, f"unknown run {run_id!r}")
        d = reports / run_id
        return RunDetail(
            run_id=run_id, metrics=_load_json(d / "metrics.json") or {},
            taxonomy=_load_json(d / "taxonomy_summary.json"),
            question_types=_load_json(d / "question_types.json"),
            latency=_load_json(d / "latency.json"),
        )  # fmt: skip

    @app.get("/api/summary")
    def summary() -> dict[str, Any]:
        s = _load_json(reports / "summary" / "summary.json")
        if s is None:
            raise ApiError(404, "no summary yet: run `spinneret summarize`")
        return s

    # --- built frontend ---------------------------------------------------------------

    web = Path(cfg.web_dir)
    if (web / "index.html").is_file():
        root = web.resolve()

        @app.get("/{path:path}", include_in_schema=False)
        def frontend(path: str) -> FileResponse:
            if path.startswith("api/"):
                raise ApiError(404, f"unknown endpoint /{path}")
            target = (root / path).resolve()
            if path and target.is_file() and target.is_relative_to(root):
                return FileResponse(target)
            return FileResponse(root / "index.html")  # client-side routes

    return app
