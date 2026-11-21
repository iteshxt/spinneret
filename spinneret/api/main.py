"""FastAPI application (specs §9). Week 2: health, databases, schema, examples."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from spinneret.api.models import (
    ColumnInfo,
    DatabaseSummary,
    Example,
    ForeignKeyInfo,
    Health,
    SchemaInfo,
    TableInfo,
)
from spinneret.config import ServeConfig
from spinneret.data.hardness import hardness
from spinneret.data.spider import load_examples, load_schemas

DEV_ORIGIN = "http://localhost:5173"


class NotFound(HTTPException):
    def __init__(self, what: str) -> None:
        super().__init__(status_code=404, detail=what)


def create_app(cfg: ServeConfig) -> FastAPI:
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

    app = FastAPI(title="Spinneret", version="0.1.0")
    app.add_middleware(CORSMiddleware, allow_origins=[DEV_ORIGIN], allow_methods=["*"],
                       allow_headers=["*"])  # fmt: skip

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        code = {404: "not_found", 400: "invalid_input", 503: "model_loading"}.get(
            exc.status_code, "error"
        )
        body = {"error": {"code": code, "message": str(exc.detail)}}
        return JSONResponse(status_code=exc.status_code, content=body)

    @app.get("/api/health", response_model=Health)
    def health() -> Health:
        return Health(status="ok", databases=len(schemas), models_loaded=[])

    @app.get("/api/databases", response_model=list[DatabaseSummary])
    def databases() -> list[DatabaseSummary]:
        return [
            DatabaseSummary(db_id=db_id, n_tables=len(s.tables), n_columns=s.n_columns)
            for db_id, s in sorted(schemas.items())
        ]

    @app.get("/api/databases/{db_id}/schema", response_model=SchemaInfo)
    def schema(db_id: str) -> SchemaInfo:
        s = schemas.get(db_id)
        if s is None:
            raise NotFound(f"unknown database {db_id!r}")
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
        if db_id not in schemas:
            raise NotFound(f"unknown database {db_id!r}")
        if not 1 <= limit <= 50:
            raise HTTPException(status_code=400, detail="limit must be between 1 and 50")
        return examples[db_id][:limit]

    return app
