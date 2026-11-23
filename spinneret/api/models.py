"""Request / response models for the HTTP API (specs §9)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DatabaseSummary(BaseModel):
    db_id: str
    n_tables: int
    n_columns: int


class ColumnInfo(BaseModel):
    name: str
    type: str
    is_pk: bool


class TableInfo(BaseModel):
    name: str
    columns: list[ColumnInfo]


class ForeignKeyInfo(BaseModel):
    source: str  # table.column
    target: str  # table.column


class SchemaInfo(BaseModel):
    db_id: str
    tables: list[TableInfo]
    foreign_keys: list[ForeignKeyInfo]


class Example(BaseModel):
    question: str
    hardness: str


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


# --- models, query and execution (week 5) --------------------------------------------


class ModelInfo(BaseModel):
    id: str
    label: str
    generator: str
    ranker: str | None
    default: bool
    loaded: bool
    skeleton: bool | None  # None until loaded
    params_m: float | None  # generator + ranker, millions
    dev_em: float | None  # 0–1, from the model's evaluation run
    dev_ex: float | None
    p50_ms: float | None  # laptop GPU, batch 1, full serving path
    p95_ms: float | None
    run: str | None


class Health(BaseModel):
    status: str  # ok | loading | error
    device: str
    databases: int
    models_loaded: list[str]
    error: str | None = None


class QueryOptions(BaseModel):
    model: str | None = None
    beam_size: int | None = Field(None, ge=1, le=16)
    decoding: Literal["eg", "beam"] = "eg"
    max_rows: int | None = Field(None, ge=1, le=500)


class QueryRequest(BaseModel):
    db_id: str
    question: str
    options: QueryOptions = QueryOptions()


class CandidateOut(BaseModel):
    rank: int
    sql: str
    score: float
    status: str  # ok | error | timeout | rejected | not_run


class TableItem(BaseModel):
    name: str
    score: float
    kept: bool


class ColumnItem(BaseModel):
    table: str
    name: str
    score: float
    kept: bool


class SchemaItems(BaseModel):
    tables: list[TableItem]
    columns: list[ColumnItem]


class ExecutionOut(BaseModel):
    status: str  # ok | error | timeout | rejected | failed
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool
    error: str | None


class QueryResponse(BaseModel):
    model: str
    sql: str
    skeleton: str | None
    selected_rank: int
    candidates: list[CandidateOut]
    schema_items: SchemaItems | None
    matched_values: dict[str, list[str]]
    execution: ExecutionOut
    timings_ms: dict[str, float]


class ExecuteRequest(BaseModel):
    db_id: str
    sql: str
    max_rows: int | None = Field(None, ge=1, le=500)


class RunSummary(BaseModel):
    run_id: str
    em: float | None
    ex: float | None
    valid_sql: float | None
    created: str | None


class RunDetail(BaseModel):
    run_id: str
    metrics: dict[str, Any]
    taxonomy: dict[str, Any] | None
    question_types: dict[str, Any] | None
    latency: dict[str, Any] | None
