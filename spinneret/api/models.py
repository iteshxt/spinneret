"""Request / response models for the HTTP API (specs §9)."""

from __future__ import annotations

from pydantic import BaseModel


class Health(BaseModel):
    status: str
    databases: int
    models_loaded: list[str]


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
