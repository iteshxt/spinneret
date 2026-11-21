"""Loading Spider 1.0 files: examples, schemas and database paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spinneret.data.schema import Schema

# Example files and the schema / database locations they use.
SPLIT_FILES: dict[str, str] = {
    "train_spider": "train_spider.json",
    "train_others": "train_others.json",
    "dev": "dev.json",
    "test": "test.json",
}
REQUIRED_FILES = ("tables.json", "train_spider.json", "train_others.json", "dev.json")


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_examples(spider_dir: Path, source: str) -> list[dict[str, Any]]:
    """Load one example file (``train_spider``, ``train_others``, ``dev`` or ``test``)."""
    return load_json(spider_dir / SPLIT_FILES[source])


def load_schemas(spider_dir: Path, *, test: bool = False) -> dict[str, Schema]:
    """All schemas keyed by ``db_id``; ``test=True`` reads ``test_tables.json``."""
    name = "test_tables.json" if test else "tables.json"
    return {entry["db_id"]: Schema.from_spider(entry) for entry in load_json(spider_dir / name)}


def database_path(spider_dir: Path, db_id: str, *, test: bool = False) -> Path:
    folder = "test_database" if test else "database"
    return spider_dir / folder / db_id / f"{db_id}.sqlite"


def missing_files(spider_dir: Path) -> list[str]:
    missing = [name for name in REQUIRED_FILES if not (spider_dir / name).is_file()]
    if not (spider_dir / "database").is_dir():
        missing.append("database/")
    return missing
