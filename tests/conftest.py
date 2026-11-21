from __future__ import annotations

from pathlib import Path

import pytest

from spinneret.data.schema import Schema

ROOT = Path(__file__).resolve().parents[1]
SPIDER_DIR = ROOT / "data" / "raw" / "spider"
OFFICIAL_EVAL_DIR = ROOT / "third_party" / "test-suite-sql-eval"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if (SPIDER_DIR / "tables.json").is_file():
        return
    skip = pytest.mark.skip(reason="Spider not downloaded — run `uv run spinneret download`")
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip)


# A tables.json-style entry shaped like Spider's concert_singer database.
CONCERT_SINGER = {
    "db_id": "concert_singer",
    "table_names_original": ["stadium", "singer", "concert", "singer_in_concert"],
    "table_names": ["stadium", "singer", "concert", "singer in concert"],
    "column_names_original": [
        [-1, "*"],
        [0, "Stadium_ID"],
        [0, "Location"],
        [0, "Name"],
        [0, "Capacity"],
        [0, "Highest"],
        [0, "Lowest"],
        [0, "Average"],
        [1, "Singer_ID"],
        [1, "Name"],
        [1, "Country"],
        [1, "Song_Name"],
        [1, "Song_release_year"],
        [1, "Age"],
        [1, "Is_male"],
        [2, "concert_ID"],
        [2, "concert_Name"],
        [2, "Theme"],
        [2, "Stadium_ID"],
        [2, "Year"],
        [3, "concert_ID"],
        [3, "Singer_ID"],
    ],
    "column_names": [],
    "column_types": [
        "text",
        "number",
        "text",
        "text",
        "number",
        "number",
        "number",
        "number",
        "number",
        "text",
        "text",
        "text",
        "text",
        "number",
        "others",
        "number",
        "text",
        "text",
        "text",
        "text",
        "number",
        "text",
    ],
    "primary_keys": [1, 8, 15, [20, 21]],
    "foreign_keys": [[18, 1], [21, 8], [20, 15]],
}


@pytest.fixture
def concert_schema() -> Schema:
    return Schema.from_spider(CONCERT_SINGER)


@pytest.fixture(scope="session")
def spider_dir() -> Path:
    return SPIDER_DIR
