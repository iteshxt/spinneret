"""Static copy of the read-only API for the hosted website (design §13).

The hosted site is static files; questions go to the live server on the owner's laptop.
When that server is off, the site still shows the databases, schemas, example questions,
models and evaluation results from these files. They are the API's own responses (with no
models loaded), so the two cannot drift apart unless the data or reports change, and
``tests/test_snapshot.py`` checks that they have not.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spinneret.config import ServeConfig

EXAMPLES_LIMIT = 50  # the API's maximum; the site asks for fewer and slices


def build_snapshot(cfg: ServeConfig) -> dict[str, Any]:
    """Return ``{relative file name: JSON body}`` for every read-only endpoint."""
    from fastapi.testclient import TestClient

    from spinneret.api.main import create_app

    app = create_app(cfg, loader=lambda _cfg, _schemas: {}, background=False)
    with TestClient(app) as client:

        def get(path: str) -> Any:
            r = client.get(path)
            r.raise_for_status()
            return r.json()

        files: dict[str, Any] = {
            "models.json": get("/api/models"),
            "databases.json": get("/api/databases"),
            "summary.json": get("/api/summary"),
        }
        for db in files["databases.json"]:
            db_id = db["db_id"]
            files[f"schema/{db_id}.json"] = get(f"/api/databases/{db_id}/schema")
            files[f"examples/{db_id}.json"] = get(
                f"/api/databases/{db_id}/examples?limit={EXAMPLES_LIMIT}"
            )
    return files


def dump(body: Any) -> str:
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")) + "\n"


def write_snapshot(cfg: ServeConfig, out: Path) -> list[Path]:
    """Write the snapshot under ``out``, replacing any older one."""
    files = build_snapshot(cfg)
    for old in sorted(out.rglob("*.json")) if out.is_dir() else []:
        old.unlink()
    written = []
    for name, body in files.items():
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dump(body), encoding="utf-8")
        written.append(path)
    return written
