from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "web" / "public" / "snapshot"


@pytest.mark.data
def test_snapshot_matches_the_api(spider_dir):
    """The hosted site's static copy is what the API returns today."""
    from spinneret.api.snapshot import build_snapshot, dump
    from spinneret.config import load_serve_config

    cfg = load_serve_config(ROOT / "configs" / "serve.yaml").model_copy(
        update={"spider_dir": spider_dir, "reports_dir": ROOT / "reports"}
    )
    expected = {name: dump(body) for name, body in build_snapshot(cfg).items()}
    on_disk = {str(p.relative_to(SNAPSHOT)): p.read_text("utf-8") for p in SNAPSHOT.rglob("*.json")}
    assert sorted(on_disk) == sorted(expected), "run `uv run spinneret snapshot`"
    stale = [name for name in expected if on_disk[name] != expected[name]]
    assert not stale, f"stale snapshot files {stale}: run `uv run spinneret snapshot`"
