"""``spinneret`` command-line entry point (specs §2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from spinneret.config import DEFAULT_MODEL, load_config

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Spinneret text-to-SQL.")

ConfigOpt = Annotated[
    Path | None, typer.Option("--config", "-c", exists=True, dir_okay=False, help="YAML config")
]


@app.command()
def download(
    raw_dir: Annotated[Path, typer.Option(help="Where to put the Spider data")] = Path("data/raw"),
    zip_path: Annotated[
        Path | None, typer.Option("--zip", exists=True, help="Use an already-downloaded zip")
    ] = None,
    force: Annotated[bool, typer.Option(help="Re-download and re-extract")] = False,
) -> None:
    """Download, verify (SHA-256) and extract Spider 1.0 into RAW_DIR/spider."""
    from spinneret.data.download import download_spider

    spider_dir = download_spider(raw_dir, zip_path=zip_path, force=force)
    typer.echo(f"Spider ready in {spider_dir}")


def _token_counter(model_name: str, processed_dir: Path):  # noqa: ANN202
    from spinneret.model.tokenizer import load_tokenizer, read_added_tokens

    tokenizer = load_tokenizer(model_name, read_added_tokens(processed_dir, model_name))
    tokenizer.model_max_length = 1_000_000  # we only count; silence the >512 warning

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=True).input_ids)

    return count


@app.command()
def prepare(
    config: ConfigOpt = None,
    no_tokenizer: Annotated[
        bool, typer.Option(help="Skip token counting and the 512-token budget")
    ] = False,
    quiet: Annotated[bool, typer.Option(help="No progress bars")] = False,
) -> None:
    """Build data/processed/{train,valid,dev}.jsonl, the value cache and stats.json."""
    from spinneret.data.prepare import prepare as run_prepare

    cfg = load_config(config)
    counter = None if no_tokenizer else _token_counter(cfg.model.name, cfg.data.processed_dir)
    stats = run_prepare(cfg, count_tokens=counter, progress=not quiet)
    for split, s in stats["splits"].items():
        typer.echo(
            f"{split:6s} {s['questions']:5d} questions · {s['databases']:3d} DBs · "
            f"hardness {s['hardness']} · truncated {s['truncated']}"
        )
    typer.echo(f"Wrote {cfg.data.processed_dir}/{{train,valid,dev}}.jsonl and stats.json")


@app.command("audit-tokenizer")
def audit_tokenizer(
    model: Annotated[str, typer.Option(help="Hugging Face model id")] = DEFAULT_MODEL,
    processed_dir: Annotated[Path, typer.Option()] = Path("data/processed"),
) -> None:
    """Find characters MODEL's tokenizer cannot encode and record tokens to add."""
    from spinneret.data.prepare import read_jsonl
    from spinneret.model.tokenizer import ADDED_TOKENS_FILE, audit

    records = []
    for split in ("train", "valid", "dev"):
        path = processed_dir / f"{split}.jsonl"
        if not path.is_file():
            raise typer.BadParameter(f"{path} not found — run `spinneret prepare` first")
        records.extend(read_jsonl(path))

    report = audit(model, [r["target"] for r in records], [r["source"] for r in records])
    out = processed_dir / ADDED_TOKENS_FILE
    existing = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
    existing[model] = report
    out.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")

    typer.echo(f"{model}: missing in targets {report['missing_in_targets']}")
    typer.echo(f"  missing in inputs {report['missing_in_sources']}")
    typer.echo(f"  tokens to add {report['added_tokens']}")
    typer.echo(
        "  target round-trip failures: "
        f"{report['target_roundtrip_failures_before']} before → "
        f"{report['target_roundtrip_failures_after']} after"
    )
    typer.echo(f"Wrote {out}")


if __name__ == "__main__":
    app()
