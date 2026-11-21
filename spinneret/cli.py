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


@app.command()
def train(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, dir_okay=False)],
    resume: Annotated[
        str | None,
        typer.Option(help="Resume from 'local' (checkpoints/) or 'hub' (last-checkpoint)"),
    ] = None,
    max_epochs: Annotated[int | None, typer.Option(help="Override train.max_epochs")] = None,
    no_hub: Annotated[bool, typer.Option(help="Do not push to the Hugging Face Hub")] = False,
) -> None:
    """Fine-tune a generator (T5 / CodeT5) from a YAML config."""
    from spinneret.model.train import train as run_train

    cfg = load_config(config)
    if max_epochs is not None:
        cfg.train.max_epochs = max_epochs
    if no_hub:
        cfg.hub.push_during_training = False
    if resume not in (None, "local", "hub"):
        raise typer.BadParameter("--resume must be 'local' or 'hub'")
    run_train(cfg, resume=resume)


@app.command("train-ranker")
def train_ranker_cmd(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, dir_okay=False)],
    resume: Annotated[str | None, typer.Option(help="'local' or 'hub'")] = None,
    max_epochs: Annotated[int | None, typer.Option(help="Override train.max_epochs")] = None,
    no_hub: Annotated[bool, typer.Option(help="Do not push to the Hugging Face Hub")] = False,
) -> None:
    """Train the schema-item ranker (RoBERTa cross-encoder)."""
    from spinneret.ranker.train import train_ranker

    cfg = load_config(config)
    if max_epochs is not None:
        cfg.train.max_epochs = max_epochs
    if no_hub:
        cfg.hub.push_during_training = False
    train_ranker(cfg, resume=resume)


@app.command("build-ranked")
def build_ranked_cmd(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, dir_okay=False)],
) -> None:
    """Build ranked inputs (T2/T3) with the config's ranker_checkpoint."""
    from spinneret.ranker.infer import build_ranked

    cfg = load_config(config)
    out_dir = build_ranked(cfg)
    stats = json.loads((out_dir / "stats.json").read_text())
    for split, s in stats["splits"].items():
        typer.echo(f"{split:6s} coverage@({cfg.data.top_k_tables},{cfg.data.top_k_columns}) "
                   f"{s['coverage'] * 100:.1f}% of {s['examples']}")  # fmt: skip
    typer.echo(f"Wrote {out_dir}")


@app.command()
def predict(
    checkpoint: Annotated[str, typer.Option(help="Checkpoint folder or hub:<user>/<repo>")],
    split: Annotated[str, typer.Option()] = "dev",
    decoding: Annotated[str, typer.Option(help="greedy | beam | eg")] = "eg",
    beam: Annotated[int | None, typer.Option(help="Beam size (default: from config)")] = None,
    out: Annotated[
        Path | None, typer.Option(help="Output folder (default reports/<run_id>)")
    ] = None,
    batch_size: Annotated[int, typer.Option()] = 16,
    limit: Annotated[int | None, typer.Option(help="Only the first N examples")] = None,
) -> None:
    """Generate SQL for a split; writes predictions.jsonl, pred.txt and meta.json."""
    from spinneret.eval.predict import predict as run_predict

    if decoding not in ("greedy", "beam", "eg"):
        raise typer.BadParameter("--decoding must be greedy, beam or eg")
    out_dir = run_predict(checkpoint, split=split, decoding=decoding, beam_size=beam,
                          out_dir=out, batch_size=batch_size, limit=limit)  # fmt: skip
    typer.echo(f"Wrote {out_dir}")


@app.command()
def evaluate(
    run: Annotated[Path, typer.Option(exists=True, file_okay=False, help="reports/<run_id>")],
    spider_dir: Annotated[Path, typer.Option()] = Path("data/raw/spider"),
    etype: Annotated[str, typer.Option(help="all | match | exec")] = "all",
) -> None:
    """Official exact-set-match and execution accuracy for a predictions run."""
    from spinneret.eval.report import evaluate_run, format_metrics

    metrics = evaluate_run(run, spider_dir, etype=etype)
    typer.echo(format_metrics(metrics))
    typer.echo(f"Wrote {run / 'metrics.json'}")


@app.command()
def serve(
    config: Annotated[Path | None, typer.Option("--config", "-c", dir_okay=False)] = None,
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
) -> None:
    """Start the HTTP API (FastAPI + uvicorn)."""
    import uvicorn

    from spinneret.api.main import create_app
    from spinneret.config import load_serve_config

    uvicorn.run(create_app(load_serve_config(config)), host=host, port=port)


if __name__ == "__main__":
    app()
