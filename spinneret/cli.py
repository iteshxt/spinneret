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
    batch_size: Annotated[
        int | None, typer.Option(help="Override train.per_device_batch_size (e.g. 8 on a 4 GB GPU)")
    ] = None,
    accumulation: Annotated[
        int | None, typer.Option(help="Override train.gradient_accumulation_steps")
    ] = None,
    eval_batch_size: Annotated[
        int | None, typer.Option(help="Override train.eval_batch_size")
    ] = None,
    gradient_checkpointing: Annotated[
        bool, typer.Option(help="Trade compute for memory (e.g. T5-base on a 4 GB GPU)")
    ] = False,
) -> None:
    """Fine-tune a generator (T5 / CodeT5) from a YAML config."""
    from spinneret.model.train import train as run_train

    cfg = load_config(config)
    if max_epochs is not None:
        cfg.train.max_epochs = max_epochs
    if no_hub:
        cfg.hub.push_during_training = False
    for field, value in (
        ("per_device_batch_size", batch_size),
        ("gradient_accumulation_steps", accumulation),
        ("eval_batch_size", eval_batch_size),
    ):
        if value is not None:
            if value < 1:
                raise typer.BadParameter(f"{field} must be at least 1")
            setattr(cfg.train, field, value)
    if gradient_checkpointing:
        cfg.train.gradient_checkpointing = True
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
    top_k_tables: Annotated[int | None, typer.Option(help="Override data.top_k_tables")] = None,
    top_k_columns: Annotated[int | None, typer.Option(help="Override data.top_k_columns")] = None,
    split: Annotated[
        list[str] | None, typer.Option(help="Split(s) to build (default: train, valid, dev)")
    ] = None,
    out: Annotated[Path | None, typer.Option(help="Output folder (default: inputs dir)")] = None,
) -> None:
    """Build ranked inputs (T2/T3) with the config's ranker_checkpoint."""
    from spinneret.ranker.infer import build_ranked

    cfg = load_config(config)
    for field, value in (("top_k_tables", top_k_tables), ("top_k_columns", top_k_columns)):
        if value is not None:
            if value < 1:
                raise typer.BadParameter(f"{field} must be at least 1")
            setattr(cfg.data, field, value)
    splits = tuple(split) if split else ("train", "valid", "dev")
    out_dir = build_ranked(cfg, splits=splits, out_dir=out)
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
    inputs: Annotated[
        Path | None,
        typer.Option(exists=True, file_okay=False, help="Read <split>.jsonl from here instead"),
    ] = None,
) -> None:
    """Generate SQL for a split; writes predictions.jsonl, pred.txt and meta.json."""
    from spinneret.eval.predict import predict as run_predict

    if decoding not in ("greedy", "beam", "eg"):
        raise typer.BadParameter("--decoding must be greedy, beam or eg")
    out_dir = run_predict(checkpoint, split=split, decoding=decoding, beam_size=beam,
                          out_dir=out, batch_size=batch_size, limit=limit,
                          inputs_dir=inputs)  # fmt: skip
    typer.echo(f"Wrote {out_dir}")


@app.command("top-beam")
def top_beam(
    run: Annotated[Path, typer.Option(exists=True, file_okay=False, help="An eg run folder")],
    out: Annotated[Path | None, typer.Option(help="Output folder")] = None,
) -> None:
    """Derive the beam top-1 (no execution) run from an execution-guided run."""
    from spinneret.eval.predict import top_beam_run

    typer.echo(f"Wrote {top_beam_run(run, out)}")


@app.command()
def evaluate(
    run: Annotated[Path, typer.Option(exists=True, file_okay=False, help="reports/<run_id>")],
    spider_dir: Annotated[Path, typer.Option()] = Path("data/raw/spider"),
    etype: Annotated[str, typer.Option(help="all | match | exec")] = "all",
    plug_value: Annotated[
        bool, typer.Option(help="Also report EX with gold values plugged in")
    ] = False,
) -> None:
    """Official exact-set-match and execution accuracy for a predictions run."""
    from spinneret.eval.report import evaluate_run, format_metrics

    metrics = evaluate_run(run, spider_dir, etype=etype, plug_value=plug_value)
    typer.echo(format_metrics(metrics))
    typer.echo(f"Wrote {run / 'metrics.json'}")


@app.command()
def analyze(
    run: Annotated[Path, typer.Option(exists=True, file_okay=False, help="reports/<run_id>")],
    spider_dir: Annotated[Path, typer.Option()] = Path("data/raw/spider"),
    review: Annotated[int, typer.Option(help="Size of the manual-review sample (0 = none)")] = 0,
) -> None:
    """Failure taxonomy and question-type table for a scored run (specs §8.4–8.5)."""
    from spinneret.eval.analysis import QUESTION_TYPES, analyze_run

    out = analyze_run(run, spider_dir, review=review)
    tax, types = out["taxonomy"], out["question_types"]["types"]
    typer.echo(
        f"EX-wrong {tax['ex_wrong']} of {tax['examples']} · ranker_miss {tax['ranker_miss']}"
    )
    for cat, n in tax["primary"].items():
        if n:
            typer.echo(f"  {cat:24s} {n:4d}")
    for t in QUESTION_TYPES:
        v = types[t]
        typer.echo(f"  {t:16s} n={v['count']:4d}  EX {(v['ex'] or 0) * 100:5.1f}%")
    typer.echo(f"Wrote {run}/taxonomy.csv, taxonomy_summary.json, question_types.json")


@app.command("review-agreement")
def review_agreement_cmd(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    column: Annotated[str, typer.Option(help="Column with the hand labels")] = "manual_primary",
) -> None:
    """Agreement between manual and automatic taxonomy labels in manual_review.csv."""
    from spinneret.eval.analysis import review_agreement

    typer.echo(json.dumps(review_agreement(path, column), indent=2))


@app.command()
def latency(
    checkpoint: Annotated[str, typer.Option(help="Checkpoint folder or hub:<user>/<repo>")],
    out: Annotated[Path, typer.Option(help="Folder for latency.json (e.g. reports/<run_id>)")],
    limit: Annotated[int | None, typer.Option(help="Only N questions (seeded sample)")] = None,
    beam: Annotated[int | None, typer.Option(help="Beam size (default: from config)")] = None,
    device: Annotated[str | None, typer.Option(help="cuda | cpu (default: auto)")] = None,
) -> None:
    """Per-question latency (values, ranking, generation, execution), batch 1."""
    from spinneret.eval.latency import STAGES, benchmark, write_latency

    result = benchmark(checkpoint, limit=limit, beam_size=beam, device=device)
    for stage in STAGES:
        p = result["latency_ms"][stage]
        typer.echo(f"{stage:11s} p50 {p['p50']:8.1f} ms   p95 {p['p95']:8.1f} ms")
    typer.echo(f"inputs identical to prepared data: {result['inputs_match_prepared'] * 100:.1f}%")
    typer.echo(f"Wrote {write_latency(result, out)}")


@app.command()
def summarize(
    reports: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path("reports"),
    out: Annotated[Path, typer.Option(help="Output folder")] = Path("reports/summary"),
) -> None:
    """Collect all evaluation runs into summary.json and summary.md (specs §8.3)."""
    from spinneret.eval.summary import write_summary

    j, m = write_summary(reports, out)
    typer.echo(f"Wrote {j} and {m}")


@app.command("model-cards")
def model_cards(
    config: Annotated[Path, typer.Option("--config", "-c", dir_okay=False)] = Path(
        "configs/serve.yaml"
    ),
    out: Annotated[Path, typer.Option(help="Folder for the rendered cards")] = Path(
        "docs/model-cards"
    ),
    push: Annotated[
        bool, typer.Option(help="Also upload each card as README.md to its Hub repo")
    ] = False,
) -> None:
    """Render model cards for the served models and ranker from the evaluation reports."""
    from spinneret.cards import push_cards, write_cards
    from spinneret.config import load_serve_config

    cards = write_cards(load_serve_config(config), out)
    for ref, path in cards.items():
        typer.echo(f"{ref} -> {path}")
    if push:
        for repo in push_cards(cards):
            typer.echo(f"uploaded README.md to {repo}")


@app.command()
def serve(
    config: Annotated[Path | None, typer.Option("--config", "-c", dir_okay=False)] = None,
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
) -> None:
    """Start the HTTP API and the built web app (FastAPI + uvicorn); default configs/serve.yaml."""
    import logging

    import uvicorn

    from spinneret.api.main import create_app
    from spinneret.config import load_serve_config

    if config is None and Path("configs/serve.yaml").is_file():
        config = Path("configs/serve.yaml")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")
    uvicorn.run(create_app(load_serve_config(config)), host=host, port=port)


if __name__ == "__main__":
    app()
