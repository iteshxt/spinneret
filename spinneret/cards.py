"""Model cards for the served models and the ranker (specs §5.8), rendered from the
committed evaluation reports and each checkpoint's ``config.yaml`` / ``state.json``.

Cards are written locally (``docs/model-cards/``) for review; uploading them to the Hub
is a separate, explicit step (``spinneret model-cards --push``) that leaves repo
visibility unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spinneret.config import ServeConfig, ServeModel, load_config
from spinneret.hub import is_hub_ref, repo_id, resolve_checkpoint

REPO_URL = "https://github.com/iteshxt/spinneret"
LEVELS = ("easy", "medium", "hard", "extra", "all")
EXAMPLE_Q = "What is the average, minimum, and maximum age of all singers from France?"
EXAMPLE_SQL = "select avg ( age ) , min ( age ) , max ( age ) from singer where country = 'France'"
EXAMPLE_SKELETON = "select avg ( _ ) , min ( _ ) , max ( _ ) from _ where _ = _"
EXAMPLE_FULL = (
    f"{EXAMPLE_Q} | concert_singer | stadium : stadium_id , location , name , capacity , "
    "highest , lowest , average | singer : singer_id , name , country ( France ) , song_name , "
    "song_release_year , age , is_male | concert : concert_id , concert_name , theme , "
    "stadium_id , year | singer_in_concert : concert_id , singer_id"
)
EXAMPLE_RANKED = (
    f"{EXAMPLE_Q} | concert_singer | singer : age , country ( France ) , singer_id , name , "
    "is_male | singer_in_concert : singer_id , concert_id | stadium : average , stadium_id , "
    "highest , lowest , name | concert : year , stadium_id , concert_id , concert_name , theme | "
    "fk : concert.stadium_id = stadium.stadium_id , singer_in_concert.singer_id = "
    "singer.singer_id , singer_in_concert.concert_id = concert.concert_id"
)
LIMITATIONS = """\
- English questions in the style of Spider; other phrasing and domains are less reliable.
- Errors on unseen schemas are mostly wrong joins and wrong columns; queries joining three or
  more tables, and nested or set queries, are much weaker than simple lookups (see the
  per-type table above).
- Values must match the database's spelling; the model copies values it is shown, and numbers
  written in words ("100 million") are often wrong.
- Execution-guided decoding needs the database: without it, take the top beam (lower EX).
- Intended for read-only querying and research; check generated SQL before relying on it."""
LICENCE = """\
- Weights: fine-tuned from {base} ({base_licence}).
- Training data: [Spider](https://yale-lily.github.io/spider) (Yu et al., EMNLP 2018),
  CC BY-SA 4.0 — this model is a derivative and is shared under the same licence.
- Code, training configs and full evaluation: [{repo}]({repo})."""


def _load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.1f}"


def _secs(ms: float | None) -> str:
    return "—" if ms is None else f"{ms / 1000:.2f} s"


def _frontmatter(fields: dict[str, Any]) -> str:
    lines = ["---"]
    for k, v in fields.items():
        if isinstance(v, list):
            lines.append(f"{k}:")
            lines += [f"  - {x}" for x in v]
        else:
            lines.append(f"{k}: {v}")
    return "\n".join([*lines, "---", ""])


def _checkpoint_files(ref: str) -> tuple[Any, dict[str, Any]]:
    folder = resolve_checkpoint(ref)
    return load_config(folder / "config.yaml"), _load(folder / "state.json") or {}


def generator_card(
    model: ServeModel, reports: Path, ranker_ref: str | None, *, checkpoint: tuple | None = None
) -> str:
    """Card for one served generator. ``checkpoint`` = (config, state) to skip the download."""
    cfg, state = checkpoint or _checkpoint_files(model.generator)
    data = cfg.data
    run = reports / model.run if model.run else None
    metrics = (_load(run / "metrics.json") if run else None) or {}
    lat = (_load(run / "latency.json") if run else None) or {}
    qtypes = ((_load(run / "question_types.json") if run else None) or {}).get("types", {})
    em, ex = metrics.get("exact_match", {}), metrics.get("execution", {})
    total = (lat.get("latency_ms") or {}).get("total") or {}
    ranked = data.schema_input == "ranked"
    skeleton = data.target_format == "skeleton_sql"
    base = cfg.model.name
    this = repo_id(model.generator) if is_hub_ref(model.generator) else model.generator
    ranker = repo_id(ranker_ref) if ranker_ref and is_hub_ref(ranker_ref) else ranker_ref
    best = state.get("best") or {}

    out = [_frontmatter({
        "license": "cc-by-sa-4.0", "language": "en", "library_name": "transformers",
        "pipeline_tag": "text2text-generation", "base_model": base,
        "datasets": ["xlangai/spider"], "tags": ["text-to-sql", "spider", "t5", "sql"],
    })]  # fmt: skip
    title = model.label if model.label.startswith("Spinneret") else f"Spinneret {model.label}"
    out += [f"# {title}", ""]
    what = (
        f"`{base}` fine-tuned on Spider to turn an English question and a database schema "
        "into SQLite SQL."
    )
    if ranked:
        what += (
            f" It reads only the tables and columns kept by the schema ranker "
            f"([{ranker}](https://huggingface.co/{ranker})), top {data.top_k_tables} tables × "
            f"{data.top_k_columns} columns."
        )
    else:
        what += " It reads the whole schema."
    if skeleton:
        what += " It writes the query's skeleton first, then the full query."
    out += [what, "", f"Part of [Spinneret]({REPO_URL}): Baseline (T5-small), Lite (T5-small + "
            "ranker) and Spinneret (T5-base + ranker).", ""]  # fmt: skip

    beam = (metrics.get("meta") or {}).get("beam_size", model.beam_size)
    out += ["## Results on Spider dev", "",
            f"1,034 questions on 20 databases not seen in training; official test-suite "
            f"evaluation; execution-guided beam search with {beam} beams (the first candidate "
            "that runs is kept).", "",
            "| | " + " | ".join(LEVELS) + " |", "|---" * (len(LEVELS) + 1) + "|",
            "| Exact match % | " + " | ".join(_pct(em.get(lv)) for lv in LEVELS) + " |",
            "| Execution accuracy % | " + " | ".join(_pct(ex.get(lv)) for lv in LEVELS) + " |",
            "", f"Valid (executable) SQL: {_pct(metrics.get('valid_sql_rate'))}%.", ""]  # fmt: skip
    if qtypes:
        out += ["| Question type | n | EX % |", "|---|---|---|"]
        out += [f"| {t} | {v['count']} | {_pct(v['ex'])} |" for t, v in qtypes.items()]
        out += [""]
    if lat:
        size = lat.get("generator_params_m")
        rsize = lat.get("ranker_params_m")
        mem = lat.get("peak_gpu_memory_mb")
        out += ["## Size and speed", "",
                f"- Parameters: {size}M" + (f" (+ {rsize}M ranker)" if rsize else ""),
                f"- Latency per question, full pipeline at batch 1 on a laptop "
                f"{lat.get('device', 'GPU')}: median {_secs(total.get('p50'))}, "
                f"p95 {_secs(total.get('p95'))}",
                f"- Peak GPU memory: {mem / 1024:.2f} GB" if mem else "", ""]  # fmt: skip

    source = EXAMPLE_RANKED if ranked else EXAMPLE_FULL
    target = f"{EXAMPLE_SKELETON} | {EXAMPLE_SQL}" if skeleton else EXAMPLE_SQL
    out += ["## Input and output", "",
            "Input: `question | database | table : column , column ( matched value ) | … "
            + ("| fk : a.x = b.y , …`" if ranked else "`")
            + ". Names are lower-cased; database values that match words in the question are "
            "shown in brackets next to their column.", "",
            "```text", source, "```", "", "Output:", "", "```text", target, "```", ""]  # fmt: skip
    out += ["## How to use", "",
            "The easiest way is the Spinneret code, which builds the input (value matching"
            + (", schema ranking" if ranked else "") + "), decodes and runs the SQL:", "",
            "```bash", f"git clone {REPO_URL} && cd spinneret && uv sync --extra train",
            "uv run spinneret download", "uv run spinneret serve   # demo on :8000",
            "```", "",
            "With `transformers` alone, given an input string built as above:", "",
            "```python", "from transformers import AutoModelForSeq2SeqLM, AutoTokenizer",
            f'tok = AutoTokenizer.from_pretrained("{this}")',
            f'model = AutoModelForSeq2SeqLM.from_pretrained("{this}")',
            "ids = tok(source, return_tensors=\"pt\", truncation=True, max_length=512)",
            f"out = model.generate(**ids, num_beams={beam}, num_return_sequences={beam}, "
            "max_new_tokens=256)",
            "candidates = tok.batch_decode(out, skip_special_tokens=True)",
            *(['sql = candidates[0].split(" | ", 1)[-1]  # drop the skeleton'] if skeleton else []),
            "```", ""]  # fmt: skip

    others = "including" if data.include_train_others else "without"
    where = (metrics.get("meta") or {}).get("trained_on") or (
        "free Colab T4 GPUs and a laptop GTX 1650, resuming through Hub checkpoints"
    )
    train = cfg.train
    out += ["## Training", "",
            f"- Data: Spider `train_spider` (7,000 questions) {others} `train_others` (1,659), "
            f"minus a validation set of {int(data.validation_fraction * 100)}% of the training "
            "databases (split by database).",
            f"- Optimiser: {train.optimizer}, learning rate {train.learning_rate}, "
            f"effective batch {train.per_device_batch_size * train.gradient_accumulation_steps}, "
            f"early stopping on validation "
            f"{train.select_best_by.replace('valid_', '').upper()}.",
            f"- Best epoch: {state.get('best_epoch', '—')} (validation EM "
            f"{_pct(best.get('valid_em'))}%, EX {_pct(best.get('valid_ex'))}%).",
            f"- Hardware: {where}; the full config is `config.yaml` in this repo.", ""]  # fmt: skip
    out += ["## Limitations", "", LIMITATIONS, ""]
    licence = "Apache-2.0"
    out += ["## Licence and attribution", "",
            LICENCE.format(base=f"`{base}`", base_licence=licence, repo=REPO_URL), ""]  # fmt: skip
    return "\n".join(line for line in out if line is not None)


def ranker_card(summary: dict[str, Any] | None, state: dict[str, Any]) -> str:
    best = state.get("best") or {}
    e5 = (summary or {}).get("e5_ranker_k") or {}
    out = [_frontmatter({
        "license": "cc-by-sa-4.0", "language": "en", "base_model": "FacebookAI/roberta-base",
        "datasets": ["xlangai/spider"], "tags": ["text-to-sql", "spider", "schema-linking"],
    })]  # fmt: skip
    out += ["# Spinneret — schema ranker (RoBERTa-base)", "",
            "A cross-encoder that reads a question together with a database schema and scores "
            "every table and column for whether the SQL answer needs it. Spinneret's Lite and "
            "Spinneret models only see the top 4 tables × 5 columns it keeps, which shortens "
            "their input and removes distracting schema items.", "",
            f"Part of [Spinneret]({REPO_URL}). Use it through the Spinneret code "
            "(`SchemaRanker` in `spinneret/ranker/infer.py`): the classification heads "
            "(`heads.pt`) are not a standard `transformers` head.", "",
            "## Results", "",
            f"- Validation: table AUC {best.get('table_auc', 0):.3f}, column AUC "
            f"{best.get('column_auc', 0):.3f}; coverage (every needed item kept) at top "
            f"({best.get('k_tables', 4)}, {best.get('k_columns', 5)}): "
            f"{_pct(best.get('valid_coverage'))}%.",
            f"- Spider dev coverage at (4, 5): {_pct((e5.get('4x5') or {}).get('dev_coverage'))}%.",
            ""]  # fmt: skip
    if e5:
        out += ["Dev coverage % by top-k (tables × columns):", "",
                "| tables \\ columns | 4 | 5 | 6 |", "|---|---|---|---|"]  # fmt: skip
        for k1 in (3, 4, 5):
            cells = [_pct((e5.get(f"{k1}x{k2}") or {}).get("dev_coverage")) for k2 in (4, 5, 6)]
            out.append(f"| {k1} | " + " | ".join(cells) + " |")
        out += [""]
    out += ["## Training", "",
            "- Data: Spider train (incl. `train_others`), labels = the tables and columns used "
            "by the gold SQL; validation split by database.",
            f"- Best epoch: {state.get('best_epoch', '—')}; selection by validation coverage.",
            "", "## Licence and attribution", "",
            LICENCE.format(base="`FacebookAI/roberta-base`", base_licence="MIT", repo=REPO_URL),
            ""]  # fmt: skip
    return "\n".join(out)


def write_cards(cfg: ServeConfig, out_dir: Path) -> dict[str, Path]:
    """Render every served model's card and each ranker's; returns {checkpoint ref: path}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = Path(cfg.reports_dir)
    written: dict[str, Path] = {}

    def save(ref: str, text: str) -> None:
        name = (repo_id(ref) if is_hub_ref(ref) else Path(ref).name).split("/")[-1]
        path = out_dir / f"{name}.md"
        path.write_text(text, encoding="utf-8")
        written[ref] = path

    for m in cfg.models:
        rref = cfg.rankers.get(m.ranker) if m.ranker else None
        save(m.generator, generator_card(m, reports, rref))
    summary = _load(reports / "summary" / "summary.json")
    for ref in dict.fromkeys(cfg.rankers.values()):
        state = _load(resolve_checkpoint(ref) / "state.json") or {}
        save(ref, ranker_card(summary, state))
    return written


def push_cards(cards: dict[str, Path]) -> list[str]:
    """Upload each card as README.md to its Hub repo (visibility is not changed)."""
    from huggingface_hub import HfApi

    api = HfApi()
    done = []
    for ref, path in cards.items():
        if not is_hub_ref(ref):
            continue
        api.upload_file(path_or_fileobj=str(path), path_in_repo="README.md",
                        repo_id=repo_id(ref), commit_message="model card")  # fmt: skip
        done.append(repo_id(ref))
    return done
