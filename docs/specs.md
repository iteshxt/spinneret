# Spinneret — Technical Specification

| | |
|---|---|
| **Status** | Draft v3 — tiers T1 Baseline (T5-small) · T2 Lite (T5-small + ranker) · T3 Spinneret (T5-base + ranker); RoBERTa-base ranker; Colab CLI training; Hugging Face Hub |
| **Companion docs** | [prd.md](prd.md) (what and why) · [design.md](design.md) (architecture + UI) |
| **Scope** | Everything needed to build, train, evaluate, publish and serve Spinneret: environment, data, models, decoding, execution, evaluation, Colab workflow, Hub, API, frontend, CLI, tests. |

Conventions: **MUST** = required for the P0 build; **SHOULD** = P1; **MAY** = P2/stretch. Requirement IDs (FR*, G*) and tiers (T1–T3) refer to [prd.md](prd.md).

---

## 1. Environment and tooling

### 1.1 Hardware

| Machine | Spec | Role |
|---|---|---|
| Laptop | NVIDIA GTX 1650, 4 GB VRAM, Turing (no tensor cores); 16 GB RAM; 12 CPU threads; ~57 GB free disk | Development, evaluation, demo serving (all three tiers); fallback for T1 training |
| Google Colab free tier (verified 2026-11-21) | NVIDIA Tesla T4, 15 GB VRAM; 12 GB RAM; ~113 GB disk; Python 3.13; time-limited, reclaimable sessions | All training (ranker, T1, T2, T3), launched from the laptop with the Colab CLI (§5.9) |

### 1.2 Software

| Component | Choice | Notes |
|---|---|---|
| Python | **3.12** via `uv` | System Python is 3.14; ML wheels are most reliable on 3.12. `uv python pin 3.12`. |
| Package manager | `uv` (`pyproject.toml` + `uv.lock`) | `uv sync`, `uv run …` |
| Deep learning | PyTorch (CUDA build) + Hugging Face `transformers`, `accelerate`, `sentencepiece` | Exact versions pinned in `uv.lock`. |
| Model hub | `huggingface_hub` (Python API + `hf` CLI) | push/pull checkpoints, model cards; owner logged in as `iteshxt` |
| Remote training | Google Colab CLI (`colab`, `googlecolab/google-colab-cli`, installed with `uv tool`) | sessions, file upload/download, remote exec; Linux/macOS only; OAuth token in the OS keyring |
| SQL tooling | `sqlparse` (statement guard, normalization, skeletons), `nltk` (needed by Spider eval scripts), `rapidfuzz` (value matching) | |
| API | `fastapi`, `uvicorn`, `pydantic` v2 | |
| CLI | `typer` | single `spinneret` entry point |
| Config | YAML (`pyyaml`) validated into pydantic models | |
| Logging | TensorBoard (`runs/`) | no account needed |
| Quality | `ruff` (lint + format), `pytest` | |
| Frontend | Node 24, Vite, React, TypeScript, Tailwind CSS, TanStack Query, React Router, CodeMirror 6 (`@uiw/react-codemirror` + `@codemirror/lang-sql`), Recharts | versions pinned in `web/package-lock.json` |
| Frontend tests | Vitest + React Testing Library; Playwright smoke test (MAY) | |

### 1.3 Repository layout

```
spinneret/
├── docs/                      prd.md · specs.md · design.md · report/ (after the project)
├── configs/                   t5_small.yaml · ranker.yaml · t5_small_ranked.yaml · t5_base_ranked.yaml
│                              codet5_small.yaml · codet5_base_ranked.yaml · ranker_large.yaml · serve.yaml
├── spinneret/                 Python package
│   ├── cli.py                 Typer entry point
│   ├── config.py              pydantic config models
│   ├── hub.py                 Hugging Face Hub push/pull + model cards
│   ├── data/
│   │   ├── spider.py          load Spider JSON + tables.json
│   │   ├── schema.py          Schema / Table / Column dataclasses
│   │   ├── split.py           train/validation split by database
│   │   ├── labels.py          gold schema-item labels from the parsed `sql` field
│   │   ├── serialize.py       question + (full or ranked) schema → model input
│   │   ├── values.py          database-value matching (BRIDGE-style)
│   │   ├── normalize.py       SQL target normalization
│   │   ├── skeleton.py        SQL → skeleton; parse "skeleton | sql" outputs
│   │   └── prepare.py         builds data/processed/*.jsonl + stats
│   ├── ranker/
│   │   ├── dataset.py         question + schema pairs, token spans per item, chunking
│   │   ├── model.py           RoBERTa cross-encoder with table/column heads
│   │   ├── train.py           training loop (focal loss, coverage-based selection)
│   │   └── infer.py           SchemaRanker.rank(question, db_id) → scored items
│   ├── model/
│   │   ├── tokenizer.py       tokenizer audit + added tokens
│   │   ├── train.py           Seq2SeqTrainer wrapper (local or Colab, Hub checkpointing)
│   │   ├── translator.py      Translator protocol, HFTranslator, RankedTranslator
│   │   └── ensemble.py        EnsembleTranslator (MAY)
│   ├── decoding/
│   │   └── execution_guided.py
│   ├── db/
│   │   ├── executor.py        read-only SQLite execution
│   │   └── guard.py           SQL statement guard
│   ├── eval/
│   │   ├── predict.py         run a model over a split → predictions.jsonl
│   │   ├── official.py        wrapper around Spider evaluation scripts
│   │   ├── coverage.py        ranker AUC + coverage@k
│   │   ├── taxonomy.py        failure categorization
│   │   └── analysis.py        per-difficulty / per-question-type / ablation tables
│   └── api/
│       ├── main.py            FastAPI app factory
│       ├── routes.py
│       ├── models.py          request/response pydantic models
│       └── deps.py            model registry + schema registry singletons
├── third_party/
│   └── test-suite-sql-eval/   git submodule (Apache-2.0), official evaluation
├── scripts/
│   └── colab/                 Colab CLI workflow (§5.9)
│       ├── start.sh           colab new --gpu T4 → ship code + HF login → bootstrap.sh on the VM
│       ├── bootstrap.sh       (runs on the VM) uv, Python 3.12, deps, download + prepare Spider
│       ├── train.sh           launch a spinneret command in the background on the VM (nohup)
│       ├── logs.sh            tail the remote training log
│       └── stop.sh            colab stop + confirm nothing is still assigned
├── web/                       React app (Vite)
├── tests/                     pytest suites
├── data/                      (git-ignored) raw/spider/ · processed/ · cache/
├── checkpoints/               (git-ignored) one folder per run
├── runs/                      (git-ignored) TensorBoard logs
├── reports/                   one folder per evaluation run (small JSON committed)
├── pyproject.toml · uv.lock · README.md · .gitignore
```

---

## 2. CLI

All stages run through one entry point: `uv run spinneret <command>`.

| Command | Purpose | Key options |
|---|---|---|
| `download` | Fetch the Spider zip, verify its SHA-256, unzip into `data/raw/spider/`, check expected files | `--zip PATH` (use a manually downloaded zip), `--force` |
| `prepare` | Build processed JSONL splits, schema-item labels, value cache, dataset statistics | `--config`, `--no-tokenizer`, `--quiet` |
| `audit-tokenizer` | Report characters/strings a tokenizer cannot round-trip | `--model` |
| `train-ranker` | Train the schema-item ranker | `--config configs/ranker.yaml`, `--resume local\|hub` |
| `train` | Fine-tune a generator (T1, T2, T3, CodeT5 variants) | `--config`, `--resume local\|hub` |
| `build-ranked` | Build ranked inputs for every split with a trained ranker | `--ranker`, `--config` |
| `predict` | Generate SQL for a split | `--model-config` (or `--checkpoint` + `--ranker`), `--split dev\|valid\|test`, `--decoding greedy\|beam\|eg`, `--beam`, `--top-k-tables`, `--top-k-columns` |
| `evaluate` | Official EM/EX (+ test-suite if available) and ranker coverage on a predictions run | `--run reports/<run_id>` |
| `analyze` | Failure taxonomy + per-difficulty + per-question-type + ablation tables | `--run reports/<run_id>` (or several) |
| `push` | Upload a checkpoint + generated model card to the Hugging Face Hub | `--checkpoint`, `--repo <user>/<name>`, `--public` |
| `serve` | Start the FastAPI app | `--config configs/serve.yaml`, `--host`, `--port` |

---

## 3. Configuration

One YAML file per experiment; the file is copied into the checkpoint folder (and the Hub repo) so every checkpoint knows how it was made.

| Config | Tier | Model | Schema input | Target | Priority |
|---|---|---|---|---|---|
| `t5_small.yaml` | **T1 Baseline** | `google-t5/t5-small` | full | SQL | P0 |
| `ranker.yaml` | ranker for T2/T3 | `FacebookAI/roberta-base` | full (names + values) | item labels | P0 |
| `t5_small_ranked.yaml` | **T2 Lite** | `google-t5/t5-small` | ranked top-k + FKs | skeleton \| SQL | P0 |
| `t5_base_ranked.yaml` | **T3 Spinneret** | `google-t5/t5-base` | ranked top-k + FKs | skeleton \| SQL | P0 |
| `ranker_large.yaml` | ranker upgrade (Gate A) | `FacebookAI/roberta-large` | full (names + values) | item labels | conditional |
| `codet5_small.yaml` | T1 fallback (Gate B) | `Salesforce/codet5-small` | full | SQL | conditional |
| `codet5_base_ranked.yaml` | T3 fallback (Gate B) | `Salesforce/codet5-base` | ranked top-k + FKs | skeleton \| SQL | conditional |

All are trained on the Colab T4 through the workflow in §5.9; T1 can also train on the laptop.

Example — the T3 config:

```yaml
# configs/t5_base_ranked.yaml
run_name: t5-base-ranked
seed: 42

model:
  name: google-t5/t5-base           # or Salesforce/codet5-base
  added_tokens: auto                # "auto" = use tokenizer audit result

data:
  spider_dir: data/raw/spider
  processed_dir: data/processed
  include_train_others: true        # decided: train on 8,659 questions
  validation_fraction: 0.10         # of train_spider *databases*
  db_content: true                  # value matching
  max_values_per_column: 2
  schema: ranked                    # full | ranked
  ranker_checkpoint: hub:iteshxt/spinneret-ranker-roberta-base
  top_k_tables: 4
  top_k_columns: 5
  include_foreign_keys: true
  target_format: skeleton_sql       # sql | skeleton_sql
  max_source_len: 512
  max_target_len: 256

train:
  optimizer: adafactor              # adafactor | adamw
  learning_rate: 1.0e-4             # sweep per tier, see §5.2
  lr_schedule: linear               # constant | linear
  warmup_ratio: 0.03
  per_device_batch_size: 8
  gradient_accumulation_steps: 4    # effective batch 32
  max_epochs: 60
  precision: fp32                   # T5 is unstable in fp16
  gradient_checkpointing: false     # enable if OOM
  eval_every_epochs: 1
  eval_decoding: greedy             # fast validation decoding
  early_stopping_patience: 5        # in evaluations
  select_best_by: valid_ex          # tie-break: valid_em

hub:
  repo: iteshxt/spinneret-t5-base-ranked
  push_during_training: true        # hub_strategy="checkpoint" → resumable on Colab
  private_during_training: true

decoding:
  beam_size: 8
  mode: eg                          # greedy | beam | eg
```

---

## 4. Data

### 4.1 Source: Spider 1.0

| Item | Value |
|---|---|
| Total | 10,181 questions · 5,693 unique SQL queries · 200 databases · 138 domains |
| `train_spider.json` | 7,000 questions (≈140 databases) — **training** |
| `train_others.json` | 1,659 questions from 6 older datasets (Restaurants, GeoQuery, Scholar, Academic, IMDB, Yelp) — **training** (decided; PRD Q2) |
| `dev.json` | 1,034 questions on 20 databases unseen in training — **main benchmark** |
| Test set | 2,147 questions on 40 databases — now public (the Spider site announced it as released when the leaderboard closed in Feb 2024); the download includes `test.json`, `test_tables.json`, `test_gold.sql` and `test_database/`; used once at the end (MAY) |
| Schemas | `tables.json` (and `test_tables.json` for test) |
| Databases | `database/<db_id>/<db_id>.sqlite` (SQLite 3) |
| Gold files | `dev_gold.sql`, `train_gold.sql` — lines of `SQL<TAB>db_id`; regenerated from JSON if missing |
| Licence | CC BY-SA 4.0 — **never committed**; `spinneret download` fetches it |

**Download:** the official Google Drive file from the Spider site (id `1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J`, fetched over HTTPS from `drive.usercontent.google.com`): `spider_data.zip`, 205,800,266 bytes, SHA-256 `00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b` — verified before extraction, and identical to the Hugging Face mirror `HAL-9001/spider-databases`. If Google Drive is unreachable, download the zip by hand (or from the mirror) and pass it with `--zip`; the checksum still applies. Extraction skips `__MACOSX`/`.DS_Store` entries and rejects paths that escape the target folder. The Hugging Face `xlangai/spider` dataset is **not** sufficient — it has the question/SQL pairs but no databases.

**Example record (`dev.json`):**

| Field | Example | Used for |
|---|---|---|
| `db_id` | `concert_singer` | schema + database lookup |
| `question` | `How many singers do we have?` | model input |
| `query` | `SELECT count(*) FROM singer` | gold SQL (target, evaluation) |
| `question_toks`, `query_toks`, `query_toks_no_value` | token lists | not used |
| `sql` | parsed SQL dict | difficulty level, schema-item labels, taxonomy |

**`tables.json` entry fields:** `db_id`, `table_names_original`, `table_names`, `column_names_original` (list of `[table_index, name]`, index −1 = `*`), `column_names`, `column_types`, `primary_keys` (column indices), `foreign_keys` (pairs of column indices). Spinneret always uses the `*_original` names, because those are what SQL refers to.

### 4.2 Splits

| Split | Built from | Purpose |
|---|---|---|
| `train` | `train_spider.json` (minus validation databases) + all of `train_others.json` | fine-tuning (ranker and generators) |
| `valid` | ~10% of `train_spider` **databases** (all their questions), chosen by seeded shuffle of sorted `db_id`s until ≥ 10% of `train_spider` questions are covered | checkpoint selection, LR sweep, early stopping |
| `dev` | `dev.json` | reported results — **never used for tuning** |
| `test` | Spider test (MAY) | one final run |

Splitting by database (not by question) keeps validation cross-domain, like dev. Validation databases come only from `train_spider`, so validation looks like Spider rather than like the six older datasets.

### 4.3 Difficulty levels

Computed with the official `eval_hardness` logic from Spider's `evaluation.py` (the `Evaluator` class), using three counts on the parsed SQL:

- **component1** — WHERE, GROUP BY, ORDER BY, LIMIT present (1 each) + (number of FROM tables − 1) + number of `OR` + number of `LIKE` conditions
- **component2** — number of nested sub-queries and INTERSECT / UNION / EXCEPT parts
- **others** — +1 each for: more than one aggregate, more than one SELECT column, more than one WHERE condition, more than one GROUP BY column

| Level | Rule (verbatim logic) |
|---|---|
| easy | c1 ≤ 1 and others = 0 and c2 = 0 |
| medium | (others ≤ 2 and c1 ≤ 1 and c2 = 0) or (c1 ≤ 2 and others < 2 and c2 = 0) |
| hard | (others > 2 and c1 ≤ 2 and c2 = 0) or (2 < c1 ≤ 3 and others ≤ 2 and c2 = 0) or (c1 ≤ 1 and others = 0 and c2 ≤ 1) |
| extra | everything else |

`spinneret/data/hardness.py` ports this logic; a test checks it against the vendored official script on every train, `train_others` and dev example (all agree). Dev distribution, as computed by `prepare`: **248 easy / 446 medium / 174 hard / 166 extra** (matches the commonly reported figures).

### 4.4 Model input serialization

Format (UnifiedSKG / Shaw et al. style), all schema names lower-cased, single spaces:

```
<question> | <db_id> | <table> : <col> , <col> ( <value> , <value> ) , <col> | <table> : …  [ | fk : <t.c> = <t.c> , … ]
```

Example (`concert_singer`, full schema, values shown only where the question mentions one):

```
Which singers are from France? | concert_singer | stadium : stadium_id , location , name , capacity , highest , lowest , average | singer : singer_id , name , country ( France ) , song_name , song_release_year , age , is_male | concert : concert_id , concert_name , theme , stadium_id , year | singer_in_concert : concert_id , singer_id
```

Rules:

1. Question text is kept as written (case preserved), stripped, internal whitespace collapsed.
2. **Full schema (T1):** tables in `tables.json` order; columns in their original order; the `*` column is omitted.
3. **Ranked schema (T2, T3):** tables and columns chosen and ordered by the ranker (§5.6), followed by the foreign keys between kept tables (`| fk : singer_in_concert.singer_id = singer.singer_id , …`).
4. Values (if `db_content: true`) follow the column in parentheses, case preserved (see §4.5).
5. No task prefix (e.g. no `translate English to SQL:`) — each model is fine-tuned for one task.
6. **Length budget 512 tokens.** If a full-schema input is longer: (a) re-order tables by relevance score (count of question tokens that overlap the table name or its column names), (b) drop values first, then (c) drop trailing columns from the least relevant tables until it fits. Every truncation is recorded (`truncated: true`); `prepare` reports the truncated share per split. Ranked inputs almost never need truncation.

### 4.5 Database-value matching (BRIDGE-style)

Purpose: give the model the exact spelling of literal values (e.g. `France`, `Accepted`) so generated `WHERE` clauses execute correctly.

1. For each database, build and cache (`data/cache/values/<db_id>.json`) the distinct values of every TEXT column, skipping values longer than 64 characters and columns with more than 5,000 distinct values.
2. Generate question n-grams (1–6 words, lower-cased, punctuation stripped).
3. A value matches if, case-insensitively, it equals an n-gram, or its `rapidfuzz` ratio with an n-gram is ≥ 85 and the value has ≥ 3 characters.
4. Keep at most `max_values_per_column` (default 2) matches per column — longest/highest-scoring first. Numbers are never matched (they appear verbatim in questions).
5. Used identically by the ranker and the generators, at training, evaluation and serving time.

### 4.6 Target normalization

The training target is the gold SQL, normalized so the model learns one consistent surface form:

1. Keywords, functions, table names, aliases and column names → lower case.
2. **String literals keep their case**; double-quoted literals become single-quoted (`"France"` → `'France'`, embedded `'` escaped as `''`).
3. One space between all tokens, including around `(`, `)` and `,`; none around the `.` of qualified names; no trailing semicolon — e.g. `select count ( * ) from singer as t1 where t1.name = 'Joe'`.
4. Normalization MUST be execution-preserving: a unit test runs gold and normalized gold on their database and requires identical results for 100% of train and dev.

T2 and T3 targets add a skeleton prefix (§5.7).

### 4.7 Tokenizer audit

The original T5 SentencePiece vocabulary cannot encode `<` (it becomes `<unk>`), which silently breaks every "less than" query. `spinneret audit-tokenizer`:

1. Encodes and decodes every normalized source and target string in train + dev.
2. Reports every character that maps to `<unk>` and every string whose round trip differs (whitespace-normalized comparison).
3. Writes the list of tokens to add to `data/processed/added_tokens.json`. For T5 these are **`" <="`, `" <>"`, `" <"` and `"<"`**: a bare added `<` swallows the space before it when decoding (`age< 30`), so — like PICARD's T5 checkpoints — the space-prefixed forms are added, longest first, with bare `<` as a fallback for values. Result on Spider: `<` is the only character missing from targets (391 queries); round-trip failures go from 391 to 0. (Inputs also contain a few `` ` `` and `\` characters, which are only reported.)

`train` adds these tokens and resizes the model's embedding matrix. A unit test asserts that `select name from singer where age < 30 and year <= 2000` and a `<>` query round-trip exactly. T5-small and T5-base share one tokenizer, so one audit covers both. CodeT5 uses a byte-level BPE tokenizer and needs no added tokens (the audit should report zero).

### 4.8 Schema-item labels

For every train/valid/dev example, the gold schema items come from the parsed `sql` field already present in the Spider JSON (no re-parsing of gold SQL needed):

- **tables:** every table in any FROM clause, including sub-queries and set-operation branches;
- **columns:** every column referenced anywhere — SELECT, WHERE, JOIN conditions, GROUP BY, HAVING, ORDER BY, sub-queries. `*` is ignored.

Stored per record as `gold_tables` and `gold_columns`. Used to train the ranker (§5.5), to build T3 training inputs (§5.6), for ranker coverage (§8.1) and for the taxonomy (§8.4).

### 4.9 Processed record format

`data/processed/{train,valid,dev}.jsonl`, one JSON object per line:

```json
{
  "id": "dev-00042",
  "split": "dev",
  "source_file": "dev.json",
  "db_id": "concert_singer",
  "question": "Which singers are from France?",
  "source": "Which singers are from France? | concert_singer | stadium : … ",
  "target": "select name from singer where country = 'France'",
  "gold_sql": "SELECT Name FROM singer WHERE Country = \"France\"",
  "skeleton": "select _ from _ where _ = _",
  "hardness": "easy",
  "gold_tables": ["singer"],
  "gold_columns": {"singer": ["name", "country"]},
  "n_tables": 1,
  "source_tokens": 118,
  "target_tokens": 16,
  "skeleton_target_tokens": 29,
  "truncated": false,
  "matched_values": {"singer.country": ["France"]}
}
```

`source` here is the full-schema input. Ranked inputs (T2, T3) depend on a trained ranker, so they are built later by `build-ranked` into `data/processed/ranked/<ranker_run>/{train,valid,dev}.jsonl` with the same fields plus `ranked_tables`, `ranked_columns` (with scores) and `skeleton_target`.

`data/processed/stats.json` holds per-split counts (by source file), difficulty distribution, source/target token-length percentiles and truncation rate — these feed the report's *Datasets* section.

---

## 5. Models and training

### 5.1 Models

| Tier / role | Hugging Face id | Params | Tokenizer | Served size (laptop GPU) |
|---|---|---|---|---|
| **T1 Baseline** generator | `google-t5/t5-small`, full schema | ~60M | SentencePiece (+ `<`, `<=`) | ~0.24 GB (fp32) |
| **Ranker** (shared by T2 and T3) | `FacebookAI/roberta-base` | ~125M | byte-level BPE | ~0.25 GB (fp16) |
| **T2 Lite** generator | `google-t5/t5-small`, ranked schema + skeleton | ~60M | same as T1 | ~0.24 GB (fp32) |
| **T3 Spinneret** generator | `google-t5/t5-base`, ranked schema + skeleton | ~220M | same as T1 | ~0.9 GB (fp32) |
| Ranker upgrade (Gate A) | `FacebookAI/roberta-large` | ~355M | byte-level BPE | ~0.7 GB (fp16) |
| T1 / T2 fallback (Gate B) | `Salesforce/codet5-small` | ~60M | byte-level BPE (32k) | ~0.24 GB |
| T3 fallback (Gate B) | `Salesforce/codet5-base` | ~220M | byte-level BPE (32k) | ~0.9 GB |
| Size × technique cell (P2) | `google-t5/t5-base`, full schema | ~220M | same as T1 | — (evaluation only) |
| Ensemble (MAY) | any of the above | — | — | see §6.3 |

All generators load through `AutoTokenizer` / `AutoModelForSeq2SeqLM`, so switching is a config change (FR11). Serving all three tiers together needs about 1.6 GB of GPU memory because T2 and T3 share one ranker.

**Gate A — ranker size (end of week 2):** if the RoBERTa-base ranker's coverage@(4,5) on the valid split is below 95%, first try raising *k* at inference (E5); if coverage is still below 95%, train `ranker_large.yaml` and use it for T2/T3.

**Gate B — generator family (end of week 3):** for any tier, train its same-size CodeT5 counterpart with the identical pipeline if its dev EX falls below that tier's minimum in PRD §8 (T1 42, T2 52, T3 66), or if tokenizer round-trip failures persist after adding tokens.

### 5.2 Generator training procedure

- Hugging Face `Seq2SeqTrainer`, teacher forcing, cross-entropy loss.
- Inputs truncated to 512 tokens, targets to 256 (skeleton + SQL included; `prepare` reports the share over 256 and the limit is raised to 384 if it exceeds 1%).
- Each evaluation pass: greedy-decode the `valid` split, compute EM (official script) and EX (§7), log to TensorBoard, keep the best checkpoint by `valid_ex` (tie-break `valid_em`), stop after `early_stopping_patience` evaluations without improvement.
- **LR sweep** with Adafactor, short runs (≈ 10 epochs) on the valid split:
  - T5-small (T1, T2): {1e-4, 3e-4, 1e-3} (1e-3 constant is the T5 paper's fine-tuning default); T2 reuses T1's best LR unless time allows its own sweep;
  - T5-base (T3): {5e-5, 1e-4, 3e-4} (fewer runs if Colab time is short — 1e-4 is the default).
- Seeds fixed for Python, NumPy, PyTorch.

### 5.3 Memory plan

| Model | Machine | Precision | Micro-batch × accumulation | Fallbacks on out-of-memory |
|---|---|---|---|---|
| T5-small (T1, T2) | Colab T4 15 GB (or laptop 4 GB for T1) | fp32 | 16 × 2 on T4; 8 × 4 on laptop | micro-batch halved → gradient checkpointing |
| T5-base (T3) | Colab T4 15 GB | fp32 | 8 × 4 | micro-batch 4 → gradient checkpointing |
| Ranker RoBERTa-base | Colab T4 15 GB | fp16 mixed (RoBERTa is stable in fp16; T4 has tensor cores) | 16 × 1 | micro-batch 8 |
| Ranker RoBERTa-large (Gate A only) | Colab T4 15 GB | fp16 mixed | 8 × 2 | micro-batch 4 |

T5 trains in **fp32** everywhere: T5 is known to overflow in fp16. Adafactor keeps optimizer memory well below AdamW's.

**Serving on the laptop** loads T1 (≈0.24 GB fp32), T2's T5-small (≈0.24 GB), T3's T5-base (≈0.9 GB) and the shared ranker (≈0.25 GB fp16) — about 1.6 GB plus activations, inside 4 GB.

### 5.4 Checkpoint layout

```
checkpoints/<run_id>/            run_id = <model>-<YYYYMMDD-HHMM>-<run_name>
├── model/                       save_pretrained() weights (safetensors) + tokenizer (incl. added tokens)
├── config.yaml                  exact config used
├── trainer_state.json
└── best_metrics.json            valid EM / EX (or ranker coverage) at the selected step
```

### 5.5 Schema-item ranker (P0, shared by T2 and T3)

A cross-encoder that reads the question together with the whole schema and outputs, for every table and every column, the probability that the gold SQL uses it (after RESDSQL's ranking-enhanced encoder; this is Spinneret's simplified version).

| Aspect | Specification |
|---|---|
| Input | Text pair: question + schema serialized as `table : col , col ( value ) | table : …` (names lower-cased, values from §4.5), max 512 tokens. If longer, tables are split into chunks, each paired with the question and scored separately. |
| Item representation | Character offsets of each table name and column name are mapped to tokens (`return_offsets_mapping`); the item vector is the mean of its tokens' final hidden states. Each column vector is concatenated with its table's vector. |
| Heads | Two linear heads → sigmoid: table head on table vectors, column head on column vectors. |
| Labels | `gold_tables`, `gold_columns` (§4.8). |
| Loss | Focal loss (γ = 2) on tables + columns, summed — most items are negatives. |
| Optimizer | AdamW; LR 1e-5 (large) / 3e-5 (base); linear decay, 10% warmup; up to 16 epochs. |
| Selection | Best checkpoint by validation **coverage@(4,5)** (§8.1), tie-break column AUC. |
| Backbone | `FacebookAI/roberta-base` by default; `roberta-large` only after Gate A (§5.1). |
| Output | `SchemaRanker.rank(question, db_id) -> RankedSchema` with every item's score; T2 and T3 keep the top-k. |

### 5.6 Ranked schema input (T2, T3)

1. Keep the top `top_k_tables` = 4 tables by score, and in each the top `top_k_columns` = 5 columns by score (the k values used by RESDSQL); fewer if the database has fewer.
2. Also keep primary/foreign-key columns that link two kept tables, so JOIN conditions remain writable, and list those links after the tables (`| fk : …`).
3. Order tables, and columns inside each table, by descending score — the order itself tells the generator what matters most.
4. **Training inputs** use the ranker's scores on the training set, but every gold table/column the ranker missed is inserted at its scored position, so the generator never learns to write items it cannot see. **Valid/dev/serving inputs** use the ranker output only — this is why ranker coverage caps T2's and T3's accuracy.

### 5.7 Skeleton-first targets (T2, T3)

The generator first writes the SQL **skeleton** (structure only), then the full SQL, so it commits to the query shape before filling in names and values (after RESDSQL's skeleton-aware decoder).

- **Skeleton rule:** take the normalized SQL; replace every table name, column name, alias, `*` and literal with `_`; then collapse `_ . _` → `_`, `_ as _` → `_`, and runs of `_ , _` → `_`. Keywords, operators, aggregates and parentheses stay.
- **Target:** `<skeleton> | <sql>`.

| SQL | Target |
|---|---|
| `select count ( * ) from singer` | `select count ( _ ) from _ \| select count ( * ) from singer` |
| `select name from singer where country = 'France'` | `select _ from _ where _ = _ \| select name from singer where country = 'France'` |
| `select t2.name from concert as t1 join stadium as t2 on t1.stadium_id = t2.stadium_id group by t1.stadium_id order by count ( * ) desc limit 1` | `select _ from _ join _ on _ = _ group by _ order by count ( _ ) desc limit _ \| select t2.name from …` |

- **Parsing outputs:** split at the first ` | ` and take the right side; if there is no separator, use the whole output as SQL.

### 5.8 Hugging Face Hub and Colab

| Aspect | Specification |
|---|---|
| Authentication | The owner is logged in on the laptop (`hf auth login`, account `iteshxt`; token stored by `huggingface_hub` in its local cache). For Colab, `scripts/colab/start.sh` copies that stored token **file** to the VM with `colab upload`, so the token never appears in a command line, environment flag, log or chat. Nothing token-related is written to the repo or configs. |
| Repos | One model repo per checkpoint: `iteshxt/spinneret-t5-small` (T1), `iteshxt/spinneret-ranker-roberta-base`, `iteshxt/spinneret-lite-t5-small-ranked` (T2), `iteshxt/spinneret-t5-base-ranked` (T3), plus the large ranker / CodeT5 variants if a gate triggers. |
| Visibility | Private while training; switched to public at release only after the owner confirms. |
| During training | `push_to_hub=True`, `hub_strategy="checkpoint"`: the latest checkpoint is kept at `last-checkpoint/` in the repo. `--resume hub` downloads it and continues with `resume_from_checkpoint` — this is how Colab's session limits are survived. |
| Final upload | `spinneret push` uploads the best checkpoint (safetensors), tokenizer, `config.yaml`, `best_metrics.json` and a generated `README.md` model card. |
| Model card | Base model and size, tier, served memory and latency, training data (Spider incl. `train_others`, CC BY-SA 4.0 with attribution), input/target format with an example, decoding, dev EM/EX overall and per difficulty, intended use, limitations, link to the GitHub repo. Card licence defaults to `cc-by-sa-4.0` (share-alike, matching the training data); revisit at release. |
| Loading | Any `checkpoint`/`ranker_checkpoint` field accepts a local path or `hub:<user>/<repo>` (downloaded to the Hugging Face cache). |
| Colab | Training runs through the Colab CLI workflow in §5.9. |

### 5.9 Colab CLI training workflow

Verified on 2026-11-21: the free tier allocates a T4 through the CLI with no compute units. The workflow is five small scripts in `scripts/colab/`, run from the laptop, so a change in the young CLI (v0.8) is a one-file fix.

| Step | Script | What it does |
|---|---|---|
| 1 | `start.sh <session>` | `colab new --gpu T4 -s <session>`; packs the current commit with `git archive` (no data, no checkpoints) and uploads it; uploads the stored Hugging Face token file to the VM's Hugging Face cache; runs `bootstrap.sh` on the VM. |
| 2 | `bootstrap.sh` (on the VM) | installs `uv` and Python 3.12, runs `uv sync`, then `spinneret download` and `spinneret prepare` (the VM's network is fast; ~200 MB of Spider data). Prints `nvidia-smi` as a GPU check. |
| 3 | `train.sh <session> <spinneret args…>` | starts the command under `nohup` in the background on the VM, writing to `/content/logs/<run>.log`, and returns immediately — training continues if the laptop sleeps or disconnects. |
| 4 | `logs.sh <session> <run>` | prints the tail of the remote log (progress, validation EM/EX, last Hub push). |
| 5 | `stop.sh <session>` | `colab stop -s <session>`, then checks `colab sessions` and `colab usage` until nothing is assigned. |

Rules:

- Checkpoints live on the Hub (§5.8), never only on the VM: `/content` is lost when the session ends or is reclaimed. A reclaimed run restarts with `start.sh` + `train.sh … --resume hub`.
- Ranked inputs for T2/T3 are rebuilt on the VM with `build-ranked` from the ranker on the Hub, so no processed data needs uploading.
- Evaluation of finished checkpoints runs on the laptop (`predict` + `evaluate`) so results land directly in `reports/`.
- `colab exec` exits 0 even when the remote command fails, so `start.sh` and `train.sh` check for explicit success markers (`SPINNERET_BOOTSTRAP_OK`, `SPINNERET_STARTED <run>`) in the output.
- Sessions are always stopped when a run finishes, to save the free GPU quota. `colab` commands that create or stop sessions run in the owner's terminal tab, so the owner's CLI state stays in sync.
- Fallbacks: the Colab web UI with the same scripts; the laptop for T1.

---

## 6. Inference and decoding

### 6.1 Generation

`Translator.generate(question, db_id, k) -> list[Candidate]`:

- `HFTranslator` (T1): builds the full-schema source with the same serializer and value matcher as training;
- `RankedTranslator` (T2, T3): calls `SchemaRanker.rank()` first, builds the ranked source (§5.6), and strips the skeleton from each output (§5.7);
- decodes with `num_beams = k`, `num_return_sequences = k` (default k = 4 for T1 and T2, 8 for T3), `max_new_tokens = 256`;
- returns candidates in beam order with their sequence log-probabilities.

```python
@dataclass
class Candidate:
    sql: str
    skeleton: str | None    # T2, T3 only
    score: float            # sequence log-probability
    rank: int               # 0 = top beam
    model: str              # tier / checkpoint id
```

### 6.2 Execution-guided selection

```
for cand in candidates (beam order):
    if not guard.allows(cand.sql):          → status "rejected", continue
    run on read-only DB with timeout        → status "ok" | "error" | "timeout"
    if status == "ok": return cand (selected)
return candidates[0] with status "failed"   # nothing executed
```

An empty result set counts as `ok` — many gold Spider queries legitimately return no rows. Decoding modes for ablations: `greedy` (k = 1), `beam` (top beam, no execution), `eg` (above).

### 6.3 Ensemble (MAY)

`EnsembleTranslator` implements the same `Translator` protocol: it pools candidates from several translators, executes each, groups candidates whose result sets are identical, and returns the group with the most members (ties broken by summed score). Only built if P0 and P1 are complete.

---

## 7. Database execution layer

| Concern | Rule |
|---|---|
| Connection | `sqlite3.connect("file:<path>?mode=ro", uri=True)`, then `PRAGMA query_only = ON` — the read-only connection is the real guarantee; the guard is defence in depth. |
| Statement guard | Parse with `sqlparse`; exactly one statement (trailing `;` allowed); first keyword `SELECT` or `WITH`; reject if any **keyword token** (not substring) is one of `INSERT UPDATE DELETE DROP ALTER CREATE REPLACE ATTACH DETACH PRAGMA VACUUM REINDEX TRUNCATE`. |
| Timeout | `set_progress_handler` aborts after **5 s** wall-clock → status `timeout`. |
| Row cap | `fetchmany(501)`; return at most **500** rows and `truncated: true` if more exist. |
| Encoding | Some Spider databases contain non-UTF-8 bytes: `text_factory` decodes with `errors="replace"`. |
| JSON conversion | `None` → `null`; `bytes` → `"<blob N bytes>"`; numbers unchanged. |
| Comparison | EX is computed by the official script; Spinneret's own `results_equal()` (used by the ensemble) compares results as multisets of rows, order-insensitive unless the gold has `ORDER BY`. |

---

## 8. Evaluation

### 8.1 Metrics

| Metric | Definition | Tool |
|---|---|---|
| **EM** — exact-set-match | Prediction and gold are parsed into clauses; each clause is compared as a set, so order inside a clause doesn't matter; **literal values are ignored**. Correct only if all clauses match. | official `evaluation.py --etype match` |
| **EX** — execution accuracy | Prediction and gold are executed on the Spider database; correct if results match. Uses the predicted values. | official `evaluation.py --etype exec` with `--db` = Spider databases |
| Partial (per-clause) F1 | select, select (no AGG), where, where (no OP), group (no Having), group, order, and/or, IUEN, keywords | same script |
| EX with gold values plugged in (SHOULD) | `--plug_value`: substitutes gold literals into the prediction; **EX(plug) − EX ≈ share of errors caused only by wrong values** | same script |
| Test-suite accuracy (MAY) | EX over the distilled test-suite databases (~3.27 GB); removes most EX false positives; Spider's official metric since Nov 2020 | same script, `--db` = test-suite databases |
| **Ranker coverage@(k1,k2)** | share of examples where every gold table is in the top-k1 tables **and** every gold column is in the top-k2 columns of its table — an upper bound on T3 accuracy | `eval/coverage.py` |
| Ranker AUC | ROC-AUC for tables and for columns | `eval/coverage.py` |
| Valid-SQL rate | share of selected predictions that execute without error | Spinneret |
| Latency | p50 / p95 per question: ranking, inference and execution separately | Spinneret |

All accuracy metrics are reported **overall and per difficulty level** (easy / medium / hard / extra), for every tier.

The official scripts come from `taoyds/test-suite-sql-eval` (Apache-2.0), vendored as a git submodule in `third_party/`. Spinneret's wrapper MUST reproduce the script's printed numbers exactly (a test compares them).

### 8.2 Evaluation run layout

```
reports/<run_id>/                run_id = <tier>-<checkpoint>-<split>-<decoding>
├── predictions.jsonl            id, db_id, question, gold_sql, pred_sql, skeleton, candidates[], ranked items, exec_status, hardness, em, ex
├── pred.txt                     one SQL per line, dev order (input to official script)
├── metrics.json                 EM/EX overall + per difficulty + partial F1 + coverage + valid-SQL rate + latency
├── taxonomy.csv                 one row per wrong prediction with its categories
└── taxonomy_summary.json        counts per category, per difficulty, with 2 examples each
```

`metrics.json` and `taxonomy_summary.json` are small and are committed; the API serves them to the evaluation page.

### 8.3 Planned experiments

| ID | Experiment | Compares | Priority |
|---|---|---|---|
| E1 | **Tier comparison** (main result) | T1 vs T2 vs T3 on dev — accuracy, size, served memory and latency; values on; execution-guided decoding | P0 |
| E2 | Decoding | T3: greedy vs beam-top-1 vs execution-guided; beam 4 vs 8 (accuracy vs latency) | P1 |
| E3 | Value matching | T1: `db_content` off vs on | P1 |
| E4 | Ranker vs skeleton | T2 without skeleton (ranked schema, SQL target). T1 → T2-no-skeleton isolates the ranker; T2-no-skeleton → T2 isolates the skeleton. One extra T5-small run (cheap). | P1 |
| E5 | Ranker *k* (inference only) | T3 with k1 ∈ {3, 4, 5}, k2 ∈ {4, 5, 6} — no retraining | P1 |
| E6 | `train_others` | T1 trained with vs without it | P1 |
| E7 | Model size | T2 → T3 (same ranker and targets, T5-small vs T5-base) — part of E1 | P0 |
| E8 | Ranker size | RoBERTa-base vs RoBERTa-large coverage and T3 EX (only if Gate A triggers) | conditional |
| E9 | T5 vs CodeT5 | same size, same pipeline (only if Gate B triggers) | conditional |
| E10 | Full-schema T5-base | completes the size × technique grid (T5-base without ranker) | P2 |
| E11 | int8 inference | T3 quantized to int8 for CPU serving: accuracy and latency vs fp32 | P2 |
| E12 | Test set | best configuration, run once | P2 |

### 8.4 Failure taxonomy

Every prediction with EX = 0 (plus EM = 0 / EX = 1 cases, reported separately) is labelled by comparing the parsed prediction with the parsed gold (Spider's `process_sql.get_sql`). Labels are **multi-label**; the **primary** label is the first match in this order:

| # | Category | Rule (prediction vs gold) |
|---|---|---|
| 1 | `invalid_sql` | prediction fails to parse or fails to execute |
| 2 | `wrong_table` | different set of tables in FROM (same count) |
| 3 | `join_error` | different number of tables (missing / extra join) or different join conditions |
| 4 | `nesting_setop_error` | nested sub-query or INTERSECT / UNION / EXCEPT present in one but not the other, or differs |
| 5 | `wrong_column` | different columns in SELECT or WHERE (ignoring aggregates) |
| 6 | `aggregation_error` | different / missing / extra aggregate (COUNT, SUM, AVG, MIN, MAX) |
| 7 | `group_by_error` | GROUP BY or HAVING differs |
| 8 | `condition_error` | same WHERE columns but different operator, connector (AND/OR) or negation |
| 9 | `order_limit_error` | ORDER BY column/direction or LIMIT differs |
| 10 | `value_error` | EM correct but EX wrong → only a literal value is wrong |
| 11 | `other` | none of the above (e.g. DISTINCT) |
| — | `semantically_equivalent` | EM wrong but EX correct — not an error, reported to show EM's false negatives |

For T2 and T3, every `wrong_table` / `wrong_column` / `join_error` is additionally flagged **`ranker_miss`** when a needed gold item was filtered out by the ranker — separating ranking failures from generation failures.

**Manual review (MUST):** a seeded random sample of 100 EX-wrong predictions (from T3, or T1 if T3 is not ready) is hand-labelled with the same categories in `reports/<run_id>/manual_review.csv`; the report gives the agreement rate between manual and automatic primary labels.

### 8.5 Question-type analysis

Features from the gold SQL: number of tables (1 / 2 / 3+), has aggregate, has GROUP BY, has ORDER BY/LIMIT, has nesting/set-op. EX is reported per bucket and per tier, and per named type:

| Type | Definition |
|---|---|
| simple lookup | 1 table, no aggregate, no GROUP BY, no nesting |
| aggregation | 1 table with aggregate and/or GROUP BY |
| two-table join | exactly 2 tables, no nesting |
| multi-join | 3+ tables |
| nested / set-op | any sub-query or INTERSECT / UNION / EXCEPT |

This answers the brief's "which question types does the model handle best and worst" — and shows *where* the ranker (T1 → T2) and the larger generator (T2 → T3) gain.

---

## 9. Backend API (FastAPI)

Base path `/api`. JSON only. All configured models and all schemas load once at startup (`deps.py`); the server uses the GPU if available, else CPU.

`configs/serve.yaml`:

```yaml
device: auto                       # auto | cuda | cpu
database_dir: data/raw/spider/database
databases: dev                     # the 20 dev databases (unseen in training)
default_model: t3
rankers:
  base: hub:iteshxt/spinneret-ranker-roberta-base     # loaded once, shared by t2 and t3
models:
  - id: t1
    label: "Baseline · T5-small"
    generator: hub:iteshxt/spinneret-t5-small
    beam_size: 4
  - id: t2
    label: "Lite · T5-small + ranker"
    generator: hub:iteshxt/spinneret-lite-t5-small-ranked
    ranker: base
    beam_size: 4
  - id: t3
    label: "Spinneret · T5-base + ranker"
    generator: hub:iteshxt/spinneret-t5-base-ranked
    ranker: base
    beam_size: 8
execution: { timeout_s: 5, max_rows: 500 }
```

### 9.1 Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | `{status, device, models_loaded}` |
| GET | `/api/models` | list of `{id, label, generator, ranker, params, default, dev_ex, p50_ms}` (accuracy and latency from the latest evaluation, so the UI can show the trade-off) |
| GET | `/api/databases` | list of `{db_id, n_tables, n_columns}` |
| GET | `/api/databases/{db_id}/schema` | tables with columns (name, type, is_pk), foreign keys |
| GET | `/api/databases/{db_id}/examples?limit=6` | sample dev questions for that database |
| POST | `/api/query` | question → (ranking) → SQL → result |
| POST | `/api/execute` | run user-edited SQL (same guard, timeout, row cap) |
| GET | `/api/metrics` | list of evaluation runs |
| GET | `/api/metrics/{run_id}` | `metrics.json` + `taxonomy_summary.json` for that run |

### 9.2 `POST /api/query`

Request:

```json
{
  "db_id": "concert_singer",
  "question": "Which singers are from France?",
  "options": { "model": "t3", "beam_size": 8, "decoding": "eg", "max_rows": 500 }
}
```

Response `200`:

```json
{
  "model": "t3",
  "sql": "select name from singer where country = 'France'",
  "skeleton": "select _ from _ where _ = _",
  "selected_rank": 0,
  "candidates": [
    { "rank": 0, "sql": "select name from singer where country = 'France'", "score": -0.21, "status": "ok" },
    { "rank": 1, "sql": "select name from singer where country = 'france'", "score": -1.87, "status": "not_run" }
  ],
  "schema_items": {
    "tables":  [ { "name": "singer", "score": 0.99, "kept": true }, { "name": "concert", "score": 0.04, "kept": true } ],
    "columns": [ { "table": "singer", "name": "country", "score": 0.98, "kept": true }, { "table": "singer", "name": "name", "score": 0.95, "kept": true } ]
  },
  "matched_values": { "singer.country": ["France"] },
  "execution": {
    "status": "ok",
    "columns": ["name"],
    "rows": [["Joe Sharp"]],
    "row_count": 1,
    "truncated": false,
    "error": null
  },
  "timings_ms": { "ranking": 38, "inference": 690, "execution": 3, "total": 735 }
}
```

`skeleton` and `schema_items` are `null` for the model without a skeleton or ranker (T1). `execution.status` ∈ `ok | error | timeout | rejected | failed`. Generation succeeding but SQL failing is still `200` (the UI shows the error); `candidates[].status` adds `not_run` for beams after the selected one.

### 9.3 `POST /api/execute`

Request `{ "db_id": "...", "sql": "...", "max_rows": 500 }` → the `execution` object above.

### 9.4 Errors

| Code | When | Body |
|---|---|---|
| 400 | empty question (after trim) or question > 500 characters; SQL > 5,000 characters; unknown `model` | `{ "error": { "code": "invalid_input", "message": "…" } }` |
| 404 | unknown `db_id` or `run_id` | `{ "error": { "code": "not_found", … } }` |
| 422 | malformed JSON / wrong types (FastAPI validation) | FastAPI default |
| 503 | models not loaded yet | `{ "error": { "code": "model_loading", … } }` |

CORS: only the Vite dev origin (`http://localhost:5173`) in development; in production the built frontend is served by FastAPI from the same origin.

---

## 10. Frontend (React)

| Item | Spec |
|---|---|
| Build | Vite + React + TypeScript (strict). Dev server proxies `/api` → `http://localhost:8000`. `npm run build` → `web/dist/`, served by FastAPI at `/`. |
| Routes | `/` — Query workspace · `/evaluation` — metrics, ablations and taxonomy · `/about` — what Spinneret is, model cards, links |
| Data fetching | TanStack Query; typed API client in `web/src/api/` mirroring §9 models |
| Components | `ModelSwitcher`, `DatabasePicker`, `SchemaExplorer` (with ranker highlighting), `QuestionInput`, `ExampleChips`, `SqlPanel` (CodeMirror, editable, copy, run), `SkeletonLine`, `CandidatesList`, `ResultTable`, `StatusLine`, `HistoryList`, `MetricsOverview`, `TierComparisonChart`, `HardnessChart`, `AblationTable`, `TaxonomyTable` |
| State | Selected model, database and history in URL/search params and `sessionStorage` (history is per session; no backend persistence) |
| Accessibility | Full keyboard flow; visible focus; `aria-live="polite"` on the result region; WCAG AA contrast |

UI layout, states and visual language are specified in [design.md](design.md) §8–§12.

---

## 11. Testing

| Suite | Covers | Must include |
|---|---|---|
| `tests/test_serialize.py` | full and ranked serializer, truncation | exact expected string for `concert_singer`; 512-token budget; FK list for kept tables |
| `tests/test_normalize.py` | target normalization | execution-preserving on all train + dev gold SQL (marked slow) |
| `tests/test_skeleton.py` | skeleton rule + output parsing | the three examples in §5.7; output without separator |
| `tests/test_labels.py` | schema-item labels | tables/columns from sub-queries and joins; `*` ignored |
| `tests/test_tokenizer.py` | added tokens | `<` and `<=` round-trip with T5 |
| `tests/test_values.py` | value matching | `France` matched for "from france"; numbers never matched |
| `tests/test_ranker_data.py` | ranker inputs | item token spans align with names; chunking of long schemas; top-k + gold force-include at train time only |
| `tests/test_guard.py` | SQL guard | rejects `DROP`, `INSERT`, `ATTACH`, `PRAGMA`, two statements; allows `WITH … SELECT`, columns named `created_at` / `update_date` |
| `tests/test_executor.py` | read-only execution | write attempt fails even if the guard is bypassed; timeout fires; row cap and `truncated` flag |
| `tests/test_eval.py` | evaluation wrapper + coverage | gold-vs-gold = 100% EM and EX; wrapper numbers equal the official script's; coverage of gold ranking = 100% |
| `tests/test_taxonomy.py` | taxonomy rules | one hand-made example per category; `ranker_miss` flag |
| `tests/test_hub.py` | Hub helper | model card renders required fields; push uploads the expected files (Hub client mocked); no token in any written file |
| `tests/test_api.py` | FastAPI (TestClient, stub translators) | all endpoints, model switch, error codes |
| `web/` (Vitest) | components | ResultTable, SqlPanel, SchemaExplorer highlighting states |

`uv run pytest -m "not slow"` runs in under a minute; slow tests run before each evaluation.

---

## 12. Non-functional specifications

| Area | Specification |
|---|---|
| Performance | Laptop GPU, p50 / p95: T1 ≤ 0.8 s / ≤ 1.5 s; T2 ≤ 1.0 s / ≤ 2 s; T3 (ranking + beam 8 + execution) ≤ 2.0 s / ≤ 4 s (PRD §7); measured over the dev set by `predict` and recorded in `metrics.json`. Server ready ≤ 30 s once models are cached. |
| Reproducibility | Seeds fixed; config stored per checkpoint (locally and on the Hub); every number in the report comes from a committed `metrics.json`. |
| Security | Read-only DB connections; guard; timeout; row cap; input length limits; no shell or file access from requests; CORS restricted. |
| Secrets | Hugging Face token only in the local `huggingface_hub` login and, as a copied file, on the Colab VM; Colab login in the OS keyring; `.env*` git-ignored; a test checks written artifacts and the `git archive` bundle contain no token. |
| Data handling | Spider data, checkpoints, TensorBoard logs and caches are git-ignored. |
| Licences | Spider CC BY-SA 4.0 (attribution in README and model cards); test-suite-sql-eval Apache-2.0 (submodule keeps its licence); base models: T5 and CodeT5 Apache-2.0, RoBERTa MIT. |

---

## 13. Acceptance criteria by milestone

| Week | Done when |
|---|---|
| 1 | `spinneret download && spinneret prepare` produce train/valid/dev JSONL (train includes `train_others`) with schema-item labels + `stats.json`; tokenizer audit lists `<`, `<=` for T5; serializer, normalization, skeleton and label tests pass; `scripts/colab/` smoke run: session → code + token file uploaded → bootstrap → `nvidia-smi` → Hugging Face login verified on the VM → stop. **Done 2026-11-21** (T4 15 GB, torch 2.15.0 + CUDA, Hub user `iteshxt`; the first real Hub push happens with T1 in week 2). |
| 2 | Executor + guard tests pass; evaluation wrapper reproduces 100% on gold-vs-gold; `/api/databases` and `/schema` work; **T1 trained on Colab, evaluated on dev, pushed to the Hub (private); ranker trained, coverage@(4,5) reported; Gate A decided.** |
| 3 | Ranked inputs built; **T2 and T3 trained on Colab** with resumable Hub checkpoints; Gate B decided (§5.1). |
| 4 | `reports/` contains E1, E7 and the P1 ablations; per-difficulty, per-question-type, latency and taxonomy tables (with `ranker_miss`) and the manual review. |
| 5 | Demo runs end-to-end on all 20 dev databases with the three-tier switch and schema highlighting; evaluation page shows E1; README explains setup in ≤ 5 commands; model cards published (visibility confirmed). Final report (DOCX + PDF) follows after the project is complete. |
