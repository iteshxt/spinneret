# Spinneret — Product Requirements Document

| | |
|---|---|
| **Product** | Spinneret — natural-language to SQL |
| **Owner** | Solo project (one developer does data, model, evaluation, demo and report) |
| **Timeline** | 5 weeks |
| **Status** | Draft v3 — model plan balanced for size and speed: Baseline · Lite · Spinneret |
| **Companion docs** | [specs.md](specs.md) (technical specification) · [design.md](design.md) (architecture + UI) |

> A *spinneret* is the organ a spider uses to spin silk. Spinneret spins plain-English questions into SQL threads — trained on the **Spider** benchmark.

---

## 1. Summary

Spinneret is a system that translates a natural-language question (for example *"Which singers are older than 30?"*) into an executable SQL query for a given relational database, runs that query, and shows the result. It fine-tunes small T5 sequence-to-sequence transformers on the **Spider** text-to-SQL dataset. Training runs on the free Google Colab T4, driven from the terminal with Google's Colab CLI; serving runs on a student laptop (GTX 1650, 4 GB VRAM).

The design goal is **moderate-to-good accuracy from small, fast models**, not maximum accuracy at any size. Most of the accuracy gain comes from a small *schema ranker* that tells the generator which tables and columns matter, so the generator itself can stay small. The model is built in **three tiers**:

| Tier | Models | Total size | Role |
|---|---|---|---|
| **T1 — Baseline** | T5-small (60M), full schema | ~60M | The brief's required model; reference point |
| **T2 — Spinneret Lite** | T5-small + RoBERTa-base ranker (125M) + skeleton-first decoding | ~185M | Fast mode — smallest model with the ranker technique |
| **T3 — Spinneret** | T5-base (220M) + the same RoBERTa-base ranker + skeleton-first decoding | ~345M | Main model — best accuracy that still loads and answers quickly on a laptop |

All tiers use execution-guided selection (the first beam candidate that actually runs wins).

The project has two equal halves:

1. **A research result** — the three tiers evaluated with the standard Spider metrics (exact-set-match and execution accuracy), broken down by query difficulty and question type, ablations showing what each technique and the model size contribute, and a taxonomy of the errors the models make.
2. **A working product** — a web demo (FastAPI + React) where you pick a database, type a question, and get the generated SQL and the live result table, switching between the three tiers. Trained checkpoints are published on the Hugging Face Hub.

## 2. Problem statement

Relational databases hold most structured data, but querying them requires SQL — a skill most people who need the data do not have. Text-to-SQL systems close that gap by translating questions into queries.

The state of the art today is large language models (GPT-4-class models reach ~86% execution accuracy on Spider's test set), but they are expensive, closed, and need network access. The question this project answers is:

> **How accurate can small, open, fine-tuned models be on cross-database text-to-SQL when they must load fast and answer quickly on a student laptop — what is the best accuracy-per-parameter trade-off — and where do they still fail?**

The task is *cross-domain*: Spider's evaluation databases are never seen in training, so the model must read an unfamiliar schema and map question words to the right tables and columns.

### What the literature says about size vs technique

Published Spider dev results (fine-tuned T5 family, no LLMs):

| Model | Size | EM | EX | Source |
|---|---|---|---|---|
| T5-small | 60M | 45.8 | 46.2 | Nguyen, Phan & Piccardi 2025 |
| T5-base | 220M | 57.4 | 60.2 | same |
| CodeT5-base | 220M | 62.7 | 63.6 | Chen et al. 2023 |
| T5-large | 770M | 65.3 | 67.2 | Scholak et al. 2021 (PICARD) |
| T5-3B | 3B | 69.9 | 71.4 | same |
| **RESDSQL-Base** (T5-base + RoBERTa-large ranker + skeleton) | 220M + 355M | **71.7** | **77.9** | Li et al. 2023 |

Even T5-3B, 50× larger than T5-small, stays near 71% EX. A base-size generator *with a schema ranker* reaches ~78% EX. Size has steep diminishing returns; technique is cheap.

### Options considered

| Option | Models | Size | Expected EX | Speed / memory | Decision |
|---|---|---|---|---|---|
| Baseline | T5-small | 60M | ~46% (published) | fastest | **T1** (required baseline) |
| Lite | T5-small + RoBERTa-base ranker | ~185M | ~55–62% *(estimate)* | very fast, ~0.5 GB | **T2** (fast mode) |
| Balanced | T5-base + RoBERTa-base ranker | ~345M | ~68–75% *(estimate)* | fast, ~1.2 GB | **T3** (main model) |
| Max | T5-base + RoBERTa-large ranker | ~575M | ~72–78% (RESDSQL-Base: 77.9) | slower, ~1.6 GB | only if the base ranker's coverage is too low (§9 gate) |
| Large generators | T5-large / T5-3B | 770M–3B | 67–71% (published) | too big for the laptop | rejected |

Estimates are ours (no published result uses exactly these combinations) and will be replaced by measured numbers.

## 3. Goals and non-goals

### Goals

| ID | Goal |
|---|---|
| G1 | Fine-tune T5 models on Spider (train + `train_others`, 8,659 questions) that generate executable SQL **with literal values**. |
| G2 | Build the three tiers (T1, T2, T3) on one shared pipeline, so differences between them come only from technique and model size. |
| G3 | Get the best accuracy for the size: **T3 ≥ 65% EM and ≥ 72% EX** on the Spider dev set (the brief's 78% EX is a stretch goal), **T2 ≥ 58% EX**. |
| G4 | Keep models small and fast: T3 loads in seconds and answers in ≤ 2 s on the laptop GPU; T2 in ≤ 1 s. |
| G5 | Report EM and EX separately, overall and per Spider difficulty level, plus ablations isolating the ranker, the skeleton, decoding and model size. |
| G6 | Produce a **failure taxonomy** (wrong table, wrong column, missing join, aggregation error, …) with counts and examples, per tier. |
| G7 | Show which question types the models handle best and worst (simple lookups vs multi-table joins vs nested queries). |
| G8 | Ship a **demo**: question → generated SQL → live result table, against real Spider databases, running locally, with a switch between the three tiers. |
| G9 | Keep models swappable: T5 first, with the same-size **CodeT5** model as a config-only fallback if T5 lags; leave room for an **ensemble** later. |
| G10 | Train on Colab **from the terminal** (Colab CLI), resumable through checkpoints on the **Hugging Face Hub**, which also publishes the models with model cards. |
| G11 | Everything reproducible from the repo: one command per stage (prepare, train, evaluate, serve, push). |

### Non-goals

- Maximum accuracy at any size, competing with LLM-based systems, or the Spider leaderboard top (85–91% EX).
- T5-large or T5-3B generators — too large to load quickly on the laptop, and the evidence above shows technique matters more.
- NatSQL intermediate representation (RESDSQL's extra +2–3 points) — too much extra machinery for 5 weeks.
- PICARD constrained decoding (needs a separate Haskell parser service) — replaced by execution-guided selection.
- Multi-turn conversation (SParC/CoSQL), non-English questions, or databases other than SQLite.
- Writing to databases. Spinneret is **read-only** by design.
- User accounts, paid cloud hosting, or production hardening. The public demo is a free static site whose live server is the author's laptop, reached through a tunnel and only on while the laptop serves it (design §13).

## 4. Users and use cases

| User | What they want | How Spinneret serves them |
|---|---|---|
| **Evaluator / instructor** | Verify the project meets the brief: metrics, taxonomy, demo, report. | Evaluation page with all metrics; reproducible scripts; final report. |
| **Portfolio viewer / recruiter** | See that the builder can take a research problem end-to-end. | Clean GitHub repo, Hugging Face model pages, live demo, clear docs. |
| **Demo user** | Ask questions of a database without writing SQL. | Pick a DB, browse its schema, type a question, see SQL + results, edit and re-run SQL. |
| **Me (researcher)** | Understand model behaviour and iterate. | Per-example predictions, candidates, ranked schema items, error categories, ablations. |

### Core user stories

- **US1** — As a demo user, I pick a database from a list and see its tables and columns, so I know what I can ask about.
- **US2** — As a demo user, I type a question and get the generated SQL and its result table within a couple of seconds.
- **US3** — As a demo user, I can click example questions for the selected database to try the system quickly.
- **US4** — As a demo user, when the SQL is wrong I can edit it and re-run it.
- **US5** — As a demo user, I can see which tables and columns the model focused on (highlighted in the schema).
- **US6** — As a demo user, I can switch between Baseline, Lite and Spinneret and see the difference in accuracy and speed on the same question.
- **US7** — As an evaluator, I can open an evaluation page showing EM/EX per tier and difficulty, ablations, and the failure taxonomy.
- **US8** — As the researcher, I can retrain or re-evaluate any model from a config file with one command, launching Colab training from my terminal.
- **US9** — As the researcher, I can swap T5 for CodeT5 without changing pipeline code.

## 5. Scope and priorities

**P0 = must ship · P1 = should ship · P2 = stretch**

| Area | Feature | Priority |
|---|---|---|
| Data | Spider loader (`train_spider` + `train_others` + dev), schema serialization, validation split by database | P0 |
| Data | Tokenizer audit + fix for characters T5 cannot encode (`<`, `<=`) | P0 |
| Data | Database-value matching: append question-matched cell values to columns (BRIDGE-style) | P0 |
| Data | Schema-item labels (which tables/columns each gold query uses) for ranker training | P0 |
| Model | **T1** — T5-small, full schema | P0 |
| Model | **Schema-item ranker** — RoBERTa-base cross-encoder that scores every table and column | P0 |
| Model | **T2 Lite** — T5-small on the ranked schema with skeleton-first targets | P0 |
| Model | **T3 Spinneret** — T5-base on the ranked schema with skeleton-first targets | P0 |
| Model | RoBERTa-large ranker — only if the base ranker misses the coverage gate (§9) | P1 (conditional) |
| Model | Config-driven model choice (T5 ↔ CodeT5 of the same size) | P0 |
| Model | CodeT5 counterpart trained and compared | P1 (becomes P0 if T5 misses a gate in §9) |
| Model | T5-base on the full schema (completes the size × technique comparison) | P2 |
| Model | Ensemble of models with execution-based voting | P2 |
| Decoding | Beam search + execution-guided selection (first candidate that runs) | P0 |
| Serving | int8-quantized inference for CPU-only machines (speed vs accuracy measured) | P2 |
| Execution | Read-only SQLite executor with statement guard, timeout and row cap | P0 |
| Evaluation | EM + EX on Spider dev with the official scripts, per tier and difficulty level | P0 |
| Evaluation | Failure taxonomy: automatic labelling + manual review of a sample | P0 |
| Evaluation | Ablations: decoding, value matching, ranker, skeleton, model size, `train_others` | P1 |
| Evaluation | Test-suite accuracy (needs a 3.27 GB download) | P2 |
| Evaluation | One final run on the released Spider test set | P2 |
| Training | Colab CLI workflow: start a T4 session, ship code, run training in the background, follow logs, stop | P0 |
| Hub | Checkpoints + model cards on the Hugging Face Hub; training resumes from Hub checkpoints | P0 |
| Demo | FastAPI backend: databases, schema, models, query, execute, examples, metrics | P0 |
| Demo | React UI: DB picker, schema explorer, question box, SQL + result table | P0 |
| Demo | Editable SQL with re-run; beam candidates; ranked-schema highlighting; tier switch; session history | P1 |
| Demo | Evaluation page (metrics, ablations, taxonomy charts) | P1 |
| Docs | prd.md, specs.md, design.md, README | P0 |
| Report | Final report as **DOCX + PDF** — written after the project is complete | P0 (week 5+) |

## 6. Functional requirements

| ID | Requirement |
|---|---|
| FR1 | The system loads Spider's `tables.json` and SQLite databases and exposes, for every database, its tables, columns, column types, primary keys and foreign keys. |
| FR2 | Given `(question, db_id)`, the ranker scores every table and column; T2 and T3 keep the top-ranked items (default 4 tables, 5 columns per table) in ranked order. |
| FR3 | The system serializes the question and (ranked or full) schema, with matched values, into a single model input of at most 512 tokens. |
| FR4 | The model returns *k* SQL candidates (beam search; default *k* = 4 for T1 and T2, 8 for T3). |
| FR5 | The system executes candidates in order on a **read-only** connection and returns the first that executes without error; if none do, it returns the top candidate marked as failed. |
| FR6 | Execution is limited to a single `SELECT`/`WITH` statement, a 5-second timeout and a 500-row cap; anything else is rejected before it reaches SQLite. |
| FR7 | The API returns the SQL, all candidates with their status, the ranked schema items, the result (columns, rows, row count, truncation flag) and timings. |
| FR8 | The user can submit edited SQL for execution under the same safety rules. |
| FR9 | The evaluation pipeline produces EM and EX overall and per difficulty, partial (per-clause) scores, ranker coverage, latency, and a per-example predictions file. |
| FR10 | The failure analyser labels every incorrect prediction with one or more error categories and outputs a counts table with examples. |
| FR11 | Switching models (tier, T5 ↔ CodeT5, ranker size, local path ↔ Hub id) requires only a config change. |
| FR12 | Training can be launched on a Colab T4 from the laptop terminal, keeps running if the laptop disconnects, and resumes from the last checkpoint pushed to the Hugging Face Hub. |

## 7. Non-functional requirements

| Area | Requirement |
|---|---|
| Hardware | Training on the Colab T4 free tier (15 GB), launched with the Colab CLI; the laptop GTX 1650 (4 GB VRAM, 16 GB RAM) is the fallback for T1 and the machine for serving and evaluation. |
| Model size | All three tiers together fit in ≤ 2 GB of GPU memory when served (they share one ranker). |
| Latency | Laptop GPU, p50: T1 ≤ 0.8 s; T2 ≤ 1.0 s; T3 ≤ 2.0 s (ranking + generation + execution). Must also work on CPU (slower). |
| Load time | Server ready in ≤ 30 s from start once models are cached locally. |
| Reproducibility | Fixed seeds; each checkpoint stores its config; every reported number is regenerable from a saved predictions file. |
| Safety | Databases opened read-only; SQL guard; timeouts; no user SQL can modify data or attach files. |
| Secrets | The Hugging Face token never appears in the repo, configs, commands, logs or chat: it lives in the local `hf auth login` store and is copied to the Colab VM as a file. The Colab login lives in the OS keyring. |
| Licensing | Spider data is CC BY-SA 4.0 and is **not committed** to the repo (downloaded by script); model cards credit Spider. Spider evaluation scripts are Apache-2.0. |
| Code quality | Typed Python (ruff-clean), unit tests for serialization, ranker inputs, SQL guard, executor and API; TypeScript frontend. |

## 8. Success metrics

All on the Spider dev set (1,034 questions, 20 unseen databases), with the official scripts.

| Tier | Reference | Minimum (pass) | Target | Stretch |
|---|---|---|---|---|
| T1 — Baseline (T5-small) | published 45.8 EM / 46.2 EX | 42 / 42 | 46 / 48 | 50 / 52 |
| T2 — Lite (T5-small + ranker) | none published; estimate 55–62 EX | 50 / 52 | 55 / 58 | 60 / 63 |
| **T3 — Spinneret (T5-base + ranker)** | RESDSQL-Base 71.7 / 77.9 (with a larger ranker) | 60 / 66 | **65 / 72** | 70 / 78 (the brief's EX) |

| Other metric | Target |
|---|---|
| Ranker coverage@(4,5) (every gold table and column survives the filter) | ≥ 95% on validation (gate for upgrading the ranker) |
| Executable-SQL rate with execution-guided selection | ≥ 95% (T3) |
| Failure taxonomy | every wrong prediction labelled; 100 hand-checked; agreement rate reported |
| Demo latency (laptop GPU, p50) | T1 ≤ 0.8 s · T2 ≤ 1.0 s · T3 ≤ 2.0 s |
| Demo coverage | all 20 dev databases load and answer |

## 9. Milestones (5 weeks)

| Week | Focus | Exit criteria |
|---|---|---|
| **1** | Setup + dataset + Colab workflow | Repo, `uv` environment, Spider downloaded by script; loader for train + `train_others` + dev; serializer, value matching, validation split by DB; tokenizer audit; schema-item labels; dataset statistics. Colab CLI workflow smoke-tested end to end (session → code → GPU check → Hub push → stop). |
| **2** | Execution layer + T1 + ranker | Read-only executor + SQL guard with tests; evaluation harness verified (gold vs gold = 100%); backend skeleton; **T1 trained on Colab and evaluated on dev; ranker trained and its coverage measured.** **Gate A:** if ranker coverage@(4,5) < 95%, train the RoBERTa-large ranker. |
| **3** | T2 Lite + T3 Spinneret | Ranked inputs built; **T2 and T3 trained on Colab** with checkpoints pushed to the Hub after every epoch. Frontend skeleton built while training runs. **Gate B:** if a tier's dev EX falls below its minimum in §8, train its CodeT5 counterpart (see [specs §5.1](specs.md#51-models)). |
| **4** | Evaluation | All tiers on dev; ablations (decoding, values, ranker, skeleton, size, `train_others`); per-difficulty and question-type tables; latency per tier; failure taxonomy + manual review. |
| **5** | Demo + release | End-to-end demo (tier switch, schema highlighting, edit/re-run, evaluation page); README; model cards published on the Hub. The final report (DOCX + PDF) is written once the project is complete. |

## 10. Deliverables

1. **Code** — Python package (data, ranker, model, decoding, evaluation, API, Colab scripts) + React app, in one Git repo on GitHub.
2. **Trained checkpoints** — T1, ranker, T2 and T3 on the Hugging Face Hub, each with a model card (metrics, size, latency, training data, limitations, Spider attribution).
3. **Evaluation artifacts** — predictions files, metrics JSON, ablation table, failure-taxonomy tables.
4. **Demo** — runs locally with two commands (backend + frontend), or one after a production build.
5. **Docs** — this PRD, [specs.md](specs.md), [design.md](design.md), README.
6. **Final report — DOCX + PDF**, written after the project is complete, with these required sections and their sources:

| Report section | Built from |
|---|---|
| Title | e.g. *Spinneret: Small, Fast Text-to-SQL with Schema-Ranked T5* |
| Abstract | Problem, method (three tiers), headline EM/EX and latency, key finding from ablations and taxonomy |
| Problem Statement | §2 of this PRD |
| Proposed Work | §1, §3, §5 of this PRD + method overview from design.md |
| Detailed Description | specs.md §4–§8 (data, ranker, serialization, training, decoding, execution, evaluation) + design.md architecture and UI |
| Datasets | specs.md §4 (Spider statistics, splits, difficulty levels, preprocessing) |
| Results | Tier comparison (accuracy vs size vs latency), per-difficulty breakdown, ablations, failure taxonomy, best/worst question types, demo screenshots |

## 11. Constraints and assumptions

- **One developer, 5 weeks.** Priorities in §5 decide what gets cut. If time runs out, T2 Lite is the reportable fallback for T3; P2 items are cut first.
- **Compute:** Colab free tier verified from the terminal (T4, 15 GB, Python 3.13 on the VM, 12 GB RAM). Free sessions are time-limited and can be reclaimed, so training runs in the background on the VM, pushes a checkpoint to the Hub every epoch, and resumes from it. The laptop GTX 1650 can train T1 if Colab is unavailable.
- **Accounts:** Hugging Face (`hf auth login`, account `iteshxt`) and Google Colab (`colab` CLI login) are set up by the owner. Tokens are never pasted into chat, code or config.
- **Dataset:** Spider 1.0 only. Training uses `train_spider.json` + `train_others.json` (8,659 questions). The dev set (1,034 questions) is the benchmark, as in nearly all published work; hyper-parameters are tuned on a held-out slice of the training databases, never on dev. Some published baselines train on the 7,000 `train_spider` questions only — an ablation measures the difference.
- **Python 3.12** locally (via `uv`) — the system Python is 3.14. The Colab VM runs 3.13; `uv` installs 3.12 there too, so both sides match.
- Spider's original exact-match metric ignores literal values; execution accuracy needs them. Spinneret predicts full SQL with values.

## 12. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| T3 falls short of 65 / 72 | Medium | Medium | Ablations show where points are lost; Gate A (larger ranker) and Gate B (CodeT5-base); tune ranker *k*; report honestly against published references. |
| RoBERTa-base ranker drops needed tables/columns | Medium | High | Measure coverage first (Gate A); keep generous *k*; force-include gold items when building *training* inputs; RoBERTa-large as the upgrade. |
| Colab free tier limits (session length, reclaim, GPU availability) | High | High | Background training on the VM; Hub checkpoint every epoch and resume; laptop fallback for T1; start Colab runs early in week 2. |
| Colab CLI is young (v0.8, breaking changes) | Medium | Medium | Pin the installed version; keep the workflow in small scripts so a CLI change is a one-file fix; Colab web UI as manual fallback. |
| Slow PyPI downloads on the home network | Medium | Low | Dependencies install on the Colab VM (fast network); reuse the local `uv` cache; longer `UV_HTTP_TIMEOUT`. |
| T5 tokenizer cannot encode `<` → broken SQL | Certain (known issue) | High | Tokenizer round-trip audit; add missing tokens; unit test. CodeT5's byte-level tokenizer avoids it. |
| Large schemas exceed 512 tokens | Medium | Medium | Ranked filtering (T2/T3); relevance-ordered truncation (T1); report truncated share. |
| Wrong literal values hurt EX | High | Medium | Value matching; report EM (value-free) next to EX; "EX with gold values" diagnostic. |
| Exact-match marks correct-but-different SQL wrong | Certain | Low | Report EX alongside EM; taxonomy has a "semantically equivalent" bucket. |
| Schedule slip (solo, several models) | High | High | T1 and ranker by week 2; Colab training in parallel with frontend work; small-model ablations are cheap; strict P0/P1/P2 cuts. |

## 13. Decisions log (resolved questions)

| # | Question | Decision |
|---|---|---|
| Q1 | Where do checkpoints live? | Hugging Face Hub (account `iteshxt`), one model repo per checkpoint, with model cards. Visibility (public/private) confirmed before the first upload. |
| Q2 | Include `train_others.json`? | **Yes** — 8,659 training questions. Ablation without it for comparability. |
| Q3 | Final report format? | **DOCX + PDF**, written after the project is complete. |
| Q4 | Is 65% EM / 78% EX realistic? | Not with T5-small (≈ 46/46). 65% EM is the target for T3; 78% EX is a stretch goal. |
| Q5 | Bigger model or smaller and faster? | **Smaller and faster with a ranker:** T3 = T5-base + RoBERTa-base ranker as the main model, T2 Lite = T5-small + the same ranker as fast mode. RoBERTa-large ranker only if coverage is too low. |
| Q6 | Where does training run? | **Colab free T4 via the Colab CLI**, from the laptop terminal (verified 2026-11-21). Laptop as fallback for T1. |

## 14. References

- Yu et al., *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task*, EMNLP 2018. [arXiv:1809.08887](https://arxiv.org/abs/1809.08887)
- Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5), JMLR 2020. [arXiv:1910.10683](https://arxiv.org/abs/1910.10683)
- Wang et al., *CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation*, EMNLP 2021. [arXiv:2109.00859](https://arxiv.org/abs/2109.00859)
- Liu et al., *RoBERTa: A Robustly Optimized BERT Pretraining Approach*, 2019. [arXiv:1907.11692](https://arxiv.org/abs/1907.11692)
- Shaw et al., *Compositional Generalization and Natural Language Variation*, ACL 2021. [arXiv:2010.12725](https://arxiv.org/abs/2010.12725)
- Scholak et al., *PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models*, EMNLP 2021. [arXiv:2109.05093](https://arxiv.org/abs/2109.05093)
- Xie et al., *UnifiedSKG*, EMNLP 2022. [arXiv:2201.05966](https://arxiv.org/abs/2201.05966)
- Lin et al., *Bridging Textual and Tabular Data for Cross-Domain Text-to-SQL Semantic Parsing* (BRIDGE), Findings of EMNLP 2020. [arXiv:2012.12627](https://arxiv.org/abs/2012.12627)
- Li et al., *RESDSQL: Decoupling Schema Linking and Skeleton Parsing for Text-to-SQL*, AAAI 2023. [arXiv:2302.05965](https://arxiv.org/abs/2302.05965)
- Wang et al., *Robust Text-to-SQL Generation with Execution-Guided Decoding*, 2018. [arXiv:1807.03100](https://arxiv.org/abs/1807.03100)
- Zhong, Yu & Klein, *Semantic Evaluation for Text-to-SQL with Distilled Test Suites*, EMNLP 2020. [arXiv:2010.02840](https://arxiv.org/abs/2010.02840)
- Pourreza & Rafiei, *DIN-SQL*, NeurIPS 2023. [arXiv:2304.11015](https://arxiv.org/abs/2304.11015)
- Rajkumar et al., *Evaluating the Text-to-SQL Capabilities of Large Language Models*, 2022. [arXiv:2204.00498](https://arxiv.org/abs/2204.00498)
- Nguyen, Phan & Piccardi, *Fine-tuning text-to-SQL models with reinforcement-learning training objectives*, Natural Language Processing Journal 10 (2025) 100135.
- Chen et al., *Text-to-SQL Error Correction with Language Models of Code*, ACL 2023. [arXiv:2305.13073](https://arxiv.org/abs/2305.13073)
- Google, *Colab CLI*. [github.com/googlecolab/google-colab-cli](https://github.com/googlecolab/google-colab-cli)
