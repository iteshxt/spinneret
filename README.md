# Spinneret

Small, fast text-to-SQL. Spinneret translates plain-English questions into executable SQL for a given SQLite database, using fine-tuned T5 models and a schema ranker, trained on the [Spider](https://yale-lily.github.io/spider) benchmark.

| Tier | Models | Role |
|---|---|---|
| Baseline | T5-small, full schema | reference point |
| Lite | T5-small + RoBERTa-base schema ranker | fast mode |
| Spinneret | T5-base + the same ranker | main model |

Status: **week 1 — data pipeline**. See [docs/prd.md](docs/prd.md), [docs/specs.md](docs/specs.md) and [docs/design.md](docs/design.md).

## Setup

```bash
uv sync                      # Python 3.12 environment with the data pipeline
uv run spinneret download    # Spider 1.0 into data/raw/spider (CC BY-SA 4.0)
uv run spinneret prepare     # processed splits + stats into data/processed
uv run pytest -m "not slow"  # fast tests
```

### If PyPI downloads crawl

Some networks throttle PyPI's file CDN. Install the exact locked versions from a mirror instead — the SHA-256 hashes in `uv.lock` are enforced, so any file that differs from PyPI's is rejected:

```bash
uv export --frozen --format requirements.txt --no-emit-project --group dev -o /tmp/req.txt
uv venv --python 3.12
uv pip install --require-hashes -r /tmp/req.txt \
  --index https://mirrors.aliyun.com/pypi/simple/ --index https://pypi.tuna.tsinghua.edu.cn/simple \
  --default-index https://pypi.org/simple --index-strategy unsafe-first-match
uv pip install --no-deps -e .
uv sync --frozen --offline    # confirms the environment matches uv.lock
```

Then use `uv run --frozen …`.

## Training on Colab

Training runs on a free Colab T4, driven from this machine with Google's [Colab CLI](https://github.com/googlecolab/google-colab-cli) (`uv tool install git+https://github.com/googlecolab/google-colab-cli`, then log in once with `colab new`):

```bash
scripts/colab/start.sh t1                    # T4 session + code + HF login file + bootstrap
scripts/colab/train.sh t1 <run> <spinneret args…>   # background job on the VM
scripts/colab/logs.sh t1 <run>               # follow it
scripts/colab/stop.sh t1                     # release the GPU
```

## Data licence

Spider is distributed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) by Yu et al. (EMNLP 2018). It is downloaded by script and never committed to this repository.
