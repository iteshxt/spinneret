#!/usr/bin/env bash
# Runs ON the Colab VM (called by start.sh): Python 3.12 env, Spider data, GPU check.
set -euo pipefail
cd /content/spinneret
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null; then
  python3 -m pip install --quiet uv
fi
uv python install 3.12
uv sync --extra train --quiet
uv run spinneret download   # databases are needed for execution accuracy
if [[ -f data/processed/stats.json ]]; then
  echo "using prepared data shipped from the laptop"
else
  uv run spinneret prepare --quiet
fi

echo "--- environment"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
uv run python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
uv run python -c "from huggingface_hub import whoami; print('hugging face user:', whoami()['name'])" \
  || echo "hugging face: not logged in"
