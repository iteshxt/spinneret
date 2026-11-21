#!/usr/bin/env bash
# Start a spinneret command in the background on the Colab VM; returns immediately.
#
#   scripts/colab/train.sh <session> <run> <spinneret args…>
#   e.g. scripts/colab/train.sh t1 t5-small train --config configs/t5_small.yaml --resume hub
#
# Output goes to /content/logs/<run>.log on the VM (follow it with logs.sh). The job
# keeps running if this laptop sleeps or disconnects; checkpoints go to the Hub.
set -euo pipefail

SESSION="${1:?usage: train.sh <session> <run> <spinneret args…>}"
RUN="${2:?missing run name}"
shift 2
[[ $# -gt 0 ]] || { echo "missing spinneret arguments" >&2; exit 1; }
[[ "$RUN" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "run name: letters, digits, . _ - only" >&2; exit 1; }

ARGS="$(printf '%q ' "$@")"
LOG="/content/logs/$RUN.log"
OUT="$(echo "!cd /content/spinneret && mkdir -p /content/logs && setsid nohup env PATH=\$HOME/.local/bin:\$PATH uv run spinneret $ARGS > $LOG 2>&1 < /dev/null & echo SPINNERET_STARTED $RUN" \
  | colab exec -s "$SESSION")"
echo "$OUT"
grep -q "SPINNERET_STARTED $RUN" <<<"$OUT" || { echo "[spinneret] could not start '$RUN'" >&2; exit 1; }
echo "[spinneret] '$RUN' running in the background; follow it with scripts/colab/logs.sh $SESSION $RUN"
