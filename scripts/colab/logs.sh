#!/usr/bin/env bash
# Show the tail of a run's log on the Colab VM.
#
#   scripts/colab/logs.sh <session> <run> [lines]
set -euo pipefail

SESSION="${1:?usage: logs.sh <session> <run> [lines]}"
RUN="${2:?missing run name}"
LINES="${3:-40}"
[[ "$RUN" =~ ^[A-Za-z0-9._-]+$ && "$LINES" =~ ^[0-9]+$ ]] || { echo "bad arguments" >&2; exit 1; }

echo "!tail -n $LINES /content/logs/$RUN.log; echo; pgrep -af 'spinneret' | grep -v pgrep || echo '(no spinneret process running)'" \
  | colab exec -s "$SESSION"
