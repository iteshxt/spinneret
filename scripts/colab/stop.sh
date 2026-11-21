#!/usr/bin/env bash
# Stop a Colab session and confirm nothing is still assigned (saves the free GPU quota).
#
#   scripts/colab/stop.sh <session>
set -euo pipefail

SESSION="${1:?usage: stop.sh <session>}"
colab stop -s "$SESSION" || echo "[spinneret] stop reported an error; checking server state…" >&2
colab sessions
# The usage counter can lag ~1 minute behind the stop.
for _ in 1 2 3 4 5 6; do
  if colab usage | grep -q "Active assignments: 0"; then
    echo "[spinneret] no active Colab assignments."
    exit 0
  fi
  sleep 15
done
colab usage
echo "[spinneret] still assigned — check https://colab.research.google.com (Manage sessions)" >&2
exit 1
