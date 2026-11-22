#!/usr/bin/env bash
# Push one training run as a private Kaggle script kernel on a T4 (specs §5.10).
#
#   scripts/kaggle/push.sh <run> <spinneret train args…>
#   e.g. scripts/kaggle/push.sh e3 --config configs/ablations/e3_t1_no_values.yaml
#
# Then:  kaggle kernels status <user>/spinneret-<run>
#        kaggle kernels output <user>/spinneret-<run> -p runs/kaggle/<run>
# KAGGLE_NO_RUN=1 saves the kernel without running it (e.g. to attach secrets first).
# For Hub checkpoints, attach the HF_TOKEN secret to the kernel in Kaggle's editor
# (Add-ons → Secrets) — the token itself never passes through this script.
set -euo pipefail
RUN="${1:?usage: push.sh <run> <spinneret train args…>}"; shift
[[ "$RUN" =~ ^[a-z0-9-]+$ ]] || { echo "run name: lower-case letters, digits and - only" >&2; exit 1; }
[[ $# -gt 0 ]] || { echo "missing spinneret train arguments" >&2; exit 1; }
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
USER_NAME="$(python3 "$ROOT/scripts/kaggle/user.py")"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

ARGS_PY="$(python3 -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "$@")"
python3 - "$ROOT/scripts/kaggle/kernel.py" "$WORK/spinneret-$RUN.py" "$RUN" "$ARGS_PY" <<'PY'
import sys
src, dst, run, args = sys.argv[1:]
code = open(src).read().replace('"__RUN__"', repr(run)).replace('"__ARGS__"', repr(args))
open(dst, "w").write(code)
PY
cat > "$WORK/kernel-metadata.json" <<JSON
{
  "id": "$USER_NAME/spinneret-$RUN",
  "title": "spinneret-$RUN",
  "code_file": "spinneret-$RUN.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["$USER_NAME/spinneret-bundle"],
  "competition_sources": [],
  "kernel_sources": []
}
JSON
kaggle kernels push -p "$WORK" --accelerator NvidiaTeslaT4 ${KAGGLE_NO_RUN:+--no-run}
echo "[spinneret] pushed $USER_NAME/spinneret-$RUN — status: kaggle kernels status $USER_NAME/spinneret-$RUN"
