#!/usr/bin/env bash
# Keep a Colab session alive while background spinneret jobs run, printing progress.
#
#   scripts/colab/keepalive.sh <session> [minutes_between_checks]
#
# Colab decides a VM is idle from notebook-kernel activity, and jobs started by
# train.sh run detached from the kernel — without activity the free tier reclaims the
# VM mid-training (it happened on 2026-11-21). Every few minutes this sends one short
# status call (which counts as kernel activity) and prints each job's latest epoch.
# Dropped calls are simply retried next round. It exits when no spinneret job is left.
# Limitation: this laptop must stay awake and online while it runs.
set -euo pipefail

SESSION="${1:?usage: keepalive.sh <session> [minutes]}"
MINUTES="${2:-4}"
[[ "$MINUTES" =~ ^[0-9]+$ ]] || { echo "minutes must be a whole number" >&2; exit 1; }
source "$(dirname "$0")/_lib.sh"
RETRIES=1  # a missed check is fine; the next one comes in a few minutes

STATUS_CODE=$(cat <<'PY'
import glob, json, os, subprocess, time

def latest(path):
    last = None
    with open(path, errors="replace") as fh:
        for line in fh:
            if any(f'"event": "{e}"' in line for e in ("epoch", "early_stop", "done")):
                last = line
    if not last:
        return "starting"
    d = json.loads(last)
    keys = ("event", "epoch", "valid_em", "valid_ex", "valid_coverage", "best_epoch")
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items() if k in keys}

procs = subprocess.run(["pgrep", "-f", "bin/spinneret"], capture_output=True, text=True).stdout.split()
logs = [p for p in sorted(glob.glob("/content/logs/*.log")) if not p.endswith("bootstrap.log")]
report = {os.path.basename(p)[:-4]: latest(p) for p in logs}
print("KEEPALIVE", time.strftime("%H:%M UTC"), f"{len(procs)} job process(es)", json.dumps(report))
print("KEEPALIVE_JOBS", len(procs))
PY
)

misses=0
while true; do
  out="$(printf '%s\n' "$STATUS_CODE" | CEXEC_WALL=120 cexec "$SESSION" --timeout 60 2>/dev/null || true)"
  if grep -q "^KEEPALIVE " <<<"$out"; then
    misses=0
    grep "^KEEPALIVE " <<<"$out" | sed 's/^KEEPALIVE //'
    if grep -q "^KEEPALIVE_JOBS 0$" <<<"$out"; then
      echo "[spinneret] no spinneret job left — stop the session with scripts/colab/stop.sh $SESSION"
      exit 0
    fi
  else
    misses=$((misses + 1))
    echo "$(date +%H:%M) [spinneret] status call failed ($misses in a row); retrying next round" >&2
    if ! session_exists "$SESSION"; then
      echo "[spinneret] session '$SESSION' is gone (reclaimed?) — re-run start.sh + train.sh with --resume hub" >&2
      exit 1
    fi
  fi
  sleep $((MINUTES * 60))
done
