#!/usr/bin/env bash
# Start a spinneret command in the background on the Colab VM; returns immediately.
#
#   scripts/colab/train.sh <session> <run> <spinneret args…>
#   e.g. scripts/colab/train.sh w3 t1 train --config configs/t5_small.yaml --resume hub
#
# Output goes to /content/logs/<run>.log on the VM. The job runs in its own process
# session (start_new_session), so it survives the laptop sleeping, the CLI
# disconnecting and kernel restarts; checkpoints go to the Hub. A pid file makes the
# launch idempotent: re-running (or a retried exec) will not start a second copy.
# Then run keepalive.sh, or the free tier reclaims the "idle" VM.
set -euo pipefail

SESSION="${1:?usage: train.sh <session> <run> <spinneret args…>}"
RUN="${2:?missing run name}"
shift 2
[[ $# -gt 0 ]] || { echo "missing spinneret arguments" >&2; exit 1; }
[[ "$RUN" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "run name: letters, digits, . _ - only" >&2; exit 1; }
source "$(dirname "$0")/_lib.sh"

ARGS_JSON="$(python3 -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "$@")"
CODE=$(cat <<PY
import json, os, subprocess
os.makedirs("/content/logs", exist_ok=True)
pid_file = "/content/logs/$RUN.pid"

def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

if os.path.exists(pid_file) and alive(int(open(pid_file).read().strip() or 0)):
    print("SPINNERET_STARTED $RUN (already running, pid", open(pid_file).read().strip() + ")")
else:
    env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"],
               PYTHONUNBUFFERED="1")
    log = open("/content/logs/$RUN.log", "ab")
    proc = subprocess.Popen(["uv", "run", "spinneret", *json.loads('''$ARGS_JSON''')],
                            cwd="/content/spinneret", stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, env=env, start_new_session=True)
    open(pid_file, "w").write(str(proc.pid))
    print("SPINNERET_STARTED $RUN pid", proc.pid)
PY
)
OUT="$(printf '%s\n' "$CODE" | CEXEC_WALL=180 cexec "$SESSION" --timeout 120)"
grep -q "SPINNERET_STARTED $RUN" <<<"$OUT" || { echo "$OUT"; echo "[spinneret] could not start '$RUN'" >&2; exit 1; }
grep "SPINNERET_STARTED" <<<"$OUT"
echo "[spinneret] '$RUN' running in the background. Run scripts/colab/keepalive.sh $SESSION after starting all jobs."
