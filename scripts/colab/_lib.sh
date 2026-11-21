# Shared helpers for scripts/colab/*.sh (sourced, not executed).
#
# The Colab CLI's connection to the notebook kernel sometimes drops
# ("RuntimeError: Connection was lost"). These wrappers retry, restarting the kernel
# in between — safe because every command we send is idempotent and background
# jobs run in their own process sessions.

RETRIES="${SPINNERET_COLAB_RETRIES:-4}"

# cexec <session> [colab exec args…] < code   — streams output; retries on dropped connections
cexec() {
  local session="$1"; shift
  local code out status attempt
  code="$(cat)"
  out="$(mktemp)"
  for ((attempt = 1; attempt <= RETRIES; attempt++)); do
    set +e
    # CEXEC_WALL: wall-clock limit in seconds (0 = none); a hung call counts as dropped.
    printf '%s\n' "$code" | timeout "${CEXEC_WALL:-0}" colab exec -s "$session" "$@" 2>&1 | tee "$out"
    status=${PIPESTATUS[1]}
    set -e
    if [[ $status -eq 0 ]] && ! grep -q "Connection was lost" "$out"; then
      rm -f "$out"; return 0
    fi
    if { [[ $status -eq 124 ]] || grep -q "Connection was lost" "$out"; } && [[ $attempt -lt $RETRIES ]]; then
      echo "[spinneret] connection lost or hung — restarting kernel and retrying ($attempt/$RETRIES)" >&2
      colab restart-kernel -s "$session" >/dev/null 2>&1 || true
      sleep 10
      continue
    fi
    rm -f "$out"; return 1
  done
  rm -f "$out"; return 1
}

# cupload <session> <local> <remote>   — retries transient failures
cupload() {
  local attempt
  for ((attempt = 1; attempt <= RETRIES; attempt++)); do
    if timeout 300 colab upload -s "$1" "$2" "$3"; then return 0; fi
    echo "[spinneret] upload failed — retrying ($attempt/$RETRIES)" >&2
    colab restart-kernel -s "$1" >/dev/null 2>&1 || true
    sleep 10
  done
  return 1
}

# rbg <session> <name> <bash command>
# Start a detached job on the VM (own process session, log /content/logs/<name>.log).
# Idempotent: a pid file prevents a second copy while the first is alive.
rbg() {
  local session="$1" name="$2" cmd_json
  cmd_json="$(python3 -c 'import json, sys; print(json.dumps(sys.argv[1]))' "$3")"
  CEXEC_WALL=180 cexec "$session" --timeout 120 <<PY | grep "SPINNERET_STARTED $name"
import os, subprocess
os.makedirs("/content/logs", exist_ok=True)
pid_file = "/content/logs/$name.pid"

def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

if os.path.exists(pid_file) and alive(int(open(pid_file).read().strip() or 0)):
    print("SPINNERET_STARTED $name (already running)")
else:
    env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"],
               PYTHONUNBUFFERED="1")
    log = open("/content/logs/$name.log", "ab")
    proc = subprocess.Popen(["bash", "-c", $cmd_json], cwd="/content", stdout=log,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                            start_new_session=True)
    open(pid_file, "w").write(str(proc.pid))
    print("SPINNERET_STARTED $name pid", proc.pid)
PY
}

# rwait <session> <name> <marker>
# Poll a detached job's log with short calls, printing new lines, until <marker>
# appears (return 0) or the job exits without it (return 1). Dropped calls are retried.
rwait() {
  local session="$1" name="$2" marker="$3" shown=0 out n
  while true; do
    out="$(CEXEC_WALL=120 cexec "$session" --timeout 60 2>/dev/null <<PY || true
import os
log, pid_file = "/content/logs/$name.log", "/content/logs/$name.pid"
lines = open(log, errors="replace").read().splitlines() if os.path.exists(log) else []
alive = False
if os.path.exists(pid_file):
    try:
        os.kill(int(open(pid_file).read().strip() or 0), 0)
        alive = True
    except OSError:
        pass
print("RWAIT_LINES", len(lines))
print("RWAIT_ALIVE", alive)
for line in lines[$shown:]:
    print("RWAIT|" + line)
PY
)"
    grep "^RWAIT|" <<<"$out" | sed 's/^RWAIT|//' || true
    n="$(grep -m1 "^RWAIT_LINES" <<<"$out" | awk '{print $2}')"
    [[ -n "$n" ]] && shown="$n"
    if grep -q "$marker" <<<"$out"; then return 0; fi
    if grep -q "^RWAIT_ALIVE False" <<<"$out"; then return 1; fi
    sleep 30
  done
}
