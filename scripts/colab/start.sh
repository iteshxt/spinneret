#!/usr/bin/env bash
# Open (or reuse) a free Colab T4 session and get it ready to run Spinneret (specs §5.9).
#
#   scripts/colab/start.sh <session>
#
# 1. colab new --gpu T4 (skipped if the session already exists)
# 2. ship the working tree (tracked + untracked-but-not-ignored files; never data,
#    checkpoints or .env files) as a tarball, plus the locally prepared data (≈1 MB)
# 3. copy the local Hugging Face login *file* to the VM, so the token never appears
#    in a command line, environment variable or log
# 4. run bootstrap.sh on the VM
# Every step is idempotent, so the whole script can simply be re-run after a failure.
set -euo pipefail

SESSION="${1:?usage: start.sh <session>}"
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source "$ROOT/scripts/colab/_lib.sh"
REMOTE=/content/spinneret
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
BUNDLE="$WORK/spinneret.tar.gz"
HF_TOKEN_FILE="${HF_HOME:-$HOME/.cache/huggingface}/token"

cd "$ROOT"
{
  git ls-files -z --cached --recurse-submodules
  git ls-files -z --others --exclude-standard
} | tar --null -T - -czf "$BUNDLE"
if tar -tzf "$BUNDLE" | grep -E '(^|/)(\.env[^/]*|token)$' >/dev/null; then
  echo "refusing to ship: bundle contains a .env or token file" >&2
  exit 1
fi
echo "[spinneret] bundle: $(du -h "$BUNDLE" | cut -f1)"

# Ship locally prepared data so the VM can skip `spinneret prepare`, which is slow on
# Colab's CPUs. Re-run `spinneret prepare` locally after changing data code.
DATA_BUNDLE=""
if [[ -f data/processed/stats.json ]]; then
  DATA_BUNDLE="$WORK/spinneret-data.tar.gz"
  tar -czf "$DATA_BUNDLE" -C data processed $( [[ -d data/cache ]] && echo cache )
  echo "[spinneret] data bundle: $(du -h "$DATA_BUNDLE" | cut -f1)"
fi

if session_exists "$SESSION"; then
  echo "[spinneret] reusing session '$SESSION'"
else
  colab new --gpu T4 -s "$SESSION"
  sleep 15  # the first exec right after allocation tends to drop its connection
fi
echo "!mkdir -p $REMOTE /root/.cache/huggingface /content/logs" | CEXEC_WALL=180 cexec "$SESSION" >/dev/null
cupload "$SESSION" "$BUNDLE" /content/spinneret.tar.gz
UNPACK_DATA=""
if [[ -n "$DATA_BUNDLE" ]]; then
  cupload "$SESSION" "$DATA_BUNDLE" /content/spinneret-data.tar.gz
  UNPACK_DATA="mkdir -p $REMOTE/data && tar -xzf /content/spinneret-data.tar.gz -C $REMOTE/data && "
fi
if [[ -f "$HF_TOKEN_FILE" ]]; then
  cupload "$SESSION" "$HF_TOKEN_FILE" /root/.cache/huggingface/token
  echo "!chmod 600 /root/.cache/huggingface/token" | CEXEC_WALL=180 cexec "$SESSION" >/dev/null
else
  echo "[spinneret] no Hugging Face login found at $HF_TOKEN_FILE — Hub pushes will fail" >&2
fi

# Bootstrap runs as a detached job polled with short calls: a long streaming
# `colab exec` does not survive the CLI's occasional dropped connections.
ALREADY="$(echo '!grep -q SPINNERET_BOOTSTRAP_OK /content/logs/bootstrap.log 2>/dev/null && echo BOOTSTRAPPED_BEFORE' \
  | CEXEC_WALL=120 cexec "$SESSION" || true)"
if grep -q "^BOOTSTRAPPED_BEFORE" <<<"$ALREADY"; then
  echo "[spinneret] '$SESSION' was already bootstrapped"
else
  rbg "$SESSION" bootstrap \
    "tar -xzf /content/spinneret.tar.gz -C $REMOTE && ${UNPACK_DATA}bash $REMOTE/scripts/colab/bootstrap.sh && echo SPINNERET_BOOTSTRAP_OK"
  if ! rwait "$SESSION" bootstrap SPINNERET_BOOTSTRAP_OK; then
    echo "[spinneret] bootstrap FAILED on '$SESSION' (log above). Re-run start.sh, or stop it with: scripts/colab/stop.sh $SESSION" >&2
    exit 1
  fi
fi
echo "[spinneret] session '$SESSION' ready. Next: scripts/colab/train.sh $SESSION <run> <spinneret args…>, then keepalive.sh $SESSION"
