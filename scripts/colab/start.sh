#!/usr/bin/env bash
# Open a free Colab T4 session and get it ready to run Spinneret (specs §5.9).
#
#   scripts/colab/start.sh <session>
#
# 1. colab new --gpu T4
# 2. ship the working tree (tracked + untracked-but-not-ignored files; never data,
#    checkpoints or .env files) as a tarball
# 3. copy the local Hugging Face login *file* to the VM, so the token never appears
#    in a command line, environment variable or log
# 4. run bootstrap.sh on the VM
set -euo pipefail

SESSION="${1:?usage: start.sh <session>}"
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
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

colab new --gpu T4 -s "$SESSION"
echo "!mkdir -p $REMOTE /root/.cache/huggingface /content/logs" | colab exec -s "$SESSION"
colab upload -s "$SESSION" "$BUNDLE" /content/spinneret.tar.gz
if [[ -f "$HF_TOKEN_FILE" ]]; then
  colab upload -s "$SESSION" "$HF_TOKEN_FILE" /root/.cache/huggingface/token
  echo "!chmod 600 /root/.cache/huggingface/token" | colab exec -s "$SESSION"
else
  echo "[spinneret] no Hugging Face login found at $HF_TOKEN_FILE — Hub pushes will fail" >&2
fi
# `colab exec` exits 0 even when the remote command fails, so check for a marker.
echo "!tar -xzf /content/spinneret.tar.gz -C $REMOTE && bash $REMOTE/scripts/colab/bootstrap.sh && echo SPINNERET_BOOTSTRAP_OK" \
  | colab exec -s "$SESSION" --timeout 0 | tee "$WORK/bootstrap.log"
if ! grep -q SPINNERET_BOOTSTRAP_OK "$WORK/bootstrap.log"; then
  echo "[spinneret] bootstrap FAILED on '$SESSION' (output above). Stop it with: scripts/colab/stop.sh $SESSION" >&2
  exit 1
fi
echo "[spinneret] session '$SESSION' ready. Next: scripts/colab/train.sh $SESSION <run> <spinneret args…>"
