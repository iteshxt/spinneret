#!/usr/bin/env bash
# Create or update the PRIVATE Kaggle dataset <user>/spinneret-bundle that kernels read:
# the working tree (as start.sh ships it to Colab — never data, checkpoints, .env or
# token files) plus the locally prepared data (data/processed without E5 inputs, cache).
#
#   scripts/kaggle/dataset.sh
#
# Needs the Kaggle CLI logged in (~/.kaggle/access_token or kaggle.json).
set -euo pipefail
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$ROOT"
USER_NAME="$(python3 "$ROOT/scripts/kaggle/user.py")"
SLUG=spinneret-bundle
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

{
  git ls-files -z --cached --recurse-submodules
  git ls-files -z --others --exclude-standard
} | tar --null -T - -czf "$WORK/spinneret-code.tar.gz"
if tar -tzf "$WORK/spinneret-code.tar.gz" | grep -E '(^|/)(\.env[^/]*|token|kaggle\.json)$' >/dev/null; then
  echo "refusing to upload: bundle contains a .env, token or kaggle.json file" >&2
  exit 1
fi
tar -czf "$WORK/spinneret-data.tar.gz" -C data --exclude='processed/ablations/ranked-k*' processed cache
du -h "$WORK"/*.tar.gz

cat > "$WORK/dataset-metadata.json" <<JSON
{"title": "spinneret-bundle", "id": "$USER_NAME/$SLUG", "licenses": [{"name": "CC-BY-SA-4.0"}]}
JSON
if kaggle datasets status "$USER_NAME/$SLUG" >/dev/null 2>&1; then
  kaggle datasets version -p "$WORK" -m "update $(git rev-parse --short HEAD)" -r skip
else
  kaggle datasets create -p "$WORK" -r skip   # private unless --public is passed
fi
echo "[spinneret] dataset $USER_NAME/$SLUG uploaded; wait until it shows 'ready': kaggle datasets status $USER_NAME/$SLUG"
