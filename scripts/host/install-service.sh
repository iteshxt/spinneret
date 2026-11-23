#!/usr/bin/env bash
# Keep the live demo on whenever this laptop is on and you are logged in (design §13).
#
#   scripts/host/install-service.sh            # install, start, and open the funnel
#   scripts/host/install-service.sh --remove   # stop, uninstall, and close the funnel
#
# 1. checks the Python environment and that the served models are in the Hugging Face
#    cache (the service runs with HF_HUB_OFFLINE=1)
# 2. installs ~/.config/systemd/user/spinneret-demo.service from spinneret-demo.service and
#    starts it now and at every login; it restarts by itself if it crashes
# 3. turns Tailscale Funnel on for port 8000 and leaves it on (it survives reboots); while the
#    server is down the funnel answers with an error page and the site shows "offline"
#
# While it runs, the models hold about 2 GB of GPU memory and port 8000 is taken. To pause it:
#   systemctl --user stop spinneret-demo      (start it again with `start`)
# Logs: journalctl --user -u spinneret-demo -f
set -euo pipefail

ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
NAME=spinneret-demo.service
UNIT="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/$NAME"

if [[ "${1:-}" == "--remove" ]]; then
  systemctl --user disable --now "$NAME" 2>/dev/null || true
  rm -f "$UNIT"
  systemctl --user daemon-reload
  tailscale funnel --https=443 off 2>/dev/null || true
  echo "Removed $NAME and closed the funnel."
  exit 0
fi

if [[ ! -x "$ROOT/.venv/bin/spinneret" ]]; then
  echo "No environment in $ROOT/.venv; run: uv sync --extra train --extra api" >&2
  exit 1
fi
[[ -f "$ROOT/web/dist/index.html" ]] ||
  echo "Note: web/dist is missing, so the laptop's own address serves the API only (cd web && npm run build)."

echo "Checking that the served models are in the Hugging Face cache…"
(cd "$ROOT" && HF_HUB_OFFLINE=1 .venv/bin/python - <<'EOF'
import sys
from pathlib import Path
from spinneret.config import load_serve_config
from spinneret.hub import resolve_checkpoint

# resolve exactly as the service will (offline, same files), then look for the weights
cfg = load_serve_config(Path("configs/serve.yaml"))
refs = list(dict.fromkeys([m.generator for m in cfg.models] + list(cfg.rankers.values())))
missing = []
for ref in refs:
    try:
        folder = resolve_checkpoint(ref)
        if not any(folder.glob("*.safetensors")) and not any(folder.glob("*.bin")):
            missing.append(ref)
    except Exception:
        missing.append(ref)
if missing:
    sys.exit("Not cached: " + ", ".join(missing) + ". Run `uv run spinneret serve` once online.")
print(f"All {len(refs)} models are cached.")
EOF
)

mkdir -p "$(dirname "$UNIT")"
sed "s|@ROOT@|$ROOT|g" "$ROOT/scripts/host/spinneret-demo.service" >"$UNIT"
systemctl --user daemon-reload
systemctl --user enable --now "$NAME"
echo "Installed $UNIT and started it."

# the first time, this prints a link to allow Funnel for the tailnet and waits until you do
tailscale funnel --bg 8000
HOST="$(tailscale status --json | python3 -c 'import json, sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))')"
echo "Live API: https://$HOST (models take about a minute to load after each start)"
