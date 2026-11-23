#!/usr/bin/env bash
# Set up the on-demand live demo (design §13): nothing runs until you start it.
#
#   scripts/host/install-service.sh              # install the service and the launcher
#   scripts/host/install-service.sh --at-login   # ...and also start the demo at every login
#   scripts/host/install-service.sh --remove     # stop and uninstall everything
#
# 1. checks the Python environment and that the served models are in the Hugging Face
#    cache (the service runs with HF_HUB_OFFLINE=1)
# 2. installs ~/.config/systemd/user/spinneret-demo.service (from spinneret-demo.service);
#    it restarts by itself if it crashes while on
# 3. adds a "Spinneret demo" launcher to the app menu: a click runs `demo.sh toggle`
#    (on / off, with a notification); right-click for Start or Stop
# 4. makes sure Tailscale Funnel is allowed for the tailnet (the first time this prints a
#    link to approve), then leaves the funnel closed until the demo is started
#
# Then: the launcher, or scripts/host/demo.sh start|stop|status. While on, the models hold
# about 2 GB of GPU memory and port 8000. Logs: journalctl --user -u spinneret-demo -f
set -euo pipefail

ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
NAME=spinneret-demo.service
UNIT="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/$NAME"
LAUNCHER="${XDG_DATA_HOME:-$HOME/.local/share}/applications/spinneret-demo.desktop"

if [[ "${1:-}" == "--remove" ]]; then
  systemctl --user disable --now "$NAME" 2>/dev/null || true
  rm -f "$UNIT" "$LAUNCHER"
  systemctl --user daemon-reload
  tailscale funnel --https=443 off 2>/dev/null || true
  echo "Removed the service and the launcher, and closed the funnel."
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

mkdir -p "$(dirname "$UNIT")" "$(dirname "$LAUNCHER")"
sed "s|@ROOT@|$ROOT|g" "$ROOT/scripts/host/spinneret-demo.service" >"$UNIT"
systemctl --user daemon-reload
if [[ "${1:-}" == "--at-login" ]]; then
  systemctl --user enable "$NAME"
  echo "The demo will also start at every login."
else
  systemctl --user disable "$NAME" 2>/dev/null || true
fi
echo "Installed $UNIT."

DEMO="$ROOT/scripts/host/demo.sh"
cat >"$LAUNCHER" <<EOF
[Desktop Entry]
Type=Application
Name=Spinneret demo
GenericName=Live demo on/off
Comment=Turn the Spinneret live demo on or off (models on this laptop's GPU)
Exec=$DEMO toggle
Icon=$ROOT/scripts/host/spinneret-demo.svg
Terminal=false
Categories=Development;
Keywords=spinneret;demo;sql;
Actions=start;stop;

[Desktop Action start]
Name=Start demo
Exec=$DEMO start

[Desktop Action stop]
Name=Stop demo
Exec=$DEMO stop
EOF
command -v update-desktop-database >/dev/null && update-desktop-database -q "$(dirname "$LAUNCHER")" || true
echo "Added the \"Spinneret demo\" launcher to the app menu."

# allow Funnel once now (the first time this prints an approval link and waits), so a click
# on the launcher never stalls on it; then close it again unless the demo is running
tailscale funnel --bg 8000
systemctl --user is-active --quiet "$NAME" || tailscale funnel --https=443 off >/dev/null
echo "Done. Start the demo from the app menu, or: scripts/host/demo.sh start"
