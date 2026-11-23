#!/usr/bin/env bash
# Serve the live demo from this laptop's GPU to the hosted website (design §13).
#
#   scripts/host/serve.sh [port]      # default 8000; Ctrl-C stops the server and the tunnel
#
# 1. open a Tailscale Funnel: https://<machine>.<tailnet>.ts.net -> http://127.0.0.1:<port>
#    (TLS ends on this laptop, so Tailscale relays the traffic without reading it)
# 2. run `spinneret serve` on 127.0.0.1, kept awake with systemd-inhibit while it runs
# 3. on exit, close the funnel again, so nothing is exposed while the demo is off
#
# One-time setup (README › Hosting): install Tailscale, `sudo tailscale up`,
# `sudo tailscale set --operator=$USER`, and allow Funnel when the first run asks.
set -euo pipefail

PORT="${1:-8000}"
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$ROOT"

if ! command -v tailscale >/dev/null; then
  echo "Tailscale is not installed; see README › Hosting." >&2
  exit 1
fi
if ! tailscale status >/dev/null 2>&1; then
  echo "Tailscale is not connected; run: sudo tailscale up" >&2
  exit 1
fi
HOST="$(tailscale status --json | python3 -c 'import json, sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))')"

tailscale funnel --bg "$PORT"  # the first run prints a link to allow Funnel for the tailnet
trap 'tailscale funnel --https=443 off >/dev/null 2>&1 || true; echo "Funnel closed."' EXIT

echo "Live API: https://$HOST  (the hosted site's VITE_API_URL; its origin must be in cors_origins)"
inhibit=()
command -v systemd-inhibit >/dev/null &&
  inhibit=(systemd-inhibit --what=idle:sleep --who=spinneret --why="serving the Spinneret demo")
"${inhibit[@]}" uv run --extra api --extra train spinneret serve --port "$PORT"
