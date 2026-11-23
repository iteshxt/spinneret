#!/usr/bin/env bash
# Turn the live demo on or off, on demand (design §13). The "Spinneret demo" launcher that
# install-service.sh adds to the app menu runs `toggle`; right-click it for Start / Stop.
#
#   scripts/host/demo.sh start    # load the models, open the funnel; live in about a minute
#   scripts/host/demo.sh stop     # stop the models (frees the GPU) and close the funnel
#   scripts/host/demo.sh toggle   # whichever of the two applies
#   scripts/host/demo.sh status
#
# While it is off, nothing listens and nothing is exposed: the hosted site shows "offline".
set -euo pipefail

NAME=spinneret-demo.service
SITE=https://spinneret-beta.vercel.app
HEALTH=http://127.0.0.1:8000/api/health
ICON="$(cd "$(dirname "$0")" && pwd)/spinneret-demo.svg"

say() {
  echo "$1"
  command -v notify-send >/dev/null && notify-send -a Spinneret -i "$ICON" "Spinneret demo" "$1" || true
}

running() { systemctl --user is-active --quiet "$NAME"; }

health() { curl -s -m 3 "$HEALTH" | python3 -c 'import json, sys; print(json.load(sys.stdin)["status"])' 2>/dev/null; }

start() {
  if ! systemctl --user cat "$NAME" >/dev/null 2>&1; then
    say "Not installed yet: run scripts/host/install-service.sh"
    exit 1
  fi
  if running; then
    say "Already running: $SITE"
    return
  fi
  systemctl --user start "$NAME"
  tailscale funnel --bg 8000 >/dev/null
  say "Starting: loading the models (about a minute)…"
  for _ in $(seq 90); do # up to 3 minutes
    case "$(health)" in
      ok) say "Live: $SITE is answering questions"; return ;;
      error) say "The models failed to load; see: journalctl --user -u spinneret-demo"; exit 1 ;;
    esac
    running || { say "The server stopped; see: journalctl --user -u spinneret-demo"; exit 1; }
    sleep 2
  done
  say "Still loading after 3 minutes; see: journalctl --user -u spinneret-demo"
}

stop() {
  systemctl --user stop "$NAME" 2>/dev/null || true
  tailscale funnel --https=443 off >/dev/null 2>&1 || true
  say "Stopped: GPU memory freed and the tunnel closed"
}

status() {
  if running; then echo "on ($(health || echo starting))"; else echo "off"; fi
  tailscale funnel status 2>/dev/null | grep -q "Funnel on" && echo "funnel: open" || echo "funnel: closed"
}

case "${1:-status}" in
  start) start ;;
  stop) stop ;;
  toggle) if running; then stop; else start; fi ;;
  status) status ;;
  *) echo "usage: $0 start|stop|toggle|status" >&2; exit 2 ;;
esac
