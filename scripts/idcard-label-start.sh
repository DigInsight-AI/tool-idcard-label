#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$PYTHON_BIN"
else
  if [[ ! -x "$ROOT_DIR/.venv/bin/python" ]]; then
    echo "Creating project virtual environment: $ROOT_DIR/.venv"
    python3 -m venv "$ROOT_DIR/.venv"
  fi
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
fi
REVIEW_HOST="${REVIEW_HOST:-127.0.0.1}"
REVIEW_PORT="${REVIEW_PORT:-8765}"
REVIEW_WORKSPACE_CLASS_ID="${REVIEW_WORKSPACE_CLASS_ID:-1}"

ensure_model_dependencies() {
  if "$PYTHON_BIN" -c 'import ultralytics' >/dev/null 2>&1; then
    return
  fi

  echo "Installing local model dependencies from requirements.txt..."
  "$PYTHON_BIN" -m pip install --requirement "$ROOT_DIR/requirements.txt"
}

find_port_listeners() {
  LISTENER_PIDS=()
  while IFS= read -r pid; do
    [[ -n "$pid" ]] && LISTENER_PIDS+=("$pid")
  done < <(lsof -nP -tiTCP:"$REVIEW_PORT" -sTCP:LISTEN 2>/dev/null || true)
}

stop_port_listeners() {
  command -v lsof >/dev/null 2>&1 || {
    echo "Cannot check port $REVIEW_PORT because lsof is not installed." >&2
    exit 1
  }

  find_port_listeners
  if (( ${#LISTENER_PIDS[@]} == 0 )); then
    return
  fi

  echo "Port $REVIEW_PORT is in use; stopping PID(s): ${LISTENER_PIDS[*]}"
  kill "${LISTENER_PIDS[@]}"

  for ((attempt = 0; attempt < 50; attempt++)); do
    sleep 0.1
    find_port_listeners
    if (( ${#LISTENER_PIDS[@]} == 0 )); then
      return
    fi
  done

  echo "Port $REVIEW_PORT is still in use; force stopping PID(s): ${LISTENER_PIDS[*]}"
  kill -KILL "${LISTENER_PIDS[@]}"
}

cd "$ROOT_DIR"

ensure_model_dependencies

args=(
  serve
  --host "$REVIEW_HOST"
  --port "$REVIEW_PORT"
  --workspace-class-id "$REVIEW_WORKSPACE_CLASS_ID"
)

if [[ "${REVIEW_OPEN_BROWSER:-0}" != "1" ]]; then
  args+=(--no-browser)
fi

stop_port_listeners

echo "Starting ID Card Label Tool: http://$REVIEW_HOST:$REVIEW_PORT"
echo "Keep this terminal open; press Ctrl+C to stop the server."

export PYTHONUNBUFFERED=1
exec "$PYTHON_BIN" -m idcard_label "${args[@]}" "$@"
