#!/usr/bin/env bash
# Live preview with auto-reload: http://127.0.0.1:8001
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"

exec .venv-docs/bin/mkdocs serve --dev-addr "${HOST}:${PORT}"
