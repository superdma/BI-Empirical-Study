#!/usr/bin/env bash
# Build the static documentation site into ./site and verify internal links.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x .venv-docs/bin/mkdocs ]]; then
  cat >&2 <<'EOF'
mkdocs is not installed.  Create the local environment first:

  uv venv .venv-docs --python 3.12
  uv pip install --python .venv-docs/bin/python -r requirements-docs.txt
EOF
  exit 1
fi

.venv-docs/bin/mkdocs build --clean "$@"
python3 scripts/check_site_links.py site

echo "built: $ROOT/site ($(du -sh site | cut -f1))"
