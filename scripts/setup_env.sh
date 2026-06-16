#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p .cache/huggingface .cache/torch .cache/matplotlib

if [ "${1:-}" = "--reset" ]; then
  rm -rf .venv
fi

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

if [ "${INSTALL_RAG_DEPS:-0}" = "1" ]; then
  .venv/bin/python -m pip install -r requirements-rag.txt
fi

.venv/bin/python - <<'PY'
import sys
import site

print("Python:", sys.executable)
print("Prefix:", sys.prefix)
print("Base prefix:", sys.base_prefix)
print("Site packages:", site.getsitepackages())
assert sys.prefix != sys.base_prefix, "Not running inside the project virtualenv"
PY
