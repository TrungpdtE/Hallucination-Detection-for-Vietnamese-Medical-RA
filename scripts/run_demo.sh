#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
else
  echo "Missing .venv. Run: bash scripts/setup_env.sh --reset" >&2
  exit 1
fi

PORT="${1:-8501}"

export TRANSFORMERS_NO_TF=1
export USE_TF=0
export USE_FLAX=0
export HF_HOME="$ROOT_DIR/.cache/huggingface"
export TORCH_HOME="$ROOT_DIR/.cache/torch"
export MPLCONFIGDIR="$ROOT_DIR/.cache/matplotlib"

exec "$PYTHON" -m streamlit run src/demo/app.py \
  --server.port "$PORT" \
  --server.headless true \
  --server.fileWatcherType none \
  --browser.gatherUsageStats false
