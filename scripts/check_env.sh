#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -x ".venv/bin/python" ]; then
  echo "Missing .venv. Run: bash scripts/setup_env.sh --reset" >&2
  exit 1
fi

export TRANSFORMERS_NO_TF=1
export USE_TF=0
export USE_FLAX=0
export HF_HOME="$ROOT_DIR/.cache/huggingface"
export TORCH_HOME="$ROOT_DIR/.cache/torch"
export MPLCONFIGDIR="$ROOT_DIR/.cache/matplotlib"

.venv/bin/python - <<'PY'
import importlib.util
import site
import sys

import datasets
import streamlit
import torch
import transformers

print("python:", sys.executable)
print("prefix:", sys.prefix)
print("base_prefix:", sys.base_prefix)
print("site_packages:", site.getsitepackages())
print("tensorflow_found:", importlib.util.find_spec("tensorflow") is not None)
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("streamlit:", streamlit.__version__)

assert sys.prefix != sys.base_prefix, "Python is not running inside .venv"
assert all(".venv" in path for path in site.getsitepackages()), "site-packages is not isolated"
assert importlib.util.find_spec("tensorflow") is None, "TensorFlow should not be installed in this project env"
PY
