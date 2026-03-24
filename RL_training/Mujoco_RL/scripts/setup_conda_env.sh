#!/usr/bin/env bash
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh

ENV_NAME="${1:-mujoco_rl}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  if conda env list | awk '{print $1}' | grep -qx "env_isaaclab"; then
    conda create -y -n "${ENV_NAME}" --clone env_isaaclab
  else
    conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
  fi
fi

conda activate "${ENV_NAME}"

python -m pip install --upgrade pip wheel setuptools

if ! python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("torch") else 1)
PY
then
  if command -v nvidia-smi >/dev/null 2>&1; then
    python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
  else
    python -m pip install torch
  fi
fi

python -m pip install -r "$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)/requirements.txt"

python - <<'PY'
import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
PY
