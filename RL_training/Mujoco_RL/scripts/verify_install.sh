#!/usr/bin/env bash
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate "${1:-mujoco_rl}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

python - <<'PY'
import gymnasium
import mujoco
import stable_baselines3
import torch
print("gymnasium", gymnasium.__version__)
print("mujoco", mujoco.__version__)
print("stable_baselines3", stable_baselines3.__version__)
print("torch", torch.__version__)
print("torch.cuda.is_available()", torch.cuda.is_available())
PY
