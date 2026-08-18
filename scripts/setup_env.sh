#!/usr/bin/env bash
# Rebuild the verified environment for FreeTraining-OVSS.
#
# The version pins are NOT optional:
#   SAM 3            requires torch >= 2.3  (torch.nn.attention)
#   mmcv wheels      exist for torch 2.1-2.4, none for 2.5
#   mmseg 1.2.2      asserts mmcv >= 2.0.0rc4, < 2.2.0
# -> torch 2.4.1 is the only workable version; its mmcv wheel is 2.2.0,
#    so mmseg's MMCV_MAX must be raised to 2.3.0.
#
# Usage:  bash scripts/setup_env.sh
set -euo pipefail

ENV_NAME="${ENV_NAME:-segov3}"
PY_VER="3.11"

echo "==> Creating conda env '$ENV_NAME' (python $PY_VER)"
conda create -n "$ENV_NAME" python="$PY_VER" -y
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "==> torch 2.4.1 + cu121"
pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2"

python - <<'PY'
import torch, sys
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("ERROR: CUDA unavailable — stop and fix before continuing.")
PY

echo "==> mmcv 2.2.0 (prebuilt wheel — do NOT build from source)"
pip install mmcv==2.2.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
pip install "mmsegmentation==1.2.2"

echo "==> Patching mmseg MMCV_MAX 2.2.0 -> 2.3.0"
MMSEG_INIT="$CONDA_PREFIX/lib/python$PY_VER/site-packages/mmseg/__init__.py"
sed -i "s/MMCV_MAX = '2.2.0'/MMCV_MAX = '2.3.0'/" "$MMSEG_INIT"
grep MMCV_MAX "$MMSEG_INIT"

echo "==> SAM 3 runtime dependencies"
pip install einops psutil pycocotools hydra-core iopath timm \
            huggingface_hub omegaconf

echo "==> Verifying"
python - <<'PY'
import torch, mmcv, mmseg
from mmcv.ops import point_sample          # fails on mmcv-lite
print("OK  torch", torch.__version__,
      "| mmcv", mmcv.__version__,
      "| mmseg", mmseg.__version__,
      "| cuda", torch.cuda.is_available())
PY

mkdir -p "$HOME/logs"
pip freeze > "$HOME/logs/${ENV_NAME}-env-freeze.txt"

cat <<'EOF'

Environment ready.

Next:
  1. Download the SAM 3 checkpoint:
       huggingface-cli download facebook/sam3
     then symlink into the reference repo:
       mkdir -p weights/sam3
       ln -s "$(ls ~/.cache/huggingface/hub/models--facebook--sam3/snapshots/*/sam3.pt)" \
         weights/sam3/sam3.pt
  2. Prepare LoveDA (note the doubled Val/Val nesting in the Kaggle archive)
  3. python demo.py
EOF