#!/usr/bin/env bash
# Install ConInfer's dependencies using the stack THIS MACHINE can actually build.
#
# WHY WE DEVIATE FROM THEIR PIN. requirements.txt asks for torch==2.7.1 with
# mmcv>=2.1.0. No prebuilt mmcv wheel exists for torch 2.7, so pip falls back to a
# source build, which fails twice over on this machine:
#   1. mmcv's setup.py imports `pkg_resources`, gone from setuptools >= 70;
#   2. even patched, torch refuses to compile CUDA extensions because system nvcc
#      is 13.3 against torch's 12.x  (WEEK1_RESULTS.md section 2, lines 95-99).
# Their pin works on their machine because their nvcc matches their torch.
#
# ConInfer is CLIP/DINOv3-based and needs mmcv + mmsegmentation, exactly like
# SegEarth-OV3. So we install the combination this project already proved works --
# torch 2.4.1+cu121, mmcv 2.2.0 from the prebuilt index, mmseg 1.2.2 with MMCV_MAX
# patched -- and then everything else from their requirements.txt unchanged.
#
# ⚠️ THIS IS A DEVIATION AND MUST BE REPORTED. If their published numbers do not
# reproduce, the torch version is the first suspect. Reproduce their number BEFORE
# evaluating on our splits (CONINFER_RUNBOOK.md).
#
#   conda activate coninfer && bash scripts/install_coninfer_deps.sh
set -euo pipefail

REQ="${1:-$HOME/ConInfer/requirements.txt}"
[ -f "$REQ" ] || { echo "no requirements.txt at $REQ" >&2; exit 1; }

# ---- guards ----------------------------------------------------------------
ENV="${CONDA_DEFAULT_ENV:-none}"
if [ "$ENV" = "segov3" ]; then
  echo "⛔ REFUSING: 'segov3' is active. That environment is the only working" >&2
  echo "   combination found after five attempts and every number depends on it." >&2
  exit 1
fi
if [ "$ENV" != "coninfer" ]; then
  echo "⛔ REFUSING: active env is '$ENV', expected 'coninfer'." >&2
  echo "   conda activate coninfer" >&2
  exit 1
fi
PIPPATH="$(command -v pip)"
case "$PIPPATH" in
  *envs/coninfer/*) echo "✅ pip is $PIPPATH" ;;
  *) echo "⛔ pip resolves to $PIPPATH, outside the coninfer env. Stop." >&2; exit 1 ;;
esac

# ---- the stack that works here ---------------------------------------------
echo "→ torch 2.4.1+cu121 (their pin is 2.7.1; see the header for why)"
pip install torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu121
# ⚠️ NO `numpy<2` here. That pin belongs to `segov3`, not to ConInfer, whose own
# requirements ask for numpy==2.1.2. Forcing it caused an install/uninstall churn
# and a confusing resolver conflict against opencv-python. mmcv 2.2.0's compiled
# ops import fine against numpy 2.x on this machine -- verified below.
echo "→ mmcv 2.2.0 from the PREBUILT index (never source -- nvcc 13.3 vs torch 12.1)"
pip install mmcv==2.2.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
pip install mmengine==0.10.7 "mmsegmentation==1.2.2"

# mmseg 1.2.2 asserts mmcv < 2.2.0 and the wheel is exactly 2.2.0; the ops mmseg
# calls are unchanged between 2.1 and 2.2.
#
# ⚠️ DO NOT locate mmseg by importing it. Importing mmseg is what raises the
# assertion this patch exists to remove, so `import mmseg` fails and any path
# derived from it is empty -- the sed then silently patches nothing. Resolve the
# path from CONDA_PREFIX instead, the way scripts/setup_env.sh does.
PYVER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
INIT="$CONDA_PREFIX/lib/python$PYVER/site-packages/mmseg/__init__.py"
[ -f "$INIT" ] || { echo "⛔ mmseg __init__.py not at $INIT" >&2; exit 1; }
sed -i "s/MMCV_MAX = '2.2.0'/MMCV_MAX = '2.3.0'/" "$INIT"
# Verify the edit LANDED. The first version of this script assumed it had and
# hid the failure behind `2>/dev/null || true`, so the break surfaced later as a
# confusing AssertionError from an unrelated command.
grep -q "MMCV_MAX = '2.3.0'" "$INIT" || {
  echo "⛔ MMCV_MAX patch did not apply to $INIT" >&2
  grep -n 'MMCV_MAX' "$INIT" >&2 || true
  exit 1
}
echo "  ✅ MMCV_MAX patched to 2.3.0 in $INIT"

# ---- everything else from their file, untouched -----------------------------
# ⚠️ pydensecrf 1.0rc3 is EXCLUDED. Its sdist does not build against Cython 3 --
# `densecrf.pxd` fails to resolve `eigen.pxd` and every Eigen type errors out. It
# is a DenseCRF post-processing wrapper and may not be reached by the evaluation
# path at all, so we install everything else and then REPORT whether ConInfer
# actually imports it, rather than guessing either way.
REST="$(mktemp)"
grep -viE '^\s*(torch|torchvision|torchaudio|mmcv|mmsegmentation|mmengine|pydensecrf)\b' "$REQ" \
  | grep -vE '^\s*(#|$)' > "$REST" || true
echo "→ their remaining $(wc -l < "$REST") requirements (pydensecrf excluded, see above):"
sed 's/^/     /' "$REST"
[ -s "$REST" ] && pip install -r "$REST"
rm -f "$REST"

echo
echo "→ does ConInfer actually use pydensecrf?"
if grep -rIn --include='*.py' -e 'pydensecrf' -e 'densecrf' "$(dirname "$REQ")" 2>/dev/null | head -20 | grep -q .; then
  echo "  ⚠️ YES -- referenced at:"
  grep -rIn --include='*.py' -e 'pydensecrf' -e 'densecrf' "$(dirname "$REQ")" 2>/dev/null | head -20 | sed 's/^/     /'
  echo "  If the evaluation path reaches it, install it with Cython pinned:"
  echo "     pip install 'cython<3' && pip install --no-build-isolation pydensecrf==1.0rc3"
else
  echo "  ✅ no Python file references it; excluding it is safe."
fi

echo
python - <<'PY'
import torch
print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
import mmcv, mmseg, mmengine
print(f"mmcv {mmcv.__version__} | mmseg {mmseg.__version__} | mmengine {mmengine.__version__}")
from mmcv.ops import nms          # the compiled-ops import that mmcv-lite fails
print("mmcv compiled ops: OK")
PY
echo
echo "⚠️ Now verify segov3 is untouched:  bash scripts/setup_coninfer.sh --verify"
