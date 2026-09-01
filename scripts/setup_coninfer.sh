#!/usr/bin/env bash
# ConInfer (arXiv:2603.29271) in an environment that CANNOT damage `segov3`.
#
# ⛔ WHY THIS SCRIPT EXISTS AND WHY IT IS PARANOID.
# `segov3` is a three-way version deadlock -- torch 2.4.1+cu121, mmcv 2.2.0 from
# the prebuilt wheel index, mmsegmentation 1.2.2 with MMCV_MAX patched to 2.3.0 --
# and it is the ONLY combination found to work after five failed attempts
# (WEEK1_RESULTS.md 2). Every number in this project rests on it. A single
# `pip install` run against the wrong active environment destroys a day and, worse,
# could silently change the baseline everything is measured against.
#
# So this script:
#   1. REFUSES to run if `segov3` is the active conda environment;
#   2. creates a separate env `coninfer` and never touches the other;
#   3. records `segov3`'s package list before and after, and diffs them, so a
#      leak is caught by a test rather than by a mIoU that quietly moved.
#
# ⚠️ It does NOT pip-install ConInfer's requirements blindly -- their repo pins
# are unknown until it is cloned. It gets you a clean env and a clone; read their
# README and install inside `coninfer` only.
#
#   bash scripts/setup_coninfer.sh
set -euo pipefail

if [ "${1:-}" = "--verify" ]; then
  SNAP_DIR="${HOME}/logs/coninfer_guard"
  B="$SNAP_DIR/segov3_before.txt"; A="$SNAP_DIR/segov3_after.txt"
  [ -f "$B" ] || { echo "no baseline snapshot at $B -- run without --verify first" >&2; exit 1; }
  conda run -n segov3 python -m pip freeze > "$A"
  if diff -q "$B" "$A" >/dev/null; then
    echo "✅ 'segov3' is byte-identical to before ConInfer work began."
  else
    echo "⛔ 'segov3' HAS CHANGED. This is the failure this guard exists to catch:"
    diff "$B" "$A" || true
    echo
    echo "Restore with scripts/setup_env.sh, then re-run the LoveDA gate:"
    echo "  it must still report 47.37 mIoU and 29.68% discard."
    exit 1
  fi
  # the real gate is behavioural, not a package list
  echo
  echo "⚠️ A matching package list is necessary, not sufficient. Before trusting"
  echo "   any further measurement, re-run the baseline gate:"
  echo "     cd ~/SegEarth-OV-3 && python eval.py ./configs/cfg_loveda.py   # 47.38"
  exit 0
fi

GUARD_ENV="segov3"
NEW_ENV="coninfer"
CLONE_DIR="${HOME}/ConInfer"
SNAP_DIR="${HOME}/logs/coninfer_guard"
mkdir -p "$SNAP_DIR"

# ---- 1. refuse to run from inside the protected environment ----------------
ACTIVE="${CONDA_DEFAULT_ENV:-none}"
if [ "$ACTIVE" = "$GUARD_ENV" ]; then
  cat >&2 <<EOF
⛔ REFUSING TO RUN: the active conda environment is '$GUARD_ENV'.

That environment is the three-way version deadlock every number in this project
depends on, and it took five failed attempts to find. Nothing for ConInfer may
be installed while it is active.

    conda deactivate
    bash scripts/setup_coninfer.sh
EOF
  exit 1
fi
echo "✅ active env is '$ACTIVE', not '$GUARD_ENV'"

# ---- 2. snapshot the protected environment BEFORE ---------------------------
if conda env list | awk '{print $1}' | grep -qx "$GUARD_ENV"; then
  conda run -n "$GUARD_ENV" python -m pip freeze > "$SNAP_DIR/segov3_before.txt" 2>/dev/null
  echo "✅ snapshot: $(wc -l < "$SNAP_DIR/segov3_before.txt") packages in '$GUARD_ENV'"
else
  echo "!! '$GUARD_ENV' not found -- are you on the workstation?" >&2; exit 1
fi

# ---- 3. the new environment -------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$NEW_ENV"; then
  echo "   env '$NEW_ENV' already exists, leaving it alone"
else
  conda create -n "$NEW_ENV" python=3.11 -y
  echo "✅ created env '$NEW_ENV'"
fi

# ---- 4. the clone -----------------------------------------------------------
if [ -d "$CLONE_DIR/.git" ]; then
  echo "   clone already at $CLONE_DIR"
else
  git clone https://github.com/Dog-Yang/ConInfer.git "$CLONE_DIR"
fi

cat <<EOF

────────────────────────────────────────────────────────────────────
NEXT, BY HAND -- and read their README first, do not guess the pins:

    conda activate $NEW_ENV
    cd $CLONE_DIR
    less README.md              # note torch / mmcv / CUDA requirements
    # install ONLY inside '$NEW_ENV'

⚠️ If their README asks for a torch or mmcv version that conflicts with
   '$GUARD_ENV', that is FINE. Separate environments is the entire point.
   Never "fix" it by changing '$GUARD_ENV'.

WHEN DONE, VERIFY NOTHING LEAKED:

    bash scripts/setup_coninfer.sh --verify

────────────────────────────────────────────────────────────────────
EOF
