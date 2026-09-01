#!/usr/bin/env bash
# Repoint ConInfer's hardcoded absolute paths at this machine.
#
# ConInfer_segmentor.py lines ~172-180 hardcode the first author's own cluster:
#     REPO_DIR   = "/data/users/cwy/0_datasets/github_source_packge/dinov3-main"
#     WEIGHT_DIR = "/data/users/cwy/0_datasets/model_weight/DINOV3/dinov3_*.pth"
# The repo VENDORS dinov3 at ConInfer/dinov3 (hubconf.py is present), so REPO_DIR
# is a pure repointing. The weights are not shipped and are gated by Meta.
#
# ⚠️ Keeps a .orig backup and prints a diff, because editing a competitor's source
# before measuring it is exactly the kind of change that has to be visible and
# reversible. Every edit here is a PATH -- no logic, no hyperparameter, nothing
# that could change their result.
#
#   bash scripts/patch_coninfer_paths.sh /path/to/dinov3_..._sat493m-....pth
#
# ⚠️ DINOv3 IS MANDATORY, INCLUDING FOR THEIR BASELINE CONFIG. The if/else at
# line ~171 selects WHICH backbone, not whether to load one, and
# configs_baseline/base_config1.py also builds `ConInferSegmentation`. There is
# no lighter path for validating the environment first.
set -euo pipefail

SRC="${SRC:-$HOME/ConInfer/ConInfer_segmentor.py}"
W="${1:-}"
REPO="$HOME/ConInfer/dinov3"

[ -f "$SRC" ] || { echo "⛔ not found: $SRC" >&2; exit 1; }
[ -f "$REPO/hubconf.py" ] || { echo "⛔ no hubconf.py under $REPO" >&2; exit 1; }
if [ -z "$W" ]; then
  echo "usage: bash scripts/patch_coninfer_paths.sh /path/to/dinov3_*.pth" >&2
  echo >&2
  echo "The DINOv3 satellite checkpoint is gated by Meta and is not in their repo." >&2
  echo "Request access, download it, then pass its path here." >&2
  grep -n 'WEIGHT_DIR' "$SRC" | sed 's/^/  wants: /' >&2
  exit 1
fi
[ -f "$W" ] || { echo "⛔ weights not found: $W" >&2; exit 1; }

[ -f "$SRC.orig" ] || cp "$SRC" "$SRC.orig"     # first run only: keep pristine
cp "$SRC.orig" "$SRC"                            # always patch from pristine

python3 - "$SRC" "$REPO" "$W" <<'PY'
import re, sys
src, repo, w = sys.argv[1], sys.argv[2], sys.argv[3]
t = open(src).read()
# REPO_DIR is the same vendored checkout for both branches -- rewrite all.
t, n1 = re.subn(r'REPO_DIR\s*=\s*"[^"]*"', f'REPO_DIR = "{repo}"', t)

# ⚠️ WEIGHT_DIR is NOT the same. The if/else picks WHICH DINOv3 to load:
#     UAVid / VDD  -> dinov3_vith16plus, lvd1689m weights
#     everything else -> dinov3_vitl16,  sat493m weights   <- our datasets
# Rewriting both would point the vith16plus branch at vitl16 weights, which is
# an architecture/checkpoint mismatch that only fires on a dataset we do not run
# -- i.e. silently wrong. Patch ONLY the branch matching the given file.
key = 'sat493m' if 'sat493m' in w else ('lvd1689m' if 'lvd1689m' in w else None)
if key is None:
    raise SystemExit(
        f"⛔ cannot tell which branch {w} belongs to.\n"
        "   Expected a filename containing 'sat493m' (vitl16, used by LoveDA,\n"
        "   OpenEarthMap, Potsdam, Vaihingen) or 'lvd1689m' (vith16plus, UAVid/VDD).")
t, n2 = re.subn(rf'WEIGHT_DIR\s*=\s*"[^"]*{key}[^"]*"', f'WEIGHT_DIR = "{w}"', t)
open(src, 'w').write(t)
print(f"  REPO_DIR   rewritten {n1}x -> {repo}")
print(f"  WEIGHT_DIR rewritten {n2}x ({key} branch only) -> {w}")
if n1 == 0 or n2 != 1:
    raise SystemExit(
        f"⛔ expected REPO_DIR >=1 and exactly one {key} WEIGHT_DIR; "
        f"got {n1} and {n2}. Inspect by hand.")
PY

echo
echo "--- diff against pristine (paths only) ---"
diff "$SRC.orig" "$SRC" || true
echo
echo "⚠️ Record this in the paper's reproducibility note: their published code"
echo "   hardcodes the first author's filesystem, so two paths had to be edited."
echo "   Restore any time with:  cp $SRC.orig $SRC"
