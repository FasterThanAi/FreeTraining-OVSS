#!/usr/bin/env bash
# Install DLRSD into the SegEarth-OV-3 clone: vocabulary, config, dataset class
# and the data symlink.
#
# ⭐ IDEMPOTENT AND REVERSIBLE. custom_datasets.py is backed up to .orig once
# (never overwritten on a re-run, so the pristine copy survives), the append is
# skipped if DLRSDDataset is already defined, and a diff is printed. Same shape
# as scripts/patch_coninfer_paths.sh, for the same reason: an edit inside the
# baseline clone that nobody can see later is how a reproduction quietly stops
# being one.
#
#   bash scripts/install_dlrsd.sh --data ~/data/dlrsd --repo ~/SegEarth-OV-3
set -euo pipefail

DATA=""; REPO="$HOME/SegEarth-OV-3"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) DATA="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$DATA" ]] || { echo "⛔ --data is required (the output of prepare_dlrsd.py)" >&2; exit 2; }
DATA="${DATA/#\~/$HOME}"; REPO="${REPO/#\~/$HOME}"

[[ -d "$REPO/configs" ]] || { echo "⛔ $REPO/configs not found" >&2; exit 1; }
CD="$REPO/custom_datasets.py"
[[ -f "$CD" ]] || { echo "⛔ $CD not found" >&2; exit 1; }

for d in "$DATA/img_dir/val" "$DATA/ann_dir/val"; do
  [[ -d "$d" ]] || { echo "⛔ $d not found — run prepare_dlrsd.py first" >&2; exit 1; }
done
# ⚠️ `wc -l` left-pads its output on BSD, so a string compare against "2100"
# fails on a directory that holds exactly 2100 files. Strip it and compare as
# integers -- a count check that silently never fires is worse than no check.
NI=$(ls "$DATA/img_dir/val" | wc -l | tr -d '[:space:]')
NA=$(ls "$DATA/ann_dir/val" | wc -l | tr -d '[:space:]')
echo "  data: $NI images / $NA labels"
[[ "$NI" -eq "$NA" ]] || { echo "⛔ image/label counts differ" >&2; exit 1; }
[[ "$NI" -eq 2100 ]] || echo "  ⚠️ expected 2100, found $NI — not the full DLRSD"

# ---- 1. vocabulary and config
for f in configs/segearth/cls_dlrsd.txt configs/segearth/cfg_dlrsd.py \
         configs/segearth/dlrsd_dataset.py; do
  [[ -f "$HERE/$f" ]] || { echo "⛔ $HERE/$f not found. Run this script from a "\
    "checkout of FreeTraining-OVSS (it resolves its sources relative to "\
    "itself), not from a copy." >&2; exit 1; }
done
cp "$HERE/configs/segearth/cls_dlrsd.txt" "$REPO/configs/cls_dlrsd.txt"
cp "$HERE/configs/segearth/cfg_dlrsd.py"  "$REPO/configs/cfg_dlrsd.py"
# the corrected-vocabulary arm (prereg/predict_dlrsd_vocabulary.md), if present
for f in cls_dlrsd_v2.txt cfg_dlrsd_v2.py; do
  [[ -f "$HERE/configs/segearth/$f" ]] && cp "$HERE/configs/segearth/$f" "$REPO/configs/$f" \
    && echo "  ✅ configs/$f  (vocabulary arm)"
done
echo "  ✅ configs/cls_dlrsd.txt  ($(wc -l < "$REPO/configs/cls_dlrsd.txt") classes)"
echo "  ✅ configs/cfg_dlrsd.py"

# ⛔ cfg_dlrsd.py deliberately does NOT set confidence_threshold, so it inherits
# from _base_. That value changes SAM 3's DECODER behaviour and therefore
# seg_logits themselves -- a generated config once hardcoded 0.5 and silently
# ran a different model from the baseline (@ARGMAX_SCALING_RESULTS.md). Print
# whatever is inherited so it lands in the log rather than being assumed.
BASE="$REPO/configs/base_config.py"
if [[ -f "$BASE" ]]; then
  echo "  inherited from base_config.py:"
  grep -nE "confidence_threshold|prob_thd|slide_crop" "$BASE" | sed 's/^/    /' || \
    echo "    (none set in base_config.py — check the segmentor defaults)"
fi

# ---- 2. dataset class
if grep -q "class DLRSDDataset" "$CD"; then
  echo "  ✅ DLRSDDataset already present — not appending again"
else
  [[ -f "$CD.orig" ]] || cp "$CD" "$CD.orig"
  printf '\n\n' >> "$CD"
  cat "$HERE/configs/segearth/dlrsd_dataset.py" >> "$CD"
  echo "  ✅ DLRSDDataset appended (backup at custom_datasets.py.orig)"
  echo "  --- diff ---"
  diff "$CD.orig" "$CD" | sed 's/^/    /' | head -50 || true
fi

# ---- 3. data symlink. `ln -sfn`, never `ln -s`: with an existing directory
# `ln -s target dest` creates the link INSIDE dest, which produced a
# `images/val/val -> images/val` loop on OpenEarthMap (WEEK3 §11).
mkdir -p "$REPO/data"
ln -sfn "$DATA" "$REPO/data/DLRSD"
echo "  ✅ data/DLRSD -> $(readlink "$REPO/data/DLRSD")"

# ---- 4. the class list must match the dataset class, or the rows are mislabelled
# ⚠️ `python` is not guaranteed to be on PATH (the Mac side of this project
# has only python3). Resolve whichever exists rather than assuming.
PY_BIN="$(command -v python || command -v python3)"
"$PY_BIN" - "$HERE" "$CD" <<'PY'
import re, sys
here, cd = sys.argv[1], sys.argv[2]
vocab = [l.strip() for l in open(f'{here}/configs/segearth/cls_dlrsd.txt') if l.strip()]
src = open(cd).read()
i = src.index('class DLRSDDataset')
block = src[i:i + 2000]
m = re.search(r'classes=\((.*?)\)', block, re.S)
meta = [x.strip().strip("'\"") for x in m.group(1).split(',') if x.strip()]
print(f'  cls_dlrsd.txt : {len(vocab)} prompts')
print(f'  METAINFO      : {len(meta)} classes')
if vocab != meta:
    raise SystemExit(
        f'⛔ THE PROMPT LIST AND THE METAINFO DISAGREE.\n'
        f'   prompts : {vocab}\n'
        f'   metainfo: {meta}\n'
        f'   The i-th prompt IS class i. A mismatch produces a clean table '
        f'with the rows named wrongly, and nothing crashes (WEEK3 §11).')
print('  ✅ prompt i == class i, verified')
PY

cat <<'MSG'

✅ installed. Next, in order:

  1. python ~/FreeTraining-OVSS/scripts/test_dlrsd_sink.py   # CPU, must pass first
  2. the baseline run                                        # ~1.5 h on the GPU

⚠️ There is NO published DLRSD number to gate against. The substitutes are the
   pixel-exact label accounting from prepare_dlrsd.py and the category
   fingerprint from dlrsd_class_map.py, and both are weaker than a gate.
MSG
