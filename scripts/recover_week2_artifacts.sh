#!/usr/bin/env bash
# =============================================================================
# Step 2 — recover the Week 2 artefacts into version control.
#
# WHY THIS EXISTS
# ---------------
# `measure_discard_rate.py` has never been committed, on any branch:
#
#     git log --all --diff-filter=A --name-only | grep -i discard   ->  (nothing)
#
# Every number in WEEK1_RESULTS.md §7, §8 and §9 came from that script -- the
# 29.68% discard rate, the tau-sweep, all three confusion matrices, the 3:1
# error budget, the 1.73:1 recovery cost. Its outputs live only in
# ~/outputs/week2_tau* and `outputs/` is gitignored.
#
# So the empirical core of the thesis currently survives on one machine's
# untracked filesystem. This script copies it into the repo.
#
# RUN THIS ON THE LINUX WORKSTATION. It does not need the GPU.
# It stages nothing destructive and does not commit -- it shows you what it
# found and prints the commit command at the end.
# =============================================================================
set -u

REPO="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ -z "$REPO" ]; then
    echo "ERROR: not inside a git repository. cd into the repo first."
    exit 1
fi
OUT="${OUTPUTS_DIR:-$HOME/outputs}"
SEGEARTH="${SEGEARTH_DIR:-$HOME/SegEarth-OV-3}"
DEST="$REPO/results/week2"

echo "repo     : $REPO"
echo "outputs  : $OUT"
echo "segearth : $SEGEARTH"
echo

# ---------------------------------------------------------------------------
echo "###########  1. the script itself (highest priority)  ###########"
echo
FOUND=$(find "$HOME" -name 'measure_discard_rate.py' -not -path '*/node_modules/*' \
          -not -path '*/.git/*' 2>/dev/null | head -5)
if [ -z "$FOUND" ]; then
    echo "  !! measure_discard_rate.py NOT FOUND ANYWHERE ON THIS MACHINE."
    echo "  !! If it is truly gone, §7-§9 cannot be reproduced and must be"
    echo "  !! rewritten from scratch before the paper. Check backups first."
else
    echo "  found:"
    echo "$FOUND" | sed 's/^/    /'
    SRC=$(echo "$FOUND" | head -1)
    if [ ! -f "$REPO/scripts/measure_discard_rate.py" ]; then
        cp "$SRC" "$REPO/scripts/measure_discard_rate.py"
        echo "  -> copied into scripts/"
    else
        echo "  -> scripts/measure_discard_rate.py already exists; diffing:"
        diff -q "$SRC" "$REPO/scripts/measure_discard_rate.py" || \
            echo "     (differs -- reconcile by hand, NOT overwritten)"
    fi
fi
echo

# ---------------------------------------------------------------------------
echo "###########  2. the baseline config  ###########"
echo
for f in "$SEGEARTH/configs/cfg_loveda.py" "$SEGEARTH/configs/cls_loveda.txt"; do
    if [ -f "$f" ]; then
        mkdir -p "$REPO/configs/segearth"
        cp -n "$f" "$REPO/configs/segearth/" && echo "  copied $(basename "$f")"
    else
        echo "  missing: $f"
    fi
done
echo "  (these pin tau=0.5 and the class prompts -- WEEK1_RESULTS §4)"
echo

# ---------------------------------------------------------------------------
echo "###########  3. summary artefacts (small; NOT the .npz cache)  ###########"
echo
mkdir -p "$DEST"
for t in 0.5 0.3 0.1; do
    d="$OUT/week2_tau$t"
    if [ ! -d "$d" ]; then echo "  missing: $d"; continue; fi
    mkdir -p "$DEST/tau$t"
    n=0
    for f in "$d"/*.csv "$d"/*.md "$d"/confusion_matrix.npy; do
        [ -f "$f" ] || continue
        sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
        if [ "$sz" -gt 20000000 ]; then
            echo "  SKIP (>20MB) $(basename "$f")"
            continue
        fi
        cp -n "$f" "$DEST/tau$t/" && n=$((n+1))
    done
    echo "  tau=$t -> $n file(s)"
done
echo
echo "  NOTE: do NOT copy the per-image .npz cache (~2.5 GB). It is a"
echo "        derived artefact and belongs in ~/outputs, which is gitignored."
echo

# ---------------------------------------------------------------------------
echo "###########  4. the values that fill the remaining _TBD_ fields  ###########"
echo
echo "--- WEEK1_RESULTS §7.2, tau=0.3 headline row ---"
head -20 "$OUT/week2_tau0.3/discard_summary.md" 2>/dev/null || echo "  (not found)"
echo
for t in 0.5 0.3 0.1; do
    echo "--- §7.3 per-class discard, tau=$t ---"
    cat "$OUT/week2_tau$t/discard_per_class.csv" 2>/dev/null || echo "  (not found)"
    echo
done
echo "  >>> Transcribe the road / barren / building rows into §7.3 and the"
echo "      tau=0.3 row into §7.2, then delete the '_TBD_' markers."
echo

# ---------------------------------------------------------------------------
echo "###########  5. size check before staging  ###########"
echo
du -sh "$DEST" 2>/dev/null
echo
find "$DEST" "$REPO/scripts/measure_discard_rate.py" -type f 2>/dev/null \
  | while read -r f; do
      sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
      [ "$sz" -gt 5000000 ] && echo "  WARNING ${sz} bytes  $f"
    done
echo "  (nothing listed above = clean)"
echo

echo "############################################################"
echo " Review the above, then:"
echo
echo "   cd \"$REPO\""
echo "   git add scripts/measure_discard_rate.py configs/segearth results/week2"
echo "   git status --short"
echo "   git commit -m 'fix: bring Week 2 discard diagnostic under version control"
echo
echo "   measure_discard_rate.py had never been committed on any branch, so"
echo "   WEEK1_RESULTS §7-§9 (29.68%% discard, tau-sweep, confusion matrices,"
echo "   3:1 error budget) were unreproducible from the repo and survived only"
echo "   in untracked ~/outputs on one machine."
echo
echo "   - add the script, cfg_loveda.py and the per-tau summary CSVs"
echo "   - .npz cache deliberately excluded (~2.5 GB, derived)'"
echo "   git push"
echo
echo " Then fill the _TBD_ fields from section 4 above and commit that separately."
echo "############################################################"
