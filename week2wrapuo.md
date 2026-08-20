#!/usr/bin/env bash
# ============================================================
# Week 2 wrap-up — A2 to A5. No GPU needed.
# Run each block, read the output, act. Not fully automated on
# purpose: A3 and A5 need your eyes before anything is committed.
# ============================================================
set -u
REPO=~/"final year pro/Final_year_project"
OUT=~/outputs
SCRATCH=~/.gemini/antigravity/scratch/FreeTraining-OVSS/scripts

echo "###############  A2 — remaining TBD values  ###############"
echo
echo "--- tau=0.3 headline (fills WEEK1_RESULTS 7.2) ---"
head -20 "$OUT/week2_tau0.3/discard_summary.md" 2>/dev/null || echo "  (not found)"
echo
for t in 0.5 0.3 0.1; do
  echo "--- discard_per_class.csv  tau=$t  (fills 7.3) ---"
  cat "$OUT/week2_tau$t/discard_per_class.csv" 2>/dev/null || echo "  (not found)"
  echo
done

echo "###############  A3 — stranded scripts  ###############"
echo
echo "--- what exists in the scratch copy but not the repo? ---"
if [ -d "$SCRATCH" ]; then
    diff -rq "$SCRATCH/" "$REPO/scripts/" 2>&1 | grep -i 'only in' || echo "  (no differences)"
else
    echo "  scratch dir gone: $SCRATCH"
    echo "  >>> if sam3_smoke_test.py was never copied, it is LOST. Check git log."
fi
echo
echo "--- is cooccurrence_gt.py anywhere? (ANALYSIS 4 depends on it) ---"
find ~ -name 'cooccurrence_gt.py' -not -path '*/node_modules/*' 2>/dev/null \
  || echo "  NOT FOUND ANYWHERE"
echo
echo "--- do the co-occurrence outputs it produced still exist? ---"
ls -la ~/outputs/cooccurrence/ 2>/dev/null || echo "  (no outputs/cooccurrence dir)"
echo
echo ">>> To copy anything missing (will NOT overwrite existing files):"
echo "    cp -n $SCRATCH/*.py \"$REPO/scripts/\""

echo
echo "###############  A5 — pre-commit inspection  ###############"
cd "$REPO" || exit 1
echo
echo "--- current branch / remote ---"
git branch --show-current 2>/dev/null
git remote -v 2>/dev/null | head -2
echo
echo "--- untracked + modified ---"
git status --short
echo
echo "--- anything large about to be committed? (>5MB) ---"
git status --porcelain | awk '{print $2}' | while read -r f; do
    [ -f "$f" ] && sz=$(stat -c%s "$f" 2>/dev/null) && \
      [ "$sz" -gt 5000000 ] && echo "  WARNING ${sz} bytes  $f"
done
echo "  (nothing listed above = clean)"
echo
echo "--- README sanity (profile-README overwrite check) ---"
head -3 README.md 2>/dev/null
echo "  >>> if that is your GitHub PROFILE readme, run: git checkout HEAD -- README.md"
echo
echo "############################################################"
echo " Then, when the above looks right:"
echo
echo "   cd \"$REPO\""
echo "   git add -A"
echo "   git commit -m 'docs: week 2 diagnostic complete"
echo ""
echo "   - 29.68% of real-class pixels discarded to background at tau=0.5"
echo "   - tau sweep: recovering 2/3 of the residual costs 5.54 mIoU"
echo "   - confusion analysis: 3:1 discard-to-confusion ratio, directional pairs"
echo "   - correct ANALYSIS 3.5 presence-gating claim (tile 3487)"
echo "   - recover stranded scripts into tracked tree'"
echo "   git push"
echo "############################################################"