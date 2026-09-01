#!/usr/bin/env bash
# Build a flat, ready-to-upload Overleaf bundle.
#
# WHY FLAT. main.tex uses \graphicspath{{../docs/}} so it builds in place on a
# machine that has the repo. Overleaf has no parent directory, so the figures
# must sit beside main.tex and that line has to go. Doing it by hand is how a
# "File not found: fig2_mechanism.pdf" happens on the day of the deadline.
#
#   bash scripts/make_overleaf_bundle.sh          -> ~/Desktop/overleaf_paper.zip
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$HOME/Desktop/overleaf_paper}"
rm -rf "$OUT" "$OUT.zip"; mkdir -p "$OUT"

cp "$REPO/paper/numbers.tex" "$REPO/paper/refs.bib" "$OUT/"

# Figures are collected from BOTH documents -- the supplementary is a separate
# compile that shares numbers.tex and refs.bib, so it must ship in the same
# project or its \input{numbers} fails.
TEXFILES="$REPO/paper/main.tex"
if [ -f "$REPO/paper/supplementary.tex" ]; then
  sed '/\\graphicspath/d' "$REPO/paper/supplementary.tex" > "$OUT/supplementary.tex"
  TEXFILES="$TEXFILES $REPO/paper/supplementary.tex"
  echo "  + supplementary.tex"
fi

# Only the figures main.tex actually includes -- an unused 33 KB PDF in the
# upload is a small thing, but a stale one that a caption still points at is not.
FIGS=$(grep -ho 'includegraphics\[[^]]*\]{[^}]*}' $TEXFILES \
       | sed 's/.*{//;s/}//' | sort -u)
for f in $FIGS; do
  if [ -f "$REPO/docs/$f" ]; then cp "$REPO/docs/$f" "$OUT/"; echo "  + $f"
  else echo "  !! MISSING: docs/$f — regenerate with scripts/fig_*.py" >&2; exit 1
  fi
done

# strip the graphicspath line; everything else is untouched
sed '/\\graphicspath/d' "$REPO/paper/main.tex" > "$OUT/main.tex"

cd "$(dirname "$OUT")" && zip -qr "$(basename "$OUT").zip" "$(basename "$OUT")"
echo
echo "bundle: $OUT.zip"
ls -1 "$OUT"
echo
echo "Overleaf: New Project -> Upload Project -> that .zip"
echo "Then swap the two lines marked % TEMPLATE for the venue class."
