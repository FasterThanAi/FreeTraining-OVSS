#!/usr/bin/env bash
# Build a flat, ready-to-upload Overleaf bundle.
#
# WHY FLAT. main.tex uses \graphicspath{{../docs/}} so it builds in place on a
# machine that has the repo. Overleaf has no parent directory, so the figures
# must sit beside main.tex and that line has to go. Doing it by hand is how a
# "File not found: fig2_mechanism.pdf" happens on the day of the deadline.
#
#   bash scripts/make_overleaf_bundle.sh          -> ~/Desktop/overleaf_paper.zip
#   bash scripts/make_overleaf_bundle.sh --slides   -> ~/Desktop/overleaf_slides.zip
#
# The slides share paper/numbers.tex and paper/refs.bib deliberately: no number
# is typed twice, so a slide cannot disagree with the paper. The bundle must
# therefore carry both, whichever document it is for.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SRC=paper; NAME=overleaf_paper
if [ "${1:-}" = "--slides" ]; then SRC=slides; NAME=overleaf_slides; shift; fi
OUT="${1:-$HOME/Desktop/$NAME}"
rm -rf "$OUT" "$OUT.zip"; mkdir -p "$OUT"

# ⚠️ numbers.tex and refs.bib live in paper/ for BOTH documents. The slides
# \input{../paper/numbers}; flattening rewrites that below.
cp "$REPO/paper/numbers.tex" "$REPO/paper/refs.bib" "$OUT/"

# Figures are collected from BOTH documents -- the supplementary is a separate
# compile that shares numbers.tex and refs.bib, so it must ship in the same
# project or its \input{numbers} fails.
TEXFILES="$REPO/$SRC/main.tex"
if [ "$SRC" = paper ] && [ -f "$REPO/paper/supplementary.tex" ]; then
  sed '/\\graphicspath/d' "$REPO/paper/supplementary.tex" > "$OUT/supplementary.tex"
  TEXFILES="$TEXFILES $REPO/paper/supplementary.tex"
  echo "  + supplementary.tex"
fi

# Only the figures main.tex actually includes -- an unused 33 KB PDF in the
# upload is a small thing, but a stale one that a caption still points at is not.
FIGS=$(grep -ho 'includegraphics\[[^]]*\]{[^}]*}' $TEXFILES \
       | sed 's/.*{//;s/}//' | sort -u)
# ⚠️ Not every image lives in docs/. The institute masthead sits beside the
# document that uses it (slides/), so look there before declaring one missing --
# adding it to the title page is what broke this bundler on 10 Sep.
for f in $FIGS; do
  if   [ -f "$REPO/docs/$f" ]; then cp "$REPO/docs/$f" "$OUT/"; echo "  + $f"
  elif [ -f "$REPO/$SRC/$f" ]; then cp "$REPO/$SRC/$f" "$OUT/"; echo "  + $f (from $SRC/)"
  else echo "  !! MISSING: $f — not in docs/ or $SRC/; regenerate with scripts/fig_*.py" >&2; exit 1
  fi
done

# Flatten the two path assumptions that only hold inside the repo:
#   \graphicspath{{../docs/}}   -> figures sit beside main.tex on Overleaf
#   \input{../paper/numbers}    -> numbers.tex was copied to the bundle root
#   \bibliography{../paper/refs} -> likewise
sed -e '/\\graphicspath/d' \
    -e 's|{\.\./paper/numbers}|{numbers}|g' \
    -e 's|{\.\./paper/refs}|{refs}|g' \
    "$REPO/$SRC/main.tex" > "$OUT/main.tex"

if grep -q '\.\./' "$OUT/main.tex"; then
  echo "  !! a ../ path survived flattening — Overleaf has no parent directory:" >&2
  grep -n '\.\./' "$OUT/main.tex" >&2; exit 1
fi

cd "$(dirname "$OUT")" && zip -qr "$(basename "$OUT").zip" "$(basename "$OUT")"
echo
echo "bundle: $OUT.zip"
ls -1 "$OUT"
echo
echo "Overleaf: New Project -> Upload Project -> that .zip"
if [ "$SRC" = slides ]; then
  echo "Overleaf compiler: pdfLaTeX. The reference frame needs BibTeX,"
  echo "which Overleaf runs automatically."
else
  echo "Then swap the two lines marked % TEMPLATE for the venue class."
fi
