#!/usr/bin/env bash
# Build both papers and FAIL LOUDLY on anything a draft would rather hide.
#
# ⛔ Why this exists. An ad-hoc `pdflatex -interaction=nonstopmode` reports a page
# count even when the document contains errors, because nonstopmode's whole job is
# to keep going. A build of main_cvpr.tex reported "7 pages, 0 undefined" while
# containing two undefined control sequences (\etal), because the check grepped for
# undefined REFERENCES and not for undefined COMMANDS. A page count from a broken
# build is worse than no page count.
#
#   bash scripts/build_papers.sh            # both
#   bash scripts/build_papers.sh main_cvpr  # one
#
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

command -v pdflatex >/dev/null || {
  echo "⛔ pdflatex not found."
  echo "   sudo apt-get install -y texlive-latex-recommended texlive-latex-extra \\"
  echo "                           texlive-fonts-recommended texlive-bibtex-extra"
  echo "   ...or use scripts/make_overleaf_bundle.sh and compile on Overleaf."
  exit 1
}

echo "── source check ──"
python scripts/check_paper_consistency.py | tail -9 || exit 1

cd paper || exit 1          # \graphicspath and \input are relative to here
DOCS=${*:-"main_cvpr main"}
fail=0

for d in $DOCS; do
  echo
  echo "════════════════ $d ════════════════"
  pdflatex -interaction=nonstopmode "$d.tex" >/dev/null 2>&1
  bibtex   "$d"                              >/dev/null 2>&1
  pdflatex -interaction=nonstopmode "$d.tex" >/dev/null 2>&1
  pdflatex -interaction=nonstopmode "$d.tex" >/dev/null 2>&1

  log="$d.log"
  [ -f "$log" ] || { echo "  ⛔ no log produced"; fail=1; continue; }

  # --- hard errors: these make a page count meaningless --------------------
  errs=$(grep -ac '^! ' "$log")
  undef_cmd=$(grep -ac 'Undefined control sequence' "$log")
  missing=$(grep -a 'LaTeX Warning: File.*not found\|^! LaTeX Error: File' "$log")

  if [ "$errs" -gt 0 ] || [ "$undef_cmd" -gt 0 ]; then
    echo "  ⛔ $errs LaTeX error(s), $undef_cmd undefined command(s):"
    grep -a -A3 '^! ' "$log" | head -20
    fail=1
  fi
  [ -n "$missing" ] && { echo "  ⛔ missing file:"; echo "$missing" | head -4; fail=1; }

  # --- soft: worth seeing, not worth failing on ---------------------------
  undef_ref=$(grep -ac 'Citation.*undefined\|Reference.*undefined' "$log")
  over=$(grep -ac 'Overfull' "$log")
  pages=$(grep -a 'Output written' "$log" | sed 's/.*(\([0-9]*\) pages.*/\1/')

  if [ "$errs" -eq 0 ] && [ "$undef_cmd" -eq 0 ]; then
    echo "  ✅ no errors, no undefined commands"
  fi
  echo "  undefined refs/citations : $undef_ref"
  echo "  overfull boxes           : $over"
  echo "  PAGES                    : ${pages:-?}"

  if [ "$d" = "main_cvpr" ] && [ -n "${pages:-}" ] && [ "$pages" -gt 8 ]; then
    echo "  ⚠️  over the 8-page conference limit"
  fi
done

echo
[ "$fail" -eq 0 ] && echo "ALL CLEAN" || echo "⛔ BUILD HAS ERRORS — the page count above is not trustworthy"
exit $fail
