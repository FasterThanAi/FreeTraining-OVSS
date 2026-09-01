#!/usr/bin/env bash
# Resolve ConInfer's chain of UNDECLARED dependencies in one pass.
#
# THE PROBLEM. Their requirements.txt is incomplete, and so is the metadata of
# things it installs. Each missing module only reveals itself when the previous
# one is fixed, so resolving them one at a time costs a round-trip each:
#   ftfy      <- mmseg imports it in mmseg/utils/tokenizer.py, never declares it
#   psutil    <- fast-pytorch-kmeans imports it, never declares it
#   ...       <- and so on, however deep it goes
#
# This walks the chain: attempt the import ConInfer's eval.py performs, read the
# missing module out of the traceback, install it, repeat.
#
# ⚠️ It installs package names read from a traceback, so it maps the known
# module->package mismatches explicitly and refuses anything that is not a plain
# identifier. Every install is echoed and summarised at the end, so the result is
# auditable rather than magic. It is capped, and it stops on the first failure.
set -uo pipefail

CLONE="${1:-$HOME/ConInfer}"
MAX=15

[ "${CONDA_DEFAULT_ENV:-none}" = "coninfer" ] || {
  echo "⛔ activate the coninfer env first: conda activate coninfer" >&2; exit 1; }

# module name -> pip package, where they differ
pkg_for() {
  case "$1" in
    cv2)          echo opencv-python ;;
    PIL)          echo pillow ;;
    sklearn)      echo scikit-learn ;;
    yaml)         echo pyyaml ;;
    skimage)      echo scikit-image ;;
    pycocotools)  echo pycocotools ;;
    *)            echo "$1" ;;
  esac
}

INSTALLED=()
cd "$CLONE" || { echo "⛔ no clone at $CLONE" >&2; exit 1; }

for i in $(seq 1 "$MAX"); do
  ERR="$(python -c 'import segearth_segmentor, ConInfer_segmentor' 2>&1)"
  if [ -z "$(printf '%s' "$ERR" | grep -o "No module named '[^']*'")" ]; then
    if printf '%s' "$ERR" | grep -q 'Traceback'; then
      echo
      echo "⛔ import still fails, but NOT for a missing module. Read it:"
      printf '%s\n' "$ERR" | tail -25
      exit 1
    fi
    echo
    echo "✅ ConInfer's imports resolve after ${#INSTALLED[@]} added packages."
    [ ${#INSTALLED[@]} -gt 0 ] && printf '   added: %s\n' "${INSTALLED[*]}"
    echo
    echo "These were missing from BOTH ConInfer's requirements.txt and the"
    echo "metadata of the packages that import them. Worth stating in the paper's"
    echo "reproducibility note."
    exit 0
  fi
  MOD="$(printf '%s' "$ERR" | grep -o "No module named '[^']*'" | tail -1 | sed "s/.*'\\(.*\\)'/\\1/")"
  MOD="${MOD%%.*}"
  case "$MOD" in
    [A-Za-z_][A-Za-z0-9_-]*) ;;
    *) echo "⛔ refusing to install a suspicious module name: '$MOD'" >&2; exit 1 ;;
  esac
  PKG="$(pkg_for "$MOD")"
  echo "[$i] missing module '$MOD' -> pip install $PKG"
  pip install --quiet "$PKG" || { echo "⛔ pip install $PKG failed" >&2; exit 1; }
  INSTALLED+=("$PKG")
done
echo "⛔ still unresolved after $MAX rounds; something else is wrong." >&2
exit 1
