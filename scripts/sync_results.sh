#!/usr/bin/env bash
# Copy generated run reports into the repo so they are version-controlled.
#
# ⛔ WHY THIS EXISTS. ~/outputs is OUTSIDE the repo and `outputs/` is gitignored,
# so every .md the analysis scripts write lives on one machine only. The written
# conclusions are safe -- they are transcribed into the *_RESULTS.md files -- but
# the primary record of each run was not backed up anywhere, and WEEK3_RESULTS
# §13 cites ~/outputs paths a reader cannot open.
#
# Markdown reports are a few KB each. There is no reason not to track them.
#
#   bash scripts/sync_results.sh            # copy, then show what changed
#   bash scripts/sync_results.sh --dry-run  # just list
set -euo pipefail

SRC="${OUTPUTS_DIR:-$HOME/outputs}"
DST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/results"
DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

[[ -d "$SRC" ]] || { echo "no $SRC on this machine -- nothing to sync"; exit 0; }
mkdir -p "$DST"

n=0
while IFS= read -r -d '' f; do
    rel="${f#$SRC/}"
    # ⚠️ size guard: a report is a few KB. Anything large is a data dump that
    # does not belong in git, and silently committing one would be worse than
    # skipping it loudly.
    sz=$(wc -c < "$f")
    if (( sz > 262144 )); then
        echo "  SKIP (${sz}B > 256K): $rel"
        continue
    fi
    if (( DRY )); then
        echo "  would copy: $rel"
    else
        mkdir -p "$DST/$(dirname "$rel")"
        cp -p "$f" "$DST/$rel"
    fi
    n=$((n+1))
done < <(find "$SRC" -name '*.md' -type f -print0)

echo "$n report(s) from $SRC"
(( DRY )) && exit 0

cd "$(dirname "$DST")"
if git diff --quiet --exit-code -- results && \
   [[ -z "$(git ls-files --others --exclude-standard -- results)" ]]; then
    echo "nothing changed."
else
    echo
    git status --short -- results
    echo
    echo "review, then:  git add results && git commit -m 'results: sync run reports'"
fi
