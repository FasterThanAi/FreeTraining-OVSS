# Generated run reports

⚠️ **These are copies, not sources.** Each file is written by a script into
`~/outputs/...` on the machine that ran it, and `scripts/sync_results.sh` copies
it here so it is version-controlled. `~/outputs` is outside the repo and
gitignored, so without this the primary record of every run lived on one disk.

**To refresh after a run:**

```bash
bash scripts/sync_results.sh
git add results && git commit -m "results: sync run reports"
```

⛔ **Do not edit anything in here.** Re-run the script that produced it instead —
a hand-edited report that no longer matches its own command is worse than no
report. The interpretation belongs in the `*_RESULTS.md` files at the repo root,
which cite these.

⚠️ Files over 256 KB are skipped and named on stderr; a report that large is a
data dump and belongs in `~/outputs` only.
