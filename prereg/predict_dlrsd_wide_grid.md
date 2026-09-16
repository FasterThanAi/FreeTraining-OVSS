# Pre-registration — is DLRSD's lever-2 search range binding?

**Committed before the wide-grid run.** Scoring goes in `DLRSD_RESULTS.md`.

## Why

Lever 2 searches each class scale over 11 log-spaced values from **0.40 to 2.50**. On
DLRSD `ship` and `mobile home` finish at **~2.46–2.49**, i.e. at the ceiling, and
`buildings` near the floor (0.38). A class pinned at the edge wants to move further
than the search allows, so the reported **C − B = +5.84 ± 1.03** may be an
underestimate.

## What is run

Same cache (`~/outputs/dlrsd_full/cache`, published vocabulary), same seed, same
stratified folds, same τ, same subsample. **Only the range changes**, at the same step:

| arm | range | points | step ratio |
|---|---|---|---|
| default | 0.40 – 2.50 | 11 | 1.201 |
| wide | **0.10 – 10.0** | **27** | 1.194 |

Both grids contain 1.0. ⭐ The default arm is the control: it must reproduce the
recorded **+5.84 ± 1.03**, since `--w-min/--w-max/--w-steps` default to the original
hardcoded grid (checked bit-identical).

## Predictions

**G1 — the default arm reports classes on the boundary**, with `ship` and `mobile home`
at the ceiling in most folds.

**G2 — the wide arm lets at least one of them go beyond 2.50.**

**G3 ⭐ — C − B changes by less than +0.5 mIoU.** *Reason:* the saturated classes are
small (`ship` 1.46%, `mobile home` 1.82% of pixels). `mobile home` is a visual confusion
that no word and no lever has reached (0.29 at every rung). `ship` could gain a few IoU,
which is a few tenths of mIoU across 17 classes. Point estimate **+0.2**.

**G4 — `mobile home` stays below 5.0 IoU** even with a scale up to 10×.

## Branch table

| outcome | reading |
|---|---|
| default arm does not reproduce +5.84 | ⛔ stop — the grid change is not a no-op |
| G3 holds | ✅ the range was binding but not costly; +5.84 stands |
| C − B rises ≥ +0.5 | ⭐ +5.84 was an underestimate. Report both ranges; the wider result is still 5-fold held out, so it is legitimate, but it must be verified end to end before replacing the recorded number |
| C − B falls | ⚠️ a wider search overfits the calibration folds; keep the default range |
| G4 fails | ⭐ `mobile home` was a scale problem after all, not purely visual — contradicts §13's W2 reading |
