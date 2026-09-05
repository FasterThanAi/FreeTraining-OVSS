# Pre-registration — argmax scaling on ConInfer (CLIP backbone)

**Written 6 Sep 2026, BEFORE the run.** `git log` is the timestamp. Not edited afterwards.

The only test of whether argmax scaling is a property of *thresholded per-class scores in
general* or of SAM 3's particular calibration. LoveDA and Potsdam are both SAM 3.

---

## What is known going in

| | SAM 3 / LoveDA | SAM 3 / Potsdam | **ConInfer / LoveDA** |
|---|---|---|---|
| published τ | 0.5 | 0.1 | **0.8** |
| per-class τ | +1.16 | +0.59 | **+2.51 ± 0.34** |
| calibration tiles for τ | ~200 | ~100 | **~25** |
| **scale (C − B)** | **+1.16** | **+4.86** | **← this run** |
| catch-all assignments that are argmax losses | 11.31% | 6.10% | **8.56%** |
| residual that is *self*-reachable | 38.52% | 52.64% | **42.15%** |

Potsdam's +4.86 came almost entirely from `tree`: precision 93.34 / recall 38.63, a **+54.7**
gap that thresholds could not touch because only **19.2%** of its residual was
self-reachable. **The scale pays where a class is under-firing AND blocked at the argmax.**

---

## Predictions

**C1 — the increment will be positive, above +0.50.**
*Why:* 8.56% of catch-all assignments are argmax losses, which no threshold reaches, and
every dataset large enough to measure has shown a positive increment.
⛔ *Falsified if* ≤ +0.50.

**C2 — but SMALLER than Potsdam's +4.86.**
*Why:* ConInfer runs at τ = 0.8, a very high operating point, so most of its residual is
threshold-reachable — which is why per-class τ is already worth +2.51 there. Its argmax-lost
share (8.56%) is below SAM 3 / LoveDA's (11.31%). Less argmax-blocked mass to recover.
⛔ *Falsified if* ≥ +4.86.

**C3 — no single class will carry more than 60% of the real-class gain.**
*Why:* per-class τ improved **all seven** ConInfer classes, which our SAM 3 result does not
manage. That suggests a uniformly skewed calibration rather than one pathological class.
Potsdam's `tree` was 72%; LoveDA's `water` was 54%.
⛔ *Falsified if* one class is ≥ 60%.

**C4 — the fitted scales will span at least 2× (max ÷ min).**
*Why:* the structural precondition. If CLIP's per-class scores are already balanced against
each other, the fit lands near w = 1 and C1 must fail with it. Stated separately so a null
is attributable rather than blank.
⛔ *Falsified if* < 2.

**C5 — calibration stays cheap: a positive worst draw at 50 tiles.**
*Why:* per-class τ needed ~25 tiles on this backbone against ~200 on SAM 3, and Potsdam's
scale was positive from 100 tiles with a worst draw of +4.42.
⛔ *Falsified if* the worst draw at 50 tiles is negative.

---

## Readings decided in advance

| outcome | what it means |
|---|---|
| C1 and C4 hold | ⭐ the scale is **not backbone-specific** — three datasets, two architectures, three operating points. The strongest form of the claim. |
| C4 holds, C1 fails | the scales move but buy nothing: CLIP's argmax is comparatively clean, and τ = 0.8 already collects what is available. A real finding about the operating point. |
| C4 fails | **an explained null** — CLIP's per-class scores are already balanced, so there is nothing to reorder. |
| C2 fails *(≥ Potsdam)* | the reasoning above is wrong: argmax-lost share does not govern the size of the gain. |
| C3 fails | the gain is one class again, and the claim narrows to "helps a specific pathology" rather than "helps generally". |

⚠️ **The gate is unchanged:** `mean − 2·sd > 0` **and** every fold positive. 1669 tiles gives
~334 per fold, the same as LoveDA, so the fold noise that made OpenEarthMap unmeasurable does
not apply.

⚠️ **Two things that must pass before any of this is read:** the instrumented cache must
print **36.99** mIoU, and rung A must land near **37**.

⛔ Recorded as it comes. No statistic invented afterwards to explain five numbers.
