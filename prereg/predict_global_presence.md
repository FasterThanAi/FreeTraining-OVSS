# Pre-registration — decoupling the presence gate from the crop grid

**Written 13 Sep 2026, BEFORE the run.** `git log` is the timestamp. Not edited afterwards.

⚠️ **I advised against running this and was overruled; the decision is recorded as the user's.**
My stated objections stand and are repeated here so they are scored too: it is a fourth attempt at
the same family after three nulls, my record on "this should work" is **0 for 4**, and it is an
inference-time change that strengthens the *baseline* rather than the contribution.

⭐ **But one objection I raised was wrong, and it is why the run is worth doing.** I called it a
repair. It is not: it is a **decomposition**. Sliding-window's **−3.85** is the *net* of two
opposing effects — more effective resolution, and a presence gate that now vetoes per crop. Nothing
has separated them. This run holds resolution fixed and removes the gate penalty, so whatever it
measures is the resolution effect **alone**. That number does not currently exist.

---

## The change

`slide_inference` already accumulates `fused`, the dual-head score **before** gating, so re-gating
it with a tile-level presence is exact — nothing has to be divided back out.

| `presence_mode` | gate | cost |
|---|---|---|
| **`per_view`** | per crop — the published behaviour | ⛔ **default, must stay bit-identical** |
| **`max`** | max presence over crops: *"present anywhere in the tile"* | free, no extra forward |
| `global` | presence from one whole-image forward | one extra pass per tile |

⛔ **Gate before anything else:** `presence_mode='per_view'` with `slide_crop=0` must still give
**47.38**. If it does not, the patch changed the published path and nothing below is readable.

---

## Predictions

**G1 — `max` recovers most of the loss: mIoU above 46.0**, i.e. more than half of −3.85 returned.
*Why:* the diagnosis attributes the loss to per-crop vetoing, and this removes exactly that.
⚠️ If it recovers *little*, the diagnosis in `SLIDING_WINDOW_RESULTS.md` is wrong and that file
must be corrected — the precision-up/recall-down signature would then have another cause.

**G2 — ⭐ `water` recovers more than half of its −14.28.** It is spatially concentrated, so it is
the class the per-crop veto should hit hardest and the one this should most help. **This is the
sharp prediction**: a named class and a threshold, on the mechanism rather than on the headline.

**G3 — mean recall rises above sliding-window's 57.3, and mean precision falls from 68.1.**
*Why:* the gate is what traded recall for precision; loosening it should trade back.

**G4 — ⚠️ `max` will BEAT single-view's 47.38, but by less than +1.5.**
*Why:* resolution doubles and the penalty is removed. ⚠️ **Stated with low confidence** — I have
predicted a positive four times running and been wrong four times, and the resolution benefit has
never been isolated, so it may simply be small or negative on 0.3 m imagery where objects are
already several pixels across.

**G5 — `global` will land within 0.5 mIoU of `max`.**
*Why:* both ask "is this class in this tile?" and differ only in whether the answer comes from one
whole-image forward or the best of four crops. A large gap would mean the presence head responds to
field of view as much as to content, which would itself be worth reporting.

---

## The branch table, written now

| outcome | reading |
|---|---|
| **G1 ✅, G4 ✅** | ⭐ the resolution benefit is real and the gate was masking it. ⛔ **This would be expensive**: it changes the deployed configuration, so both levers must be re-fitted there and every absolute number re-run, five weeks from the content freeze. **Report the decomposition; do not re-baseline the paper on it.** |
| **G1 ✅, G4 ⛔** | the gate explains the loss and resolution buys nothing here. ⭐ **The cleanest possible outcome**: it confirms the mechanism, closes multi-scale with a decomposition rather than a single net number, and changes no headline. |
| **G1 ⛔** | ⛔ the diagnosis is wrong. `SLIDING_WINDOW_RESULTS.md` §2 must be corrected and the precision/recall signature re-explained. |
| **G5 ⛔** | the presence head is field-of-view dependent — a property of SAM 3 worth a sentence, and a caveat on every `S_pres` number in this project. |

⭐ **No outcome invalidates anything recorded.** The published path is gated bit-identical, and
47.38 / +2.32 are single-view measurements that remain what they are.
