# Dropping the catch-all from the vocabulary — ⛔ killed by its own ceiling, in two minutes

**13 Sep 2026, LoveDA, full 1669 tiles, τ = 0.5.** `scripts/drop_class.py --oracle-only`.
**No GPU, no fit, ~2 minutes of CPU.**

---

## 1. The idea, and why it was worth asking

`background` is prompted as a word, but it is not a visual concept — it is LoveDA's bin for
"none of the above", and SAM 3's median presence score for it is **0.022** against 0.45–0.91 for
every real class (WEEK1 §9.2b). **It cannot be detected, yet it still competes in the argmax.**
WEEK1 §7.7 measured the consequence: 6.0% of background assignments are argmax wins at
`conf ≥ τ` — 19.4M pixels in 24 tiles, **every one of them water**.

So: drop its channel and let τ alone produce background. A pixel would become background because
nothing cleared its threshold, never because an undetectable class out-scored a detectable one.

---

## 2. ⛔ The ceiling says no

| | pixels | share of real-class |
|---|---|---|
| the catch-all wins the argmax over a real class | 1,952,989 | 4.55% |
| …and the runner-up would clear the published τ | 84,276 | **0.20%** |
| …and that runner-up is the **correct** class | 84,276 | **0.20%** |

> ⛔ **At most 0.20% of real-class pixels are recoverable, with perfect luck on every one.**
> All of them are `water`, so that is **+1.09 recall**, **+1.02 IoU** on one class, and a
> **mIoU ceiling of +0.146.**

**Dead.** A fit could only capture some fraction of a seventh of a point.

---

## 3. ⭐⭐ But the screen produced a finding that sharpens the paper

Look at the gap between the first two rows. Background wins the argmax on **1,952,989** real-class
pixels, and only **84,276 — 4.3% — have a runner-up that clears τ.**

> ⭐ **95.7% of the pixels the catch-all "steals" are pixels where nothing was confident.
> Background is not out-competing a good answer; it is winning where there is no good answer.**

This refines §7.7 materially. §7.7 established that mechanism (B) — the argmax loss — is
*unreachable by any threshold*. This adds the reason it is also unreachable by removing the
competitor: **the runner-up is below τ too.** Those pixels arrive at the catch-all whichever route
you close. Same destination, different road.

⭐ **And it independently reproduces §7.7's strangest claim.** Every recoverable pixel is `water`:
`water` is the only class where the catch-all takes a meaningful share (**11.34%** of its pixels,
against ≤ 5.06% for every other class) and the only one with any runner-up clearing τ. Two
separate measurements, a month apart, on different code paths, both landing on water alone.

⚠️ ⭐ **One clean sub-result:** of the 84,276 pixels whose runner-up clears τ, **100% of those
runners-up are the correct class.** Where a confident alternative to the catch-all exists at all,
it is right. That is a statement about SAM 3's calibration worth one sentence.

---

## 4. What this validates about the process

⭐ **This is the screen levers 3, 4 and 5 never got.** Each of those was argued from a mechanism,
built, and run — three GPU sessions to establish three nulls. This idea had an equally good
mechanism story, and a two-minute count killed it before a line of fitting code ran.

**The rule, now demonstrated rather than asserted: compute the peeking-oracle ceiling before
building anything.** If the ceiling is small, the idea is finished and it cost minutes.

⛔ **Not running the measured arm.** The ceiling is +0.146 with perfect luck; a real fit captures a
fraction of that, and it would be inside the fold noise of every number in this project.
