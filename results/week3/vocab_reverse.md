# Catch-all share — intervened on, not stratified

- cache: `/home/priyanshu/outputs/loveda_full/cache` | tiles: **500** | τ = **0.5**
- catch-all: **`background`**

Every class is an independent forward pass with its own text prompt and the only cross-class operation in the pipeline is the `argmax`, so **dropping a class from the vocabulary is exactly equivalent to dropping its channel**. All arms therefore read one cached score stack and differ in the vocabulary and in nothing else — no run-to-run variation.

⚠️ **`det` is the column that matters, not `AUC`.** AUC is symmetric — `AUC(−score) = 1 − AUC(score)` — so an AUC of 0.208 is a detector of strength 0.792 with its sign flipped, not an absent signal. The mechanism claims information is *lost*, so it is scored on `det = max(AUC, 1−AUC)`. A ⇄ marks an arm where the signal inverted.

⚠️ **The `C` and `D` arms are the experiment.** Merging also reduces the class count, which moves the base rate on its own. Each control applies the *same* merge to a real class instead of the catch-all, so arity changes identically and only the share differs.

**Two dose families.** `A` merges channels into the catch-all by max, which makes it a *union of well-detected prompts* — unnaturally competent, and the opposite of a real catch-all. ⭐ **`B` is the faithful analogue**: the classes are removed from the vocabulary entirely and their pixels labelled catch-all, which keeps its own single weak prompt. That is LoveDA's actual situation.

| arm | vocab | catch-all share | discard % | base % | AUC `conf` | **det** | AUC `conf2` | **det2** | mIoU |
|---|---|---|---|---|---|---|---|---|---|
| **A0 published** | 7 | **35.72%** | 27.81 | 42.5 | 0.592 | **0.592** | 0.561 | **0.561** | 47.43 |
| **A45 dose** | 5 | **45.37%** | 31.83 | 33.4 | 0.482 ⇄ | **0.518** | 0.572 | **0.572** | 50.22 |
| **C45 control** | 5 | **35.72%** | 27.81 | 42.5 | 0.592 | **0.592** | 0.547 | **0.547** | 49.46 |
| **B45 dose (prompt dropped)** | 5 | **45.37%** | 30.40 | 33.1 | 0.640 | **0.640** | 0.627 | **0.627** | 50.05 |
| **D45 control (prompt dropped)** | 5 | **35.72%** | 37.56 | 48.1 | 0.551 | **0.551** | 0.503 | **0.503** | 47.13 |
| **R reverse** | 6 | **35.72%** | 27.71 | 42.4 | 0.540 | **0.540** | 0.515 | **0.515** | — |

- **A0 published** — the unmodified vocabulary.
- **A45 dose** — `background` absorbs barren+road — catch-all share rises.
- **C45 control** — the same barren+road merged into `agricultural` instead — identical class count, catch-all share unchanged.
- **B45 dose (prompt dropped)** — barren+road removed from the VOCABULARY; their pixels are labelled `background`, which keeps its own single weak prompt — the faithful analogue of a real catch-all.
- **D45 control (prompt dropped)** — the same classes removed from the vocabulary, but their pixels labelled `agricultural` — identical class count, catch-all share unchanged.
- **R reverse** — the catch-all is removed from the vocabulary, which now covers the scene; sub-τ pixels have no catch-all prompt to fall into.

## Verdict


### B / D — the faithful analogue ⭐

The merged classes are absent from the **vocabulary** and their pixels are labelled catch-all, which keeps its own single weak prompt. This is what a real catch-all is, and it is the arm the paper should quote.

| signal | A0 | largest dose | control | dose effect | control effect |
|---|---|---|---|---|---|
| `conf` | 0.592 | **0.640** | 0.551 | **-0.048** | +0.041 |
| `conf2` (runner-up) | 0.561 | **0.627** | 0.503 | **-0.066** | +0.058 |

Catch-all share 35.72% → **45.37%** in the dose arm, 35.72% in the control.

⚠️ **`conf2`: inconclusive** — dose -0.066 against control +0.058. §7's mechanism is stated about the RUNNER-UP: a catch-all gives the model a plausible answer everywhere, so a strong runner-up carries no information. This is the row that tests it.

⛔ **`conf`: share is NOT the cause.** The top score is what §7a's monotone-in-share table used, so it needs its own answer. Detectability moves only -0.048 across the whole dose range, so raising the catch-all's share does not destroy this signal.

### A / C — merge by channel-max

⚠️ Here the catch-all becomes a *union of well-detected prompts*, so it is unnaturally competent at its own pixels. Retained because it was pre-registered, but it is **not** a faithful model of a catch-all — read `B/D` first.

| signal | A0 | largest dose | control | dose effect | control effect |
|---|---|---|---|---|---|
| `conf` | 0.592 | **0.518** | 0.592 | **+0.074** | +0.000 |
| `conf2` (runner-up) | 0.561 | **0.572** | 0.547 | **-0.011** | +0.014 |

Catch-all share 35.72% → **45.37%** in the dose arm, 35.72% in the control.

⚠️ **`conf` INVERTED** in A45 dose — the signal changed sign rather than disappearing, so the raw AUC understates detectability badly. Scored on `det`.

⛔ **`conf2`: share is NOT the cause.** §7's mechanism is stated about the RUNNER-UP: a catch-all gives the model a plausible answer everywhere, so a strong runner-up carries no information. This is the row that tests it. Detectability moves only -0.011 across the whole dose range, so raising the catch-all's share does not destroy this signal.

✅ **`conf`: share is causal.** The top score is what §7a's monotone-in-share table used, so it needs its own answer. Detectability falls **0.074** with the dose while the control moves +0.000 — same merge, same class count, share untouched.

### Reverse direction

Removing the catch-all from the vocabulary moves `det` -0.053 (0.592 → 0.540). ⛔ It does not rise, so the mechanism does not survive the reverse intervention. ⚠️ mIoU is not comparable for this arm — it has no catch-all class to predict — so only the detection columns mean anything.

⚠️ Score against `PREREGISTRATION.md`, committed before the first run. The `B`/`D` family and the `det` column were added AFTER seeing that `conf` inverted; their predictions are pre-registered separately in that file's addendum, and the original predictions stand as written.