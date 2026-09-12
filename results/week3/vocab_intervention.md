# Catch-all share — intervened on, not stratified

- cache: `/home/priyanshu/outputs/oem_full/cache` | tiles: **384** | τ = **0.1**
- catch-all: **`background`**

Every class is an independent forward pass with its own text prompt and the only cross-class operation in the pipeline is the `argmax`, so **dropping a class from the vocabulary is exactly equivalent to dropping its channel**. All arms therefore read one cached score stack and differ in the vocabulary and in nothing else — no run-to-run variation.

⚠️ **`det` is the column that matters, not `AUC`.** AUC is symmetric — `AUC(−score) = 1 − AUC(score)` — so an AUC of 0.208 is a detector of strength 0.792 with its sign flipped, not an absent signal. The mechanism claims information is *lost*, so it is scored on `det = max(AUC, 1−AUC)`. A ⇄ marks an arm where the signal inverted.

⚠️ **The `C` and `D` arms are the experiment.** Merging also reduces the class count, which moves the base rate on its own. Each control applies the *same* merge to a real class instead of the catch-all, so arity changes identically and only the share differs.

**Two dose families.** `A` merges channels into the catch-all by max, which makes it a *union of well-detected prompts* — unnaturally competent, and the opposite of a real catch-all. ⭐ **`B` is the faithful analogue**: the classes are removed from the vocabulary entirely and their pixels labelled catch-all, which keeps its own single weak prompt. That is LoveDA's actual situation.

| arm | vocab | catch-all share | discard % | base % | AUC `conf` | **det** | AUC `conf2` | **det2** | mIoU |
|---|---|---|---|---|---|---|---|---|---|
| **A0 published** | 9 | **0.84%** | 3.78 | 82.7 | 0.794 | **0.794** | 0.913 | **0.913** | 44.16 |
| **A10 dose** | 7 | **10.19%** | 7.67 | 47.8 | 0.208 ⇄ | **0.792** | 0.375 ⇄ | **0.625** | 44.57 |
| **C10 control** | 7 | **0.84%** | 3.78 | 82.7 | 0.794 | **0.794** | 0.910 | **0.910** | 44.82 |
| **B10 dose (prompt dropped)** | 7 | **10.19%** | 4.17 | 60.6 | 0.507 | **0.507** | 0.769 | **0.769** | 41.05 |
| **D10 control (prompt dropped)** | 7 | **0.84%** | 5.42 | 87.0 | 0.783 | **0.783** | 0.855 | **0.855** | 39.44 |
| **A25 dose** | 5 | **39.57%** | 25.33 | 32.0 | 0.181 ⇄ | **0.819** | 0.548 | **0.548** | 41.54 |
| **C25 control** | 5 | **0.84%** | 3.78 | 82.7 | 0.794 | **0.794** | 0.902 | **0.902** | 42.27 |
| **B25 dose (prompt dropped)** | 5 | **39.57%** | 11.59 | 27.8 | 0.624 | **0.624** | 0.651 | **0.651** | 36.90 |
| **D25 control (prompt dropped)** | 5 | **0.84%** | 24.60 | 96.8 | 0.704 | **0.704** | 0.808 | **0.808** | 29.91 |
| **A40 dose** | 4 | **58.20%** | 43.35 | 26.9 | 0.215 ⇄ | **0.785** | 0.552 | **0.552** | 37.28 |
| **C40 control** | 4 | **0.84%** | 3.78 | 82.7 | 0.794 | **0.794** | 0.896 | **0.896** | 38.10 |
| **B40 dose (prompt dropped)** | 4 | **58.20%** | 19.46 | 23.4 | 0.582 | **0.582** | 0.590 | **0.590** | 29.86 |
| **D40 control (prompt dropped)** | 4 | **0.84%** | 34.27 | 97.6 | 0.710 | **0.710** | 0.855 | **0.855** | 21.77 |

- **A0 published** — the unmodified vocabulary.
- **A10 dose** — `background` absorbs water+road — catch-all share rises.
- **C10 control** — the same water+road merged into `bareland` instead — identical class count, catch-all share unchanged.
- **B10 dose (prompt dropped)** — water+road removed from the VOCABULARY; their pixels are labelled `background`, which keeps its own single weak prompt — the faithful analogue of a real catch-all.
- **D10 control (prompt dropped)** — the same classes removed from the vocabulary, but their pixels labelled `bareland` — identical class count, catch-all share unchanged.
- **A25 dose** — `background` absorbs water+road+cropland+building — catch-all share rises.
- **C25 control** — the same water+road+cropland+building merged into `bareland` instead — identical class count, catch-all share unchanged.
- **B25 dose (prompt dropped)** — water+road+cropland+building removed from the VOCABULARY; their pixels are labelled `background`, which keeps its own single weak prompt — the faithful analogue of a real catch-all.
- **D25 control (prompt dropped)** — the same classes removed from the vocabulary, but their pixels labelled `bareland` — identical class count, catch-all share unchanged.
- **A40 dose** — `background` absorbs water+road+cropland+building+tree — catch-all share rises.
- **C40 control** — the same water+road+cropland+building+tree merged into `bareland` instead — identical class count, catch-all share unchanged.
- **B40 dose (prompt dropped)** — water+road+cropland+building+tree removed from the VOCABULARY; their pixels are labelled `background`, which keeps its own single weak prompt — the faithful analogue of a real catch-all.
- **D40 control (prompt dropped)** — the same classes removed from the vocabulary, but their pixels labelled `bareland` — identical class count, catch-all share unchanged.

## Verdict


### B / D — the faithful analogue ⭐

The merged classes are absent from the **vocabulary** and their pixels are labelled catch-all, which keeps its own single weak prompt. This is what a real catch-all is, and it is the arm the paper should quote.

| signal | A0 | largest dose | control | dose effect | control effect |
|---|---|---|---|---|---|
| `conf` | 0.794 | **0.582** | 0.710 | **+0.213** | +0.085 |
| `conf2` (runner-up) | 0.913 | **0.590** | 0.855 | **+0.323** | +0.058 |

Catch-all share 0.84% → **58.20%** in the dose arm, 0.84% in the control.

✅ **`conf2`: share is causal.** §7's mechanism is stated about the RUNNER-UP: a catch-all gives the model a plausible answer everywhere, so a strong runner-up carries no information. This is the row that tests it. Detectability falls **0.323** with the dose while the control moves +0.058 — same merge, same class count, share untouched.

✅ **`conf`: share is causal.** The top score is what §7a's monotone-in-share table used, so it needs its own answer. Detectability falls **0.213** with the dose while the control moves +0.085 — same merge, same class count, share untouched.

### A / C — merge by channel-max

⚠️ Here the catch-all becomes a *union of well-detected prompts*, so it is unnaturally competent at its own pixels. Retained because it was pre-registered, but it is **not** a faithful model of a catch-all — read `B/D` first.

| signal | A0 | largest dose | control | dose effect | control effect |
|---|---|---|---|---|---|
| `conf` | 0.794 | **0.785** | 0.794 | **+0.009** | +0.000 |
| `conf2` (runner-up) | 0.913 | **0.552** | 0.896 | **+0.360** | +0.017 |

Catch-all share 0.84% → **58.20%** in the dose arm, 0.84% in the control.

⚠️ **`conf` INVERTED** in A10 dose, A25 dose, A40 dose — the signal changed sign rather than disappearing, so the raw AUC understates detectability badly. Scored on `det`.

✅ **`conf2`: share is causal.** §7's mechanism is stated about the RUNNER-UP: a catch-all gives the model a plausible answer everywhere, so a strong runner-up carries no information. This is the row that tests it. Detectability falls **0.360** with the dose while the control moves +0.017 — same merge, same class count, share untouched.

⛔ **`conf`: share is NOT the cause.** The top score is what §7a's monotone-in-share table used, so it needs its own answer. Detectability moves only +0.009 across the whole dose range, so raising the catch-all's share does not destroy this signal.

⚠️ Score against `PREREGISTRATION.md`, committed before the first run. The `B`/`D` family and the `det` column were added AFTER seeing that `conf` inverted; their predictions are pre-registered separately in that file's addendum, and the original predictions stand as written.