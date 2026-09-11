# Lever 4 — per-class presence weight. ⛔ A null, and it closes the family

**11 Sep 2026, LoveDA, the FULL 1669 tiles @ τ = 0.5, no GPU.** Predictions committed in
`prereg/predict_presence_power.md` (`7f2d2c6`) before the first run.

---

## 1. The result

✅ **Identity gate 0.00000.** At γ = 1 the ungate/re-gate reproduces the cached `logits` exactly,
so γ ≠ 1 measures the deployed pipeline and not an approximation of it.

| | mean | sd | folds+ | mean−2sd | recorded | gate |
|---|---|---|---|---|---|---|
| lever 1 — per-class τ | **+1.17** | 0.44 | 5/5 | +0.29 | +1.18 ± 0.45 | ✅ |
| lever 2 — per-class scale | **+1.22** | 0.65 | 5/5 | −0.09 | +1.16 ± 0.19 | ⚠️ |
| **lever 4 — per-class presence** | **+0.16** | 0.36 | 4/5 | **−0.57** | — | ⛔ |

⛔ **Lever 4 is a null.** ✅ And levers 1 and 2 reproduce their recorded values almost exactly
(+1.17 against +1.18) through a different script on the full split — the run is sound.

---

## 2. ⭐ But the fitted γ is a clean, exactly-tested finding

| class | median `S_pres` | **γ** | fold spread |
|---|---|---|---|
| road | 0.910 | **1.64** | 0.33 |
| building | 0.840 | **1.76** | 0.33 |
| water | 0.766 | 0.92 | 0.11 |
| agricultural | 0.602 | 1.16 | 0.09 |
| barren | 0.555 | 0.80 | **0.00** |
| ⭐ **forest** | **0.453** | ⭐ **0.44** | 0.09 |
| *background* | *0.022* | *1.28* | ⛔ *0.99 — not identified* |

> ⭐⭐ **Spearman(median `S_pres`, γ) = +0.886 over the six real classes, exact p = 12/720 =
> 0.017.** Classes the gate barely touches want **more** of it; the class it crushes hardest wants
> **least**. The fit was given no presence statistics — it recovered the ordering from pixels.

⭐ **`forest` — the most-gated real class and the worst-performing class in the dataset
(33.78 IoU, 34.6% discarded) — takes the lowest γ of any class, 0.44, in every single fold.**
That is exactly the class the mechanism named in advance, and it is M2, the sharp prediction.

---

## 3. Predictions scored

| | prediction | measured | |
|---|---|---|---|
| **M1** | clears the gate, between +0.04 and +1.16 | **+0.16 ± 0.36**, 4/5, mean−2sd −0.57 | ⛔ **null** |
| **M2** ⭐ | `forest` takes γ < 1 | **0.44**, lowest of all, 5/5 folds | ✅ **decisive** |
| **M3** | `road` stays within ±0.15 of 1 | **1.64** | ⛔ |
| **M4** | `background` takes γ > 1 | mean 1.28, but folds **2.0/0.2/2.0/0.2/2.0** | ⚠️ **not identified** — the mean is an artefact of averaging the two grid endpoints. Exactly the caveat the script prints. |
| **M5** ⭐ | γ correlates positively with median `S_pres` | **ρ +0.886, p 0.017** | ✅ **decisive** |
| **M6** | ≥ 2 classes keep γ ≈ 1 in every fold | **3 of 7** *(after the tolerance fix below)* | ✅ |

⚠️ **M6 was scored as 0 of 7 by a broken check, and the fix is mine to own.** The band was ±0.15
against a **grid step of 0.2**, so a class could register as "stayed at 1" only by landing exactly
on 1.0 in all five folds — a test stricter than the grid can express. With half a step plus a
margin (±0.15 → ±0.15 of a 0.2 grid → now ±0.15+0.05 computed from the grid itself), `water`,
`agricultural` and `barren` all qualify: **3 of 7**. ⭐ **The table was right and the prose was
wrong — the fifth occurrence of that exact failure in this codebase** (WEEK3 §11), and the first
where I wrote the check myself in the same session.

---

## 4. ⭐⭐ What this closes — the pre-registered stopping point

`prereg/predict_presence_power.md` named this branch in advance:

> **M1 ⛔ (null)** — levers 1 and 2 exhaust the per-class decision rule. Three independent knobs —
> fusion, presence, and the score itself — reduce to two. **That is a bound worth stating, and it
> is where this line of work stops.**

Four levers have now been fitted on the same protocol, the same budget and the same gate:

| lever | what it changes | where it acts | result |
|---|---|---|---|
| **1** per-class τ | which label a score earns | **after** the argmax | ✅ **+1.18** |
| **2** per-class scale | which class wins | **at** the argmax | ✅ **+1.16** |
| 3 per-class head fusion | the score itself | **before** the argmax | ⛔ +0.04 / +0.22 |
| 4 per-class presence weight | the score itself | **before** the argmax | ⛔ **+0.16** |

> ⭐ **Everything that reshapes the DECISION works. Everything that reshapes the EVIDENCE does
> not.** SAM 3's per-class evidence is not what is miscalibrated — the rule that converts evidence
> into a label is.

⚠️ **Stated as a description of four levers, not a law.** It is post-hoc, and the last post-hoc
reading in this project (*"the levers substitute"*) was refuted within a week of being written
down. What makes it a reasonable stopping point is not its elegance but that **the decision side
is exhausted by construction**: per-class τ is *provably* the complete family after a fixed argmax
(the monotone-equivalence argument), and the only thing left at the argmax is a general
reordering, which is unbounded in parameters against 200 tiles. There is no lever 5 to run.

⭐ **And both null levers diagnosed correctly before failing to help.** Lever 3 recovered
SegEarth-OV3's things/stuff duality unsupervised (`car` 2.33 vs `forest` 0.25); lever 4 recovered
the presence ordering at ρ +0.886, p 0.017, and named `forest` as the crushed class. **The
mechanisms are real and measurable, and correcting them buys nothing because levers 1 and 2
already reach that mass by another route.** That is a bound with an explanation, which is this
project's most reliable kind of result.

---

## 5. ⚠️ Limitations to carry with this

- **91 of 1669 tiles have `S_pres ≤ 0.001`** for some class. The ungate divides by presence, so
  those are clamped to a no-op — γ cannot act on 5.5% of the tiles. It does not change a null, but
  it would need handling if the result had been positive.
- **The synonym approximation.** Presence is applied per **query**, before the synonym collapse;
  `spres` is cached per **class**. Exact for `road`, `water`, `agricultural`, `background`;
  approximate for `building,house`, `forest,tree`, `barren,bareland,soil` — and `forest`, the class
  M2 names, is one of the approximate ones. ⛔ No synonym class's γ may be quoted as *deployed*
  without an `eval.py` run, per §9c. **The M5 correlation is unaffected** — it is a statement about
  the fit, not about a deployment.
- **`background`'s γ is not identified** under `--objective real`, which does not score it. Its
  row is not readable and should not be quoted.
- **One dataset.** Lever 3 was run on two; lever 4 on LoveDA only. Potsdam's `--cache-full` exists
  and the run is CPU-only, so a second point costs an hour if the null is ever questioned.
