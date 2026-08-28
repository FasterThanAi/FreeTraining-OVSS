# Pre-registered prediction — iSAID

- masks: **458**  |  shapes: [(511, 511), (533, 1251), (546, 475)]
- labelled pixels: **3,221,995,162**
- catch-all class: **`background`** (index 0)
- **`background` share of GT: 97.11%**

## Class composition

| class | pixels | share |
|---|---|---|
| background ⬅ catch-all | 3,128,931,326 | 97.11% |
| harbor | 12,719,317 | 0.39% |
| soccer ball field | 12,372,567 | 0.38% |
| ship | 12,173,919 | 0.38% |
| small vehicle | 11,998,092 | 0.37% |
| plane | 11,181,830 | 0.35% |
| large vehicle | 8,314,984 | 0.26% |
| tennis court | 7,582,709 | 0.24% |
| ground track field | 6,544,978 | 0.20% |
| store tank | 3,037,189 | 0.09% |
| baseball diamond | 2,559,698 | 0.08% |
| basketball court | 1,695,942 | 0.05% |
| bridge | 1,253,429 | 0.04% |
| swimming pool | 789,176 | 0.02% |
| roundabout | 670,520 | 0.02% |
| helicopter | 169,486 | 0.01% |

## The anchors

| dataset | bg share | discard @ τ=0.1 | best detection AUC |
|---|---|---|---|
| LoveDA | 36.10% | 10.88% | 0.622 |
| OpenEarthMap | 0.84% | 3.78% | 0.913 |
| **iSAID** | **97.11%** | **?** | **?** |

## The prediction — written before any inference is run

`iSAID` sits at **97.11%**, the **catch-all regime**, like LoveDA. Expect a large residual and detection near the ~0.53 floor.

| quantity | point interpolation | honest bracket |
|---|---|---|
| real-class pixels discarded @ τ=0.1 | ~10.9% | 3.8–10.9% |
| best detection AUC | ~0.622 | 0.622–0.913 |

⚠️ **97.11% is outside the anchor range (0.84–36.10%)**, so those figures are extrapolation. Predict the ORDERING only: it should be more extreme than the nearer anchor.

**Two points define a line, so no functional form is claimed.** What is falsifiable is the ORDERING against each anchor — more catch-all means more discard and worse detection:

- vs **LoveDA** (36.10% background): discard should be **HIGHER than** 10.88%, and detection AUC **LOWER than** 0.622
- vs **OpenEarthMap** (0.84% background): discard should be **HIGHER than** 3.78%, and detection AUC **LOWER than** 0.913
- detection AUC should be **near the 0.53 floor**
- `S_pres(background)` should stay far below the real classes either way — that part is a property of SAM 3, not of the dataset, and should NOT move

> ⛔ **If a dataset near 5% background behaves like LoveDA, or one near 30% behaves like OpenEarthMap, the mechanism is wrong** and the paper's central claim needs rewriting. That is the point of running this before the pipeline rather than after.

## ⚠️ Two explanations, confounded at n=2 — which does this dataset test?

Everything so far has been attributed to the catch-all's **share**. But across the two anchors, share moves together with a second property, and the second is the more plausible cause:

| | share | **confusability** — does the catch-all LOOK like the real classes? |
|---|---|---|
| LoveDA | 36.1% | **high** — unlabelled roads, pavement and built structures that resemble `road`, `barren`, `building` |
| OpenEarthMap | 0.84% | **low** — rare genuinely-unlabelable leftovers |

`conf2` may fail on LoveDA not because background is *common*, but because background genuinely *looks like* the real classes, so a strong runner-up cannot distinguish a suppressed real class from a background region that merely resembles one. Two datasets cannot separate these.

**Decide which way this dataset breaks the tie, and write it here before running inference:**

- catch-all **visually similar** to the real classes → both hypotheses predict poor detection, and the dataset is uninformative on this question
- catch-all **visually distinct** from them → the hypotheses DISAGREE. High share predicts failure; low confusability predicts success. Either outcome eliminates one explanation.

> A dataset that can only confirm is worth less than one that can discriminate. A result this shape can be told either way afterwards, so it has to be called first.

## Next

1. Commit this file **before** running any inference.
2. `measure_discard_rate.py` with this dataset's config and its own τ.
3. `recoverability_signal.py` for the detection AUC.
4. Compare against the bracket above and record whether the prediction held.
---

## The call, made before inference

**iSAID's catch-all is visually DISTINCT from its real classes.** `background` is everything that
is not a ship, plane, tennis court or storage tank — it does not resemble them the way LoveDA's
unlabelled pavement resembles `road`. **So the two hypotheses disagree here, and this dataset
discriminates:**

| hypothesis | predicts detection AUC |
|---|---|
| **share** — a large catch-all destroys the signal | **fails**, near the 0.53 floor |
| **confusability** — a catch-all that looks like the real classes destroys it | **works**, well above the floor |

Either outcome eliminates one explanation. Note this contradicts the mechanical bracket above,
which is derived from share alone; **confusability is the hypothesis I expect to survive**, so I
expect detection to WORK on iSAID despite its 97.11% background.

⚠️ **Scope caveat, also stated in advance.** iSAID is object-centric, not land-cover: 15 small
object classes on a vast backdrop, where 15 of 16 classes occupy under 0.4% each. If the mechanism
breaks here, the honest reading may be that it holds *within land-cover segmentation* — a scope
limit, not a refutation. Deciding that now prevents choosing the convenient reading later.
