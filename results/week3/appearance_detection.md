# Week 3 — appearance-based detection

- tiles: **1669**  |  τ: **0.5**  |  atoms: **`slic`**
- atoms scored: **506,064**, covering **748,277,942** px
- recoverable (majority a real class): **42.3%** by pixels — the base rate any rule must beat
- tiles where per-image prototypes existed: atoms **82.6%**

## Signals

Prototypes are built from pixels SAM 3 is confident about — its own output, never ground truth. GT is used only to score the AUC.

`raw` is the plain weighted AUC. **`size-controlled`** recomputes it inside atom-size bins, because a negative control on random-colour images scored **0.966** raw — an atom's mean colour has noise scaling as 1/√size, so any distance feature partly measures atom size, and size correlates with the label. Read the second column. The `atom size` row is printed so the confound is visible rather than hidden.

| signal | raw AUC | size-controlled AUC |
|---|---|---|
| novelty vs GLOBAL prototypes | 0.510 | **0.514** |
| novelty vs PER-IMAGE prototypes | 0.537 | **0.528** |
| mean R | 0.580 | **0.586** |
| mean G | 0.579 | **0.580** |
| mean B | 0.586 | **0.585** |
| gradient energy (texture) | 0.612 | **0.622** |
| atom size (confound reference) | 0.533 | **0.555** |

AUC 0.50 is a coin flip. For reference, the best score-based signal across eight tests was `conf` at **0.582**.

## Verdict

⚠️ **Marginal — `gradient energy (texture)`, AUC 0.622**, about level with `conf` (0.582) and well short of what the mIoU sweep needs. Crude appearance is not enough on its own. Deep features could still clear the bar, but this is no longer evidence that they will; it is a coin flip on a week of work. Decide against the calendar, not the hope.