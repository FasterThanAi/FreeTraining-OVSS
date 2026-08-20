# Patch for ANALYSIS.md §3.5

Replace the existing §3.5 block in full with the text below.

---

### 3.5 Use the presence head in Step 1 — but do not inherit it uncritically

Your Step 1 thresholds raw prediction confidence. SegEarth-OV3's Figure 3 demonstrates that with
a large vocabulary this produces severe noise. Gate by `S_pres` *before* thresholding, exactly as
they do — that part stands.

**But the earlier claim in this section, that inheriting their fix "costs you nothing", was
wrong, and measurement has now disproved it.**

Because `P_final = P_fused · S_pres`, the presence score is a hard ceiling on every pixel in the
tile for that class. When the presence head is miscalibrated low, it destroys dense evidence
that is otherwise confident. Probed on LoveDA val tile `3487` (`WEEK1_RESULTS.md` §8.1):

| Class | S_pres | `semantic_seg` max logit | → sigmoid | Ceiling on P_final |
|---|---|---|---|---|
| road | **0.0757** | **+10.13** | 1.000 | **0.076** |
| building | 0.1309 | +6.44 | 0.998 | 0.131 |
| forest | 0.0094 | +5.03 | 0.993 | 0.009 |
| agricultural | 0.0200 | +5.44 | 0.996 | 0.020 |
| barren | 0.0481 | +2.77 | 0.941 | 0.045 |
| water | 0.0298 | +2.28 | 0.907 | 0.027 |

The semantic head is effectively certain road is present (+10.13). The presence gate reduces the
tile's best achievable score to 0.076 — below τ=0.1, let alone τ=0.5. Zero instance masks are
returned for all six classes. The tile reports **100% discard**, and 54 other tiles behave the
same way.

**This is a second, distinct failure mode.** The low-confidence residual (§2, §4) is one problem;
presence-head collapse is another, and it accounts specifically for the catastrophic tail
identified in `WEEK1_RESULTS.md` §6.4 — the 55 tiles at 100% discard against 958 tiles below 1%.

Three implications for the method:

1. **It is favourable, not fatal.** Had these tiles failed because `P_sem` was genuinely
   uncertain, there would be no signal left to recover. Instead the dense evidence is intact and
   being suppressed by a single global scalar. Recoverable information is the precondition for
   this entire project.

2. **A local prior is the natural correction for a bad global scalar.** `S_pres` is one number
   per class per image. The co-occurrence prior aggregates evidence from *neighbouring regions*.
   These are different information sources, so the prior can in principle override a presence
   veto that the dense heads already contradict. That is a stronger argument for the method than
   the residual-recovery framing alone, and it is worth stating explicitly in the paper.

3. **But the seeding problem is real, and must be resolved before Week 8.** The method labels
   unidentified regions by conditioning on the labels of *identified* neighbours. On a
   100%-discard tile there are no identified patches at all — no seeds, an empty `M_image`, no
   neighbour labels to condition on. `M_global` alone cannot place a label without an anchor.
   Options, to be decided by measurement:
   - Bootstrap from `P_fused` *before* presence gating on tiles where all classes are suppressed
     (a presence-bypass fallback, triggered by a detectable condition rather than a hand-tuned
     rule).
   - Use per-class relative ordering of `S_pres` rather than its absolute value — on 3487 the
     ranking (road < forest < agricultural < water < barren < building) may still be informative
     even where the magnitudes are useless.
   - Accept the 55 tiles as out of scope and report them as an inherited baseline limitation.
     Honest, but it forfeits the most dramatic gains available.

**Caveat: n = 1 tile.** This is currently a single well-characterised anecdote. Before it appears
in the paper, dump per-class `S_pres` across all 1669 val tiles and compare the distribution on
the catastrophic set against the healthy set. If catastrophic tiles are systematically
presence-suppressed, this is a figure and a finding that SegEarth-OV3's paper does not report.
If not, it is a curiosity about one tile and must be dropped.
