"""
Does per-class SCALING before the argmax buy anything over per-class thresholds?

THE GAP THIS TARGETS, in the paper's own words. Our method fits one threshold per
class, and Sec.~method now proves that rule is COMPLETE -- but complete *given the
argmax*. Every per-class monotone recalibration composed with a global threshold
collapses to a per-class threshold vector, so nothing that acts after the argmax
can do better. What acts BEFORE it is a strictly larger family, and we have never
tested it.

The mass sitting there is not small:

    error type                          LoveDA    reachable by per-class tau?
    below threshold                      94.0%    yes
    catch-all won the argmax              6.0%    NO
    confused with another real class      9.9%    NO   <- never attacked
                                         of all real-class pixels

`forest -> agricultural` alone is 23.8M pixels and `water -> agricultural` 19.3M
(WEEK1 §8.1). No threshold vector can touch either: lowering a threshold cannot
change which class won.

THE RULE. Fit a per-class scale w and a per-class threshold tau together:

    pred = argmax_c ( w_c * s_c )        <- w acts here, and ONLY here
    keep pred if s_pred >= tau_pred, else catch-all

⭐ `conf` deliberately stays the RAW score s, not the scaled one. That is not an
arbitrary choice, it is what makes the experiment interpretable: a scale applied
after the argmax is monotone, so it folds into tau and buys nothing (that is
exactly the completeness argument). Thresholding the raw score therefore isolates
w to the reorderings, and the comparison below measures the reorderings alone.

⚠️ HONESTY ABOUT NOVELTY. Per-class logit scaling is not a new idea -- it is prior
correction / logit adjustment from the long-tail literature. What is untested is
whether it helps a TRAINING-FREE open-vocabulary pipeline at the same supervision
budget, and whether it adds anything ON TOP OF per-class thresholds. Report it as
that, exactly as we report per-class thresholds as a known technique whose
completeness for this decision is the contribution.

THREE RUNGS, so the question is answerable:

    A  published tau, w = 1            the baseline
    B  fitted tau,    w = 1            OUR CURRENT METHOD
    C  fitted tau,    fitted w         this experiment

⭐ **The result is C minus B, not C minus A.** C beating A proves nothing -- B
already does that. Anyone reading only the C-vs-A number would credit the
thresholds to the scaling.

WHY IT NEEDS THE FULL CACHE. The (gt, pred, conf-bin) histogram is a sufficient
statistic only while `pred` is fixed. Here w changes the argmax, so the histogram
must be rebuilt per candidate w and the per-class score stack is required:

    python scripts/measure_discard_rate.py --tau 0.5 --cache-full \\
        --sample 500 --seed 0 --out ~/outputs/loveda_full

⚠️ ~24 GB for all 1669 LoveDA tiles, so use --sample. §7c ran the vocabulary
intervention on a 500-tile draw for the same reason.

⚠️ PIXEL SUBSAMPLING, AND ITS GATE. Searching over w means rebuilding histograms
hundreds of times, which is not affordable over every pixel. We subsample pixels
per tile for the search. That is only legitimate if the subsample reproduces the
full-pixel answer, so rung B is computed BOTH ways -- exactly, from per-tile
histograms over all pixels, and from the subsample -- and the run refuses to
report C unless they agree. Same discipline as verify_perclass_tau.py.

    python scripts/argmax_reorder.py --cache ~/outputs/loveda_full/cache \\
        --tau 0.5 --md ~/outputs/week4/argmax_reorder.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels                                                     # noqa: E402
from tau_oracle import confusion_at, per_class_iou, miou, NBINS   # noqa: E402
from tau_cv import fit as fit_tau, obj_miou                       # noqa: E402


# --------------------------------------------------------------------------- #
def load_full(files, nsub, nc, nbins, rng):
    """One pass over the cache. Returns a pixel subsample for the w search, and
    per-tile EXACT histograms at w=1 for the gate.

    The exact histograms cost nothing extra here -- we already have the stack in
    memory for each tile -- and they are what makes the subsample checkable.
    """
    T = len(files)
    S = np.zeros((T, nsub, nc), np.float32)
    G = np.zeros((T, nsub), np.int32)
    PT = np.zeros((T, nc, nc, nbins), np.int32)
    short = 0

    for i, f in enumerate(files):
        z = np.load(f)
        if 'logits' not in z.files:
            raise SystemExit(
                f'{f.name} has no `logits` key. This experiment needs the full '
                f'per-class score stack, which only --cache-full writes:\n\n'
                f'    python scripts/measure_discard_rate.py --cache-full ...\n')
        L = z['logits'].astype(np.float32)          # (N, H, W)
        gt = z['gt'].astype(np.int32)               # mask values, 0 = no-data
        m = gt > 0
        n_valid = int(m.sum())
        if n_valid == 0:
            continue

        flat = L[:, m].T                            # (n_valid, N)
        g = gt[m]

        # ---- exact per-tile histogram at w = 1 (the current method's statistic)
        pred = np.argmax(flat, axis=1)
        conf = flat[np.arange(len(pred)), pred]
        b = np.clip((conf * nbins).astype(np.int32), 0, nbins - 1)
        idx = (g - 1) * nc * nbins + pred * nbins + b
        PT[i] = np.bincount(idx, minlength=nc * nc * nbins).reshape(nc, nc, nbins)

        # ---- pixel subsample for the w search
        if n_valid >= nsub:
            sel = rng.choice(n_valid, nsub, replace=False)
        else:
            sel = rng.choice(n_valid, nsub, replace=True)   # tiny tiles only
            short += 1
        S[i], G[i] = flat[sel], g[sel]

        if (i + 1) % 50 == 0 or i + 1 == T:
            print(f'  {i + 1}/{T}')
    if short:
        print(f'  note: {short} tiles had fewer than {nsub} labelled pixels and '
              f'were sampled with replacement')
    return S, G, PT


def hist_at(S, G, w, nc, nbins):
    """(gt, pred, conf-bin) histogram under a per-class scale w.

    `pred` uses the SCALED scores; `conf` uses the RAW score, so w is confined to
    the argmax. See the module docstring -- a post-argmax scale is monotone and
    would just reparameterise tau.
    """
    sc = S * w                                   # (n, N)
    pred = np.argmax(sc, axis=1)
    conf = S[np.arange(len(pred)), pred]
    b = np.clip((conf * nbins).astype(np.int32), 0, nbins - 1)
    idx = (G - 1) * nc * nbins + pred * nbins + b
    return np.bincount(idx, minlength=nc * nc * nbins
                       ).reshape(nc, nc, nbins).astype(np.int64)


def score_w(S, G, w, bg, nc, nbins, objective, rounds):
    """Best achievable objective at this w: refit tau underneath it."""
    H = hist_at(S, G, w, nc, nbins)
    taus = fit_tau(H, bg, nbins, rounds=rounds, objective=objective)
    return obj_miou(confusion_at(H, taus, bg, nbins), bg, objective), taus


def fit_scale(S, G, bg, nc, nbins, grid, objective, rounds, tau_rounds, verbose=True):
    """Coordinate ascent on w, with tau refitted inside every evaluation.

    Renormalised to geometric mean 1 after each pass: only the RATIOS of w matter
    to an argmax, so without this the vector drifts and becomes unreadable.
    """
    w = np.ones(nc)
    best, taus = score_w(S, G, w, bg, nc, nbins, objective, tau_rounds)
    if verbose:
        print(f'    w = 1 baseline objective {best:.4f}')
    for r in range(rounds):
        moved = False
        for c in range(nc):
            cur = w[c]
            for cand in grid:
                if cand == cur:
                    continue
                trial = w.copy()
                trial[c] = cand
                sc, tt = score_w(S, G, trial, bg, nc, nbins, objective, tau_rounds)
                if sc > best + 1e-9:
                    best, w, taus, moved = sc, trial, tt, True
        w = w / float(np.exp(np.mean(np.log(w))))       # geometric mean -> 1
        best, taus = score_w(S, G, w, bg, nc, nbins, objective, tau_rounds)
        if verbose:
            print(f'    round {r + 1}: objective {best:.4f}  '
                  f'w = [{", ".join(f"{x:.2f}" for x in w)}]')
        if not moved:
            break
    return w, taus


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True, help='a --cache-full cache (needs `logits`)')
    ap.add_argument('--tau', type=float, required=True, help="the published threshold")
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--subsample', type=int, default=40000,
                    help='labelled pixels kept per tile for the w search')
    ap.add_argument('--objective', choices=['all', 'real'], default='real')
    ap.add_argument('--w-rounds', type=int, default=3)
    ap.add_argument('--tau-rounds', type=int, default=3,
                    help='tau coordinate-ascent passes INSIDE the w search; the '
                         'reported fits use the full 6')
    ap.add_argument('--gate', type=float, default=0.15,
                    help='max |mIoU| disagreement between the subsample and the '
                         'exact histograms before the run refuses to report C')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--md', default=None)
    args = ap.parse_args()

    LB = labels.from_cache(args.cache)
    nc, bg = LB.n, LB.bg - 1
    print(f'  classes: {LB}')

    files = sorted(Path(args.cache).expanduser().glob('*.npz'))
    if args.limit:
        files = files[:args.limit]
    T = len(files)
    if not T:
        raise SystemExit(f'no .npz under {args.cache}')
    print(f'{T} tiles | published τ = {args.tau} | subsample {args.subsample} px/tile\n')

    rng = np.random.default_rng(args.seed)
    S, G, PT = load_full(files, args.subsample, nc, NBINS, rng)
    grid = np.round(np.exp(np.linspace(np.log(0.40), np.log(2.50), 11)), 3)
    print(f'\n  w grid: {list(grid)}\n')

    order = rng.permutation(T)
    folds = np.array_split(order, args.folds)
    rows, gate_rows = [], []

    for k in range(args.folds):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(args.folds) if j != k])
        print(f'fold {k + 1}:')

        # ---------- exact, all pixels: rungs A and B (the current method)
        Htr = PT[tr].sum(0).astype(np.int64)
        Hte = PT[te].sum(0).astype(np.int64)
        tau_b_exact = fit_tau(Htr, bg, NBINS, objective=args.objective)
        A_exact = miou(confusion_at(Hte, np.full(nc, args.tau), bg, NBINS))
        B_exact = miou(confusion_at(Hte, tau_b_exact, bg, NBINS))

        # ---------- subsample: rung B again (the gate), then rung C
        Str, Gtr = S[tr].reshape(-1, nc), G[tr].reshape(-1)
        Ste, Gte = S[te].reshape(-1, nc), G[te].reshape(-1)
        ones = np.ones(nc)
        Hs_tr = hist_at(Str, Gtr, ones, nc, NBINS)
        Hs_te = hist_at(Ste, Gte, ones, nc, NBINS)
        tau_b_sub = fit_tau(Hs_tr, bg, NBINS, objective=args.objective)
        A_sub = miou(confusion_at(Hs_te, np.full(nc, args.tau), bg, NBINS))
        B_sub = miou(confusion_at(Hs_te, tau_b_sub, bg, NBINS))
        gate_rows.append((k + 1, B_exact - A_exact, B_sub - A_sub))

        w, _ = fit_scale(Str, Gtr, bg, nc, NBINS, grid, args.objective,
                         args.w_rounds, args.tau_rounds)
        Hc_tr = hist_at(Str, Gtr, w, nc, NBINS)
        tau_c = fit_tau(Hc_tr, bg, NBINS, objective=args.objective)
        Hc_te = hist_at(Ste, Gte, w, nc, NBINS)
        C_sub = miou(confusion_at(Hc_te, tau_c, bg, NBINS))

        pc = per_class_iou(confusion_at(Hc_te, tau_c, bg, NBINS)) \
            - per_class_iou(confusion_at(Hs_te, tau_b_sub, bg, NBINS))
        rows.append((k + 1, A_sub, B_sub, C_sub, w.copy(), tau_c.copy(), pc))
        print(f'    A {A_sub:.2f}   B {B_sub:.2f}   C {C_sub:.2f}   '
              f'(C−B {C_sub - B_sub:+.2f})\n')

    gains_ba = np.array([r[2] - r[1] for r in rows])
    gains_cb = np.array([r[3] - r[2] for r in rows])
    pcs = np.nanmean(np.array([r[6] for r in rows]), axis=0)

    # ---------------- the gate
    ge = np.array([g[1] for g in gate_rows])
    gs = np.array([g[2] for g in gate_rows])
    disc = float(np.max(np.abs(ge - gs)))
    passed = disc <= args.gate

    md = ['# Per-class scaling before the argmax\n',
          f'- cache: `{args.cache}`  |  tiles: **{T}**  |  classes: **{nc}**, '
          f'catch-all `{LB.names[bg]}`',
          f'- published τ: **{args.tau}**  |  {args.folds}-fold  |  objective '
          f'**`{args.objective}`**  |  subsample **{args.subsample}** px/tile\n',
          'Rule: `pred = argmax_c (w_c · s_c)`, then keep `pred` if '
          '`s_pred ≥ τ_pred`. The threshold reads the **raw** score, so `w` is '
          'confined to the argmax — a scale applied afterwards is monotone and '
          'would merely reparameterise τ.\n',
          '| rung | thresholds | scale | what it is |', '|---|---|---|---|',
          f'| **A** | published {args.tau} | 1 | the baseline |',
          '| **B** | fitted | 1 | **our current method** |',
          '| **C** | fitted | fitted | this experiment |',
          '\n⭐ **The result is C − B.** C − A proves nothing: B already delivers '
          'that.\n',
          '## Gate: does the pixel subsample reproduce the exact answer?\n',
          'Rung B computed two ways — exactly over every pixel from per-tile '
          'histograms, and from the subsample the `w` search uses. The `w` result '
          'is only readable if these agree.\n',
          '| fold | B−A exact | B−A subsample | difference |', '|---|---|---|---|']
    for k, e, s_ in gate_rows:
        md.append(f'| {k} | {e:+.2f} | {s_:+.2f} | {e - s_:+.2f} |')
    md.append(f'\nLargest disagreement **{disc:.3f}** mIoU against a bar of '
              f'{args.gate:.2f}. ' +
              ('✅ **Gate passed** — the subsample is a valid instrument here.\n'
               if passed else
               '⛔ **GATE FAILED.** The subsample does not reproduce the exact '
               'answer, so nothing below can be attributed to `w` rather than to '
               'sampling noise. Raise `--subsample` and re-run.\n'))

    md += ['## Result\n',
           '| fold | A published τ | B per-class τ | C + scaling | **C − B** |',
           '|---|---|---|---|---|']
    for k, a, b, c, *_ in rows:
        md.append(f'| {k} | {a:.2f} | {b:.2f} | {c:.2f} | **{c - b:+.2f}** |')
    md += [f'\n- **B − A = {gains_ba.mean():+.2f} ± {gains_ba.std(ddof=1):.2f}** '
           f'— the existing method, reproduced here as a sanity check.',
           f'- **C − B = {gains_cb.mean():+.2f} ± {gains_cb.std(ddof=1):.2f}** '
           f'— what the argmax reordering adds, '
           f'{int((gains_cb > 0).sum())}/{args.folds} folds positive, '
           f'range {gains_cb.min():+.2f} to {gains_cb.max():+.2f}.\n']

    md += ['## Fitted scales, per fold\n',
           '| fold | ' + ' | '.join(LB.names) + ' |',
           '|---' * (nc + 1) + '|']
    for k, _, _, _, w, _, _ in rows:
        md.append(f'| {k} | ' + ' | '.join(f'{x:.2f}' for x in w) + ' |')
    W = np.array([r[4] for r in rows])
    md.append('| **mean** | ' + ' | '.join(f'**{x:.2f}**' for x in W.mean(0)) + ' |')
    md.append('\nValues are renormalised to geometric mean 1, since only ratios '
              'affect an argmax. A class above 1 wins more argmaxes than before; '
              'below 1, fewer.\n')

    md += ['## Mean per-class Δ IoU, C over B\n', '| class | Δ |', '|---|---|']
    for c in range(nc):
        if np.isfinite(pcs[c]):
            md.append(f'| {LB.names[c]}{" *(catch-all)*" if c == bg else ""} | '
                      f'**{pcs[c]:+.2f}** |')

    # ---------------- verdict
    m, sd = gains_cb.mean(), gains_cb.std(ddof=1)
    allpos = bool((gains_cb > 0).all())
    md.append('\n## Verdict\n')
    if not passed:
        md.append('⛔ **Unreadable — the subsampling gate failed.** Fix that before '
                  'interpreting anything above.\n')
    elif m - 2 * sd > 0 and allpos:
        md += [f'⭐ **Scaling before the argmax adds {m:+.2f} ± {sd:.2f} mIoU over '
               f'per-class thresholds alone**, every fold positive.\n',
               'That is the family the completeness argument explicitly does *not* '
               'cover, so it extends the method rather than restating it. Before it '
               'is written up: (1) verify end-to-end in the segmentor, as §9c did '
               'for per-class τ — a cached-histogram result is a prediction until '
               'the pipeline reproduces it; (2) check the fitted `w` against the '
               'confusion table, since the mechanism should be visible as the '
               'classes that were absorbing other classes being scaled **down**.\n']
    elif m > 0 and not allpos:
        md.append(f'⚠️ **Positive on average ({m:+.2f} ± {sd:.2f}) but not on every '
                  f'fold** ({int((gains_cb > 0).sum())}/{args.folds}). With '
                  f'{nc - 1} extra parameters fitted on the same tiles, that is the '
                  'signature of overfitting rather than of a mechanism. Report it as '
                  'inconclusive, or re-run with a larger calibration budget before '
                  'claiming it.\n')
    else:
        md += [f'⛔ **Scaling before the argmax buys nothing: {m:+.2f} ± {sd:.2f} '
               f'mIoU over per-class thresholds.**\n',
               'A clean and useful negative. The completeness argument in the method '
               'section says per-class τ is everything available *after* the argmax; '
               'this says the obvious way to attack what lies *before* it does not '
               'pay either, at this supervision budget. State it as bounding the '
               'larger family, not as a failed attempt.\n']

    md.append(f'\n⚠️ Rungs B and C are compared on **identical subsampled pixels**, '
              f'so the difference is the rule and not the sample. Rung B is also '
              f'reported exactly above; the exact five-fold value is '
              f'**{ge.mean():+.2f}**.\n')

    text = '\n'.join(md) + '\n'
    print('\n' + text)
    if args.md:
        p = Path(args.md).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f'wrote {p}')


if __name__ == '__main__':
    main()
