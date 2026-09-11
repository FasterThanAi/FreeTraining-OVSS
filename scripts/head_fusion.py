"""
LEVER 3 -- is `max(P_sem, P_inst)` the right fusion for EVERY class?

THE GAP, in the baseline's own words. SegEarth-OV3 has two heads because the
right one is class-dependent: the instance decoder is sharp on countable
"things" and fragments amorphous "stuff"; the semantic head is the reverse.
We reproduced that on our own data (ANALYSIS §4.5): one LoveDA tile, `building`
returns 14 instance masks, `road` returns 2.

Then `segearthov3_segmentor.py:196-210` fuses them with

    s_c = max(P_inst_agg_c, P_sem_c)                 # identical for every class

A hardcoded, class-INDEPENDENT rule, applied to a duality the paper itself
argues is class-dependent. That is the third and last global choice the baseline
makes, and the only one we have not fitted:

    global choice                       our fix          measured
    one tau for all classes             per-class tau    +1.18 LoveDA, +2.51 ConInfer
    implicit w_c = 1 for all classes    per-class scale  +1.16 LoveDA, +4.92 Potsdam
    max(P_sem, P_inst) for all classes  THIS SCRIPT      -

THE RULE, and why it is parameterised this way:

    s_c = max(a_c * P_sem_c, b_c * P_inst_c),   with  max(a_c, b_c) = 1

One free parameter per class: the ratio rho_c = b_c / a_c.

  * rho = 1        ->  a = b = 1  ->  EXACTLY the published rule. The baseline
                       is inside the family, so rung D nests rung C.
  * rho -> 0       ->  the instance head is switched off for this class
  * rho -> inf     ->  the semantic head is switched off for this class

⭐ WHY `max(a, b) = 1` AND NOT A FREE PAIR. Lever 2 already fits a free per-class
scale w_c for the argmax. Any OVERALL per-class scaling inside the fusion is
therefore absorbed by w_c and would be unidentifiable -- two knobs for one
degree of freedom, which is how a fit starts wandering. Normalising the larger
weight to 1 leaves exactly the part lever 2 CANNOT express: the RATIO between
the heads, which changes which head wins at a pixel rather than rescaling the
result. It also keeps s_c in [0, 1], so the tau grid still covers the score --
see CONINFER_RESULTS, where a published threshold sat below the score floor and
could never fire.

FOUR RUNGS:

    A  published tau,  w = 1,      rho = 1        the baseline
    B  fitted tau,     w = 1,      rho = 1        lever 1
    C  fitted tau,     fitted w,   rho = 1        lever 1 + 2  (current method)
    D  fitted tau,     fitted w,   fitted rho     this experiment

⭐ The result is D - C. D - A proves nothing; C already delivers that.

⛔ THE IDENTITY GATE, and it is not optional. The cached stacks are
PRE-presence and PRE-accumulation-order, so reconstructing the pipeline from
them is only exact if inference ran as a single whole-image view. Under
sliding-window with overlap the pipeline computes

    accumulate( max(sem, inst) * presence )

while this script computes max over ALREADY-accumulated stacks, and a sum of
maxima is not the maximum of sums. So rung C is recomputed here from `inst`,
`sem` and the presence vector at rho = 1, and must reproduce the cached `logits`
to float16 precision. If it does not, the reconstruction is an approximation and
the run REFUSES to report D. Same discipline as verify_perclass_tau.py.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels                                                     # noqa: E402
from tau_oracle import confusion_at, per_class_iou, miou, NBINS   # noqa: E402
from tau_cv import fit as fit_tau, obj_miou                       # noqa: E402


# rho grid, log-spaced and SYMMETRIC about 1 so the baseline is a grid point and
# neither direction is favoured by the spacing.
RHO = np.round(np.exp(np.linspace(np.log(0.25), np.log(4.0), 13)), 4)


def fuse(sem, inst, rho):
    """max(a*sem, b*inst) with max(a,b)=1 and b/a=rho. Shapes (N, P)."""
    a = np.minimum(1.0, 1.0 / rho)          # rho>1 -> a<1 (semantic discounted)
    b = np.minimum(1.0, rho)                # rho<1 -> b<1 (instance discounted)
    return np.maximum(sem * a[:, None], inst * b[:, None])


def load(files, nsub, nc, rng, need_gate):
    """Per-tile: subsampled (sem, inst) stacks, presence vector, gt, and -- for
    the gate -- the cached `logits` stack at the same pixels."""
    SEM, INST, PRES, G, LG = [], [], [], [], []
    nviews = set()
    for i, f in enumerate(files):
        if (i + 1) % 50 == 0 or i + 1 == len(files):
            print(f'  {i + 1}/{len(files)}', flush=True)
        z = np.load(f)
        for k in ('sem', 'inst'):
            if k not in z.files:
                raise SystemExit(
                    f'{f.name} has no `{k}` key. This experiment needs a cache '
                    f'written with --cache-heads:\n\n'
                    f'    python scripts/measure_discard_rate.py --tau <tau> '
                    f'--cache-heads --out <dir>\n')
        gt = z['gt'].ravel()
        keep = np.flatnonzero(gt > 0)
        if keep.size == 0:
            continue
        if keep.size > nsub:
            keep = rng.choice(keep, size=nsub, replace=False)
        sem = z['sem'].astype(np.float32).reshape(nc, -1)[:, keep]
        inst = z['inst'].astype(np.float32).reshape(nc, -1)[:, keep]
        nviews.add(z['spres'].shape[0])
        # ⭐ The cached stacks are ALREADY presence-gated and ALREADY collapsed
        # from queries to classes (measure_discard_rate --cache-heads does both,
        # in that order, because max commutes). So nothing is applied here, and
        # `max(sem, inst)` must equal `logits` outright.
        SEM.append(sem); INST.append(inst); PRES.append(np.ones(nc, np.float32))
        G.append(gt[keep].astype(np.int32) - 1)
        if need_gate:
            LG.append(z['logits'].astype(np.float32).reshape(nc, -1)[:, keep]
                      if 'logits' in z.files else None)
    return SEM, INST, PRES, G, LG, nviews


def hists(SEM, INST, PRES, G, idx, rho, w, nc, nbins):
    """(gt, pred, conf-bin) histogram over the tiles in `idx`, at this rho."""
    H = np.zeros((nc, nc, nbins + 1), np.int64)
    for t in idx:
        s = fuse(SEM[t], INST[t], rho)
        pred = np.argmax(s * w[:, None], axis=0)
        conf = s[pred, np.arange(s.shape[1])]        # RAW score, as in the segmentor
        b = np.clip(np.rint(conf * nbins).astype(np.int64), 0, nbins)
        np.add.at(H, (G[t], pred, b), 1)
    return H


# ⭐ Sweep order: rho = 1 FIRST, then outward. Combined with the strict `>`
# below this makes the PUBLISHED rule win every tie, so a class whose fusion is
# irrelevant to the objective keeps `max` instead of drifting to whichever grid
# point happens to come first. Without it the fitter reported rho = 0.25 on
# classes where every rho scores identically, which reads as a finding and is
# an artefact of grid order.
RHO_ORDER = np.argsort(np.abs(np.log(RHO)), kind='stable')


def fit_rho(SEM, INST, PRES, G, idx, w, tau0, nc, bg, objective,
            rounds, tau_rounds):
    """Coordinate ascent on rho, with tau REFITTED inside the search.

    ⚠️ NOT separable -- rho changes the argmax, so it couples the vector exactly
    as lever 2's w does. Reported as greedy, never as exact.

    ⛔ tau MUST be refitted per candidate, and this was a bug. Holding tau at
    rung C's value while sweeping rho caps the objective below what the correct
    rho reaches: on the unit test in `test_head_fusion.py` the true vector
    scores 50.00 with tau frozen and 100.00 with tau refitted, so the search
    could not see its own answer and settled on a vector 50 points worse. Same
    reason argmax_reorder.py carries --tau-rounds inside the w search.
    """
    rho, tau = np.ones(nc), tau0
    for _ in range(rounds):
        for c in range(nc):
            best, bs = rho[c], -np.inf
            for j in RHO_ORDER:
                trial = rho.copy(); trial[c] = RHO[j]
                H = hists(SEM, INST, PRES, G, idx, trial, w, nc, NBINS)
                t = fit_tau(H, bg, NBINS, rounds=tau_rounds, objective=objective)
                sc = obj_miou(confusion_at(H, t, bg, NBINS), bg, objective)
                if sc > bs + 1e-9:            # strict: ties keep the incumbent,
                    bs, best = sc, RHO[j]     # and rho = 1 is swept first
            rho[c] = best
        tau = fit_tau(hists(SEM, INST, PRES, G, idx, rho, w, nc, NBINS),
                      bg, NBINS, objective=objective)
    return rho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True, help='a --cache-heads cache')
    ap.add_argument('--tau', type=float, required=True, help='the published threshold')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--subsample', type=int, default=40000)
    ap.add_argument('--objective', choices=['all', 'real'], default='real')
    ap.add_argument('--rho-rounds', type=int, default=2)
    ap.add_argument('--tau-rounds', type=int, default=2,
                    help='tau coordinate-ascent passes INSIDE the rho search. '
                         'Freezing tau there is a bug -- see fit_rho.')
    ap.add_argument('--gate', type=float, default=0.01,
                    help='max mean |reconstructed - cached| score before the run '
                         'refuses to report D')
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
    print(f'{T} tiles | published τ = {args.tau} | ρ grid {RHO[0]}..{RHO[-1]}\n')

    order = np.random.default_rng(args.seed).permutation(T)
    rng = np.random.default_rng(args.seed + 1000)
    SEM, INST, PRES, G, LG, nviews = load(files, args.subsample, nc, rng, True)

    # ---------------------------------------------------------------- the gate
    print(f'\n  views per tile: {sorted(nviews)}')
    gate_err, gate_n = 0.0, 0
    one = np.ones(nc)
    for t in range(len(SEM)):
        if LG[t] is None:
            continue
        recon = fuse(SEM[t], INST[t], one)
        gate_err += float(np.abs(recon - LG[t]).mean()); gate_n += 1
    gate_err = gate_err / gate_n if gate_n else float('inf')
    ok = gate_n > 0 and gate_err <= args.gate
    print(f'  identity gate: mean |max(sem,inst) − logits| = '
          f'{gate_err:.5f} (bar {args.gate})  {"PASS" if ok else "FAIL"}')
    if not ok:
        raise SystemExit(
            f'\n⛔ THE RECONSTRUCTION IS NOT THE PIPELINE.\n'
            f'   At ρ = 1 this script must reproduce the cached `logits` exactly;\n'
            f'   it is off by {gate_err:.5f} per score on average.\n'
            f'   Most likely cause: sliding-window inference with overlapping\n'
            f'   crops, where the pipeline fuses INSIDE the crop loop and a sum\n'
            f'   of maxima is not the maximum of sums. Reporting ρ off this\n'
            f'   cache would measure a different model from the one deployed.\n'
            f'   {"No `logits` found -- re-cache with --cache-heads." if gate_n == 0 else ""}\n')

    folds = np.array_split(order, args.folds)
    rows = []
    for k in range(args.folds):
        ev = folds[k]
        ca = np.concatenate([folds[j] for j in range(args.folds) if j != k])
        print(f'\nfold {k + 1}:')

        Hc = hists(SEM, INST, PRES, G, ca, one, one, nc, NBINS)
        He = hists(SEM, INST, PRES, G, ev, one, one, nc, NBINS)
        tau_b = fit_tau(Hc, bg, NBINS, objective=args.objective)
        A = miou(confusion_at(He, np.full(nc, args.tau), bg, NBINS))
        B = miou(confusion_at(He, tau_b, bg, NBINS))

        # Rung C is refitted here rather than imported from argmax_reorder:
        # that fitter consumes an already-fused stack, and C and D must come
        # from ONE code path or their difference measures the code, not ρ.
        w_c, tau_c = np.ones(nc), tau_b
        for _ in range(2):
            best = w_c.copy()
            for c in range(nc):
                bs, bw = -np.inf, best[c]
                for v in np.round(np.exp(np.linspace(np.log(0.40), np.log(2.50), 11)), 3):
                    tr = best.copy(); tr[c] = v
                    H = hists(SEM, INST, PRES, G, ca, one, tr, nc, NBINS)
                    sc = obj_miou(confusion_at(H, tau_c, bg, NBINS), bg, args.objective)
                    if sc > bs:
                        bs, bw = sc, v
                best[c] = bw
            w_c = best
            tau_c = fit_tau(hists(SEM, INST, PRES, G, ca, one, w_c, nc, NBINS),
                            bg, NBINS, objective=args.objective)
        C = miou(confusion_at(hists(SEM, INST, PRES, G, ev, one, w_c, nc, NBINS),
                              tau_c, bg, NBINS))

        rho = fit_rho(SEM, INST, PRES, G, ca, w_c, tau_c, nc, bg,
                      args.objective, args.rho_rounds, args.tau_rounds)
        tau_d = fit_tau(hists(SEM, INST, PRES, G, ca, rho, w_c, nc, NBINS),
                        bg, NBINS, objective=args.objective)
        D = miou(confusion_at(hists(SEM, INST, PRES, G, ev, rho, w_c, nc, NBINS),
                              tau_d, bg, NBINS))

        print(f'    ρ = ' + ', '.join(f'{r:.2f}' for r in rho))
        print(f'    A {A:.2f}   B {B:.2f}   C {C:.2f}   D {D:.2f}   (D−C {D - C:+.2f})')
        rows.append(dict(A=A, B=B, C=C, D=D, rho=rho, tau=tau_d))

    dc = np.array([r['D'] - r['C'] for r in rows])
    m, sd = dc.mean(), dc.std(ddof=1)
    pos = int((dc > 0).sum())
    R = np.array([r['rho'] for r in rows])

    md = [f'# Per-class head fusion — is `max(P_sem, P_inst)` right for every class?\n',
          f'- cache `{args.cache}` | tiles **{T}** | τ **{args.tau}** | '
          f'{args.folds}-fold | objective **`{args.objective}`**',
          f'- ρ grid **{RHO[0]}–{RHO[-1]}**, 13 points, symmetric about **ρ = 1 = the '
          f'published rule**\n',
          f'`s_c = max(a_c · P_sem_c, b_c · P_inst_c)`, `max(a_c, b_c) = 1`, '
          f'`ρ_c = b_c / a_c`. The overall per-class scale is left to lever 2, '
          f'which already fits it — ρ carries only the part lever 2 cannot '
          f'express, the **ratio** between the heads.\n',
          f'✅ **Identity gate: {gate_err:.5f}** against a bar of {args.gate}. At ρ = 1 the '
          f'reconstruction reproduces the cached `logits`, so ρ ≠ 1 measures the '
          f'deployed pipeline and not an approximation of it. Views per tile: '
          f'{sorted(nviews)}.\n',
          '| fold | A published | B +τ | C +scale | D +fusion | **D − C** |',
          '|---|---|---|---|---|---|']
    for i, r in enumerate(rows):
        md.append(f'| {i + 1} | {r["A"]:.2f} | {r["B"]:.2f} | {r["C"]:.2f} | '
                  f'{r["D"]:.2f} | **{r["D"] - r["C"]:+.2f}** |')
    md += [f'\n- **D − C = {m:+.2f} ± {sd:.2f}**, {pos}/{args.folds} folds positive, '
           f'mean − 2·sd = **{m - 2 * sd:+.2f}**\n',
           '## Fitted ρ per class\n',
           '| class | ' + ' | '.join(f'f{i+1}' for i in range(args.folds)) + ' | mean |',
           '|---|' + '---|' * (args.folds + 1)]
    for c, name in enumerate(LB.names):
        md.append(f'| {name} | ' + ' | '.join(f'{R[k, c]:.2f}' for k in range(args.folds))
                  + f' | **{R[:, c].mean():.2f}** |')
    md.append('\n`ρ > 1` = this class wants the **instance** head; `ρ < 1` = the '
              '**semantic** head; `ρ = 1` = the published `max` is already right.\n')

    gate = (m - 2 * sd) > 0 and pos == args.folds
    stay = int((np.abs(np.log(R)) < np.log(1.3)).all(axis=0).sum())
    md += ['## Verdict\n']
    if gate:
        md.append(f'⭐ **Per-class fusion adds {m:+.2f} ± {sd:.2f} mIoU on top of both '
                  f'existing levers**, {pos}/{args.folds} folds, mean − 2·sd '
                  f'{m - 2 * sd:+.2f}. The baseline\'s third global choice is also '
                  f'the wrong shape. ⚠️ A cached-histogram prediction until an '
                  f'`eval.py` pass reproduces it, per §9c.')
    elif pos == args.folds:
        md.append(f'⚠️ **Promising, not established: {m:+.2f} ± {sd:.2f}**, every fold '
                  f'positive (1 in {2**args.folds} by sign test) but the spread covers '
                  f'zero. Do not write it up at this width.')
    else:
        md.append(f'⛔ **Null: {m:+.2f} ± {sd:.2f}**, {pos}/{args.folds} folds positive. '
                  f'Per-class head fusion buys nothing on top of per-class τ and '
                  f'per-class scaling.')
    md.append(f'\n⭐ **{stay} of {nc} classes keep ρ within ±30% of 1 in every fold** — '
              f'those are classes for which the published `max` is already the right '
              f'rule, and that is a result about the baseline whichever way D − C goes.')
    if not gate and pos == args.folds:
        md.append('\n⚠️ Check the ρ table before concluding: if ρ is stable across folds '
                  'while the gains scatter, the fit is stable and the evaluation is '
                  'noisy, which argues for more tiles rather than against the rule.')

    out = '\n'.join(md)
    print('\n' + out)
    if args.md:
        Path(args.md).expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).expanduser().write_text(out)
        print(f'\nwrote {args.md}')


if __name__ == '__main__':
    main()
