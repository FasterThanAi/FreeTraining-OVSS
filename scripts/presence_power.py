"""
LEVER 4 -- should the presence gate be applied equally to every class?

THE GAP. SegEarth-OV3 multiplies every class's dense score by a single scalar
per class per tile:

    P_final_c = P_fused_c * S_pres_c            # same treatment for every class

S_pres is a HARD CEILING -- no pixel of class c can score above it in that tile.
And the classes sit nowhere near each other on it (WEEK1 9.2b, median over 1669
LoveDA tiles):

    road 0.91 | building 0.84 | water 0.77 | agricultural 0.60
    barren 0.55 | forest 0.45 | background 0.022

`forest` gets HALF the ceiling `road` does, and forest is the worst class in the
dataset (33.78 IoU, 34.6% of its pixels discarded).

⭐ AND THE KNOWN FAILURE PREDICTS THIS SHOULD WORK. Turning presence gating off
globally costs -11.97 mIoU (WEEK1 9.2b) -- but the measured reason is
CLASS-SPECIFIC: background's presence is 0.022, so the gate is mostly holding
BACKGROUND down. Remove it globally and background surges and wins everywhere
(healthy tiles go 0.46% -> 54.11% discard). A per-class weight can do the thing
the global switch could not: keep `background` gated while un-gating `forest`.
A knob that fails globally for a per-class reason is the definition of a knob
that should be per class.

THE RULE:

    s_c(gamma) = P_fused_c * S_pres_c ** gamma_c

  * gamma = 1   ->  EXACTLY the published rule
  * gamma = 0   ->  presence gating OFF for this class alone
  * gamma > 1   ->  gating amplified for this class

⭐ WHY IT IS NOT ABSORBED BY LEVER 2. w_c is one constant for the whole dataset;
S_pres varies TILE BY TILE. So gamma reweights per tile and no per-class constant
can express it. It also escapes the completeness argument for per-class tau,
which bounds only maps that are the same on every tile.

⭐ AND IT COSTS NO GPU. `logits` (post-presence, per class) and `spres` are both
already in every --cache-full cache, so P_fused_c = logits_c / S_pres_c recovers
the ungated score and gamma re-gates it. No re-run, full splits, CPU only.

RUNGS:

    A  published tau, w = 1,    gamma = 1     the baseline
    B  fitted tau,    w = 1,    gamma = 1     lever 1
    C  fitted tau,    fitted w, gamma = 1     lever 1 + 2  (current method)
    E  fitted tau,    fitted w, fitted gamma  this experiment

⭐ The result is E - C.

⛔ THE IDENTITY GATE. At gamma = 1 the reconstruction must reproduce the cached
`logits` exactly, or the ungate/re-gate is not the pipeline and E is refused.

⚠️ AN APPROXIMATION THAT MUST BE REPORTED. The segmentor applies presence per
QUERY, before the synonym collapse, while `spres` is cached per CLASS (max over
that class's queries). For a class with ONE prompt the two coincide and this is
exact; for LoveDA's `building,house`, `forest,tree` and `barren,bareland,soil`
it is not, because max_q(f_q * p_q) is not (max_q f_q) * (max_q p_q). The script
reports which classes are exact. A deployment would apply p_q ** gamma_c per
query, so an end-to-end check is required before any synonym class's gamma is
quoted as deployed -- the same rule 9c applied to per-class tau.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels                                                     # noqa: E402
from tau_oracle import confusion_at, per_class_iou, miou, NBINS   # noqa: E402
from tau_cv import fit as fit_tau, obj_miou                       # noqa: E402

# gamma grid. 1.0 is a grid point and is swept FIRST (see GAMMA_ORDER), so the
# published rule wins every tie and a class whose gating is irrelevant keeps it
# instead of drifting to whichever endpoint the loop happens to reach first.
GAMMA = np.round(np.linspace(0.0, 2.0, 11), 2)
GAMMA_ORDER = np.argsort(np.abs(GAMMA - 1.0), kind='stable')
EPS = 1e-3          # S_pres below this leaves P_fused undetermined in float16

W_GRID = np.round(np.exp(np.linspace(np.log(0.40), np.log(2.50), 11)), 3)


def regate(F, P, gamma):
    """P_fused * S_pres ** gamma. F is (N, px), P is (N,) per-class presence."""
    return F * np.power(P, gamma)[:, None]


def load(files, nsub, nc, rng):
    F, P, G, LG = [], [], [], []
    nviews, bad_pres = set(), 0
    for i, f in enumerate(files):
        if (i + 1) % 100 == 0 or i + 1 == len(files):
            print(f'  {i + 1}/{len(files)}', flush=True)
        z = np.load(f)
        if 'logits' not in z.files:
            raise SystemExit(
                f'{f.name} has no `logits`. This needs a --cache-full cache:\n\n'
                f'    python scripts/measure_discard_rate.py --tau <tau> '
                f'--cache-full --out <dir>\n')
        gt = z['gt'].ravel()
        keep = np.flatnonzero(gt > 0)
        if keep.size == 0:
            continue
        if keep.size > nsub:
            keep = rng.choice(keep, size=nsub, replace=False)
        lg = z['logits'].astype(np.float32).reshape(nc, -1)[:, keep]
        sp = np.asarray(z['spres'], dtype=np.float32)
        nviews.add(sp.shape[0])
        p = np.nanmean(sp, axis=0)[:nc]
        p = np.where(np.isfinite(p) & (p > EPS), p, 1.0)
        bad_pres += int((np.asarray(z['spres'], np.float32)[:, :nc] <= EPS).any())
        F.append(np.clip(lg / p[:, None], 0.0, 1.0))   # P_fused, ungated
        P.append(p)
        G.append(gt[keep].astype(np.int32) - 1)
        LG.append(lg)
    return F, P, G, LG, nviews, bad_pres


def hists(F, P, G, idx, gamma, w, nc, nbins):
    H = np.zeros((nc, nc, nbins + 1), np.int64)
    for t in idx:
        s = regate(F[t], P[t], gamma)
        pred = np.argmax(s * w[:, None], axis=0)
        conf = s[pred, np.arange(s.shape[1])]         # RAW score, as in predict()
        b = np.clip(np.rint(conf * nbins).astype(np.int64), 0, nbins)
        np.add.at(H, (G[t], pred, b), 1)
    return H


def _coord(F, P, G, idx, nc, bg, objective, grid, order, vec, w, gamma, which,
           tau_rounds):
    """One coordinate-ascent pass over `which` in ('gamma','w'). tau is refitted
    per candidate -- freezing it caps the search below its own answer, which is
    the bug test_head_fusion.py exists to catch."""
    for c in range(nc):
        best, bs = vec[c], -np.inf
        for j in order:
            trial = vec.copy(); trial[c] = grid[j]
            g, ww = (trial, w) if which == 'gamma' else (gamma, trial)
            H = hists(F, P, G, idx, g, ww, nc, NBINS)
            t = fit_tau(H, bg, NBINS, rounds=tau_rounds, objective=objective)
            sc = obj_miou(confusion_at(H, t, bg, NBINS), bg, objective)
            if sc > bs + 1e-9:
                bs, best = sc, grid[j]
        vec[c] = best
    return vec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True, help='a --cache-full cache')
    ap.add_argument('--tau', type=float, required=True)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--subsample', type=int, default=40000)
    ap.add_argument('--objective', choices=['all', 'real'], default='real')
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--tau-rounds', type=int, default=2)
    ap.add_argument('--gate', type=float, default=0.01)
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
    print(f'{T} tiles | published τ = {args.tau} | γ grid {GAMMA[0]}..{GAMMA[-1]}\n')

    order = np.random.default_rng(args.seed).permutation(T)
    rng = np.random.default_rng(args.seed + 1000)
    F, P, G, LG, nviews, bad_pres = load(files, args.subsample, nc, rng)

    one = np.ones(nc)
    err, n = 0.0, 0
    for t in range(len(F)):
        err += float(np.abs(regate(F[t], P[t], one) - LG[t]).mean()); n += 1
    err = err / n if n else float('inf')
    print(f'\n  views per tile: {sorted(nviews)}   tiles with S_pres ≤ {EPS}: {bad_pres}')
    print(f'  identity gate: mean |P_fused·S_pres − logits| = {err:.5f} '
          f'(bar {args.gate})  {"PASS" if err <= args.gate else "FAIL"}')
    if err > args.gate:
        raise SystemExit(
            f'\n⛔ THE UNGATE/RE-GATE IS NOT THE PIPELINE (off by {err:.5f}).\n'
            f'   At γ = 1 it must reproduce the cached `logits` exactly. Most\n'
            f'   likely cause: presence applied per QUERY before the synonym\n'
            f'   collapse, so a class-level S_pres cannot invert it. Reporting\n'
            f'   γ off this cache would measure a different model.\n')

    folds = np.array_split(order, args.folds)
    rows = []
    for k in range(args.folds):
        ev, ca = folds[k], np.concatenate(
            [folds[j] for j in range(args.folds) if j != k])
        print(f'\nfold {k + 1}:')
        tau_b = fit_tau(hists(F, P, G, ca, one, one, nc, NBINS), bg, NBINS,
                        objective=args.objective)
        A = miou(confusion_at(hists(F, P, G, ev, one, one, nc, NBINS),
                              np.full(nc, args.tau), bg, NBINS))
        B = miou(confusion_at(hists(F, P, G, ev, one, one, nc, NBINS),
                              tau_b, bg, NBINS))

        w = np.ones(nc)
        for _ in range(args.rounds):
            w = _coord(F, P, G, ca, nc, bg, args.objective, W_GRID,
                       np.arange(len(W_GRID)), w, w, one, 'w', args.tau_rounds)
        tau_c = fit_tau(hists(F, P, G, ca, one, w, nc, NBINS), bg, NBINS,
                        objective=args.objective)
        C = miou(confusion_at(hists(F, P, G, ev, one, w, nc, NBINS),
                              tau_c, bg, NBINS))

        gam = np.ones(nc)
        for _ in range(args.rounds):
            gam = _coord(F, P, G, ca, nc, bg, args.objective, GAMMA,
                         GAMMA_ORDER, gam, w, gam, 'gamma', args.tau_rounds)
        tau_e = fit_tau(hists(F, P, G, ca, gam, w, nc, NBINS), bg, NBINS,
                        objective=args.objective)
        E = miou(confusion_at(hists(F, P, G, ev, gam, w, nc, NBINS),
                              tau_e, bg, NBINS))

        print(f'    γ = ' + ', '.join(f'{g:.1f}' for g in gam))
        print(f'    A {A:.2f}   B {B:.2f}   C {C:.2f}   E {E:.2f}   (E−C {E - C:+.2f})')
        rows.append(dict(A=A, B=B, C=C, E=E, gam=gam))

    ec = np.array([r['E'] - r['C'] for r in rows])
    m, sd, pos = ec.mean(), ec.std(ddof=1), int((ec > 0).sum())
    Gm = np.array([r['gam'] for r in rows])

    md = ['# Per-class presence weight — should the gate apply equally to every class?\n',
          f'- cache `{args.cache}` | tiles **{T}** | τ **{args.tau}** | {args.folds}-fold '
          f'| objective **`{args.objective}`**',
          f'- γ grid **{GAMMA[0]}–{GAMMA[-1]}**, symmetric about **γ = 1 = the published '
          f'rule**\n',
          '`s_c = P_fused_c · S_pres_c^γ_c`. γ = 0 turns gating off for that class alone; '
          'γ = 1 is the baseline. ⭐ Not absorbed by lever 2: `w_c` is one constant per '
          'dataset, `S_pres` varies tile by tile.\n',
          f'✅ **Identity gate {err:.5f}** (bar {args.gate}). Views per tile {sorted(nviews)}.\n',
          '| fold | A published | B +τ | C +scale | E +presence | **E − C** |',
          '|---|---|---|---|---|---|']
    for i, r in enumerate(rows):
        md.append(f'| {i+1} | {r["A"]:.2f} | {r["B"]:.2f} | {r["C"]:.2f} | {r["E"]:.2f} '
                  f'| **{r["E"]-r["C"]:+.2f}** |')
    md += [f'\n- **E − C = {m:+.2f} ± {sd:.2f}**, {pos}/{args.folds} folds positive, '
           f'mean − 2·sd = **{m - 2*sd:+.2f}**\n', '## Fitted γ per class\n',
           '| class | ' + ' | '.join(f'f{i+1}' for i in range(args.folds)) + ' | mean |',
           '|---|' + '---|' * (args.folds + 1)]
    for c, name in enumerate(LB.names):
        md.append(f'| {name} | ' + ' | '.join(f'{Gm[k, c]:.1f}' for k in range(args.folds))
                  + f' | **{Gm[:, c].mean():.2f}** |')
    md.append('\n`γ < 1` = this class wants **less** presence gating; `γ > 1` = **more**; '
              '`γ = 1` = the published gate is already right.\n')

    stay = int((np.abs(Gm - 1.0) < 0.15).all(axis=0).sum())
    gate_ok = (m - 2 * sd) > 0 and pos == args.folds
    md.append('## Verdict\n')
    if gate_ok:
        md.append(f'⭐ **A per-class presence weight adds {m:+.2f} ± {sd:.2f} mIoU on top of '
                  f'per-class τ and per-class scaling**, {pos}/{args.folds} folds, '
                  f'mean − 2·sd {m - 2*sd:+.2f}. ⚠️ A cached prediction until `eval.py` '
                  f'reproduces it (§9c), and ⚠️ approximate for any class with synonym '
                  f'prompts — see the module docstring.')
    elif pos == args.folds:
        md.append(f'⚠️ **Promising, not established: {m:+.2f} ± {sd:.2f}**, every fold '
                  f'positive but the spread covers zero. Do not write it up at this width.')
    else:
        md.append(f'⛔ **Null: {m:+.2f} ± {sd:.2f}**, {pos}/{args.folds} folds positive. '
                  f'A per-class presence weight buys nothing on top of levers 1 and 2.')
    md.append(f'\n⭐ **{stay} of {nc} classes keep γ = 1 in every fold** — the published gate '
              f'is already right for those, and that is a result about the baseline whichever '
              f'way E − C goes.')
    md.append(f'\n⚠️ **`background`\'s γ is only weakly identified under `--objective real`**, '
              f'which does not score it. Read its row as what the fit did to help the real '
              f'classes, not as a statement about the catch-all.')

    out = '\n'.join(md)
    print('\n' + out)
    if args.md:
        Path(args.md).expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).expanduser().write_text(out)
        print(f'\nwrote {args.md}')


if __name__ == '__main__':
    main()
