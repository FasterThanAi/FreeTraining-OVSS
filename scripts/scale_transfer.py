"""Lever 2 across splits: fit (w, tau) on one split, evaluate EXACTLY on another.

`argmax_reorder.py` answers "does per-class scaling help?" with folds inside one
split. This answers the deployable question: **fit it somewhere else and does it
still work?** -- the same question `tau_transfer.py` answered for lever 1 (+1.03 on
UAVid), and the one WEEK3 §9c insists on before anything is written up.

The decision rule, verified against the segmentor line by line
(`segearthov3_segmentor.py:496-508`):

    pred = argmax_c (w_c * s_c)          the argmax reads the SCALED score
    keep if s_pred >= tau_pred           the threshold reads the RAW score

`hist_at` implements exactly that, and this script uses `hist_at` rather than a
re-implementation, so the two cannot drift apart.

Arms, all evaluated EXACTLY over every pixel of the destination:

  A  published tau, no scale                   the baseline
  B  per-class tau fitted on src, no scale     lever 1 alone -- the thing to beat
  C  per-class tau AND scale fitted on src     THE TEST
  D  (w, tau) fitted on the DESTINATION        a bound, not a method

plus a permutation control: arm C's `w` shuffled among the real classes with tau
held fixed, which isolates whether the argmax REORDERING transfers or merely the
fact that some classes were scaled at all.

Example:

  python scripts/scale_transfer.py \
    --src-cache ~/outputs/uavid_train_full/cache \
    --dst-cache ~/outputs/uavid_val_full/cache \
    --tau 0.3 --objective real --expect 56.86 \
    --src-exclude-re '_(flipped|shifted)' \
    --base-cfg ~/SegEarth-OV-3/configs/cfg_uavid_val.py \
    --deploy-cfg ~/SegEarth-OV-3/configs/cfg_uavid_val_scale.py \
    --md ~/outputs/uavid_val_full/scale_transfer.md
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels                                                      # noqa: E402
from tau_oracle import confusion_at, miou, per_class_iou, NBINS     # noqa: E402
from tau_cv import fit as fit_tau                                   # noqa: E402
from argmax_reorder import hist_at, fit_scale, load_full            # noqa: E402


def pick(cache, exclude_re, what):
    files = sorted(Path(cache).expanduser().glob('*.npz'))
    if not files:
        raise SystemExit(f'no .npz under {cache}')
    if exclude_re:
        ex = re.compile(exclude_re)
        n0 = len(files)
        files = [f for f in files if not ex.search(f.stem)]
        if not files:
            raise SystemExit(f'--{what}-exclude-re removed all {n0} files')
        print(f'  {what}: dropped {n0 - len(files)} of {n0}')
    z = np.load(files[0])
    if 'logits' not in z.files:
        raise SystemExit(
            f'{files[0].name} has no `logits` key. Lever 2 needs the full '
            f'per-class score stack, which only --cache-full writes.')
    print(f'  {what}: {len(files)} tiles from {cache}')
    return files


def load_flat(files, nc, cap_gb=6.0):
    """Every labelled pixel of every tile, as one (n, nc) stack.

    Held in memory so the permutation control and the destination oracle are
    cheap; re-reading the cache per permutation would be 200 passes over it.
    """
    tot = 0
    for f in files:
        z = np.load(f)
        tot += int((z['gt'] > 0).sum())
    gb = tot * nc * 4 / 2**30
    print(f'  destination: {tot:,} labelled pixels -> {gb:.2f} GB in memory')
    if gb > cap_gb:
        raise SystemExit(
            f'that is above the {cap_gb} GB cap. Re-cache the destination with a '
            f'larger --cache-stride, or raise --mem-cap if the machine has room.')
    S = np.empty((tot, nc), np.float32)
    G = np.empty(tot, np.int32)
    i = 0
    for k, f in enumerate(files):
        z = np.load(f)
        gt = z['gt'].astype(np.int32)
        m = gt > 0
        n = int(m.sum())
        S[i:i + n] = z['logits'].astype(np.float32)[:, m].T
        G[i:i + n] = gt[m]
        i += n
        if (k + 1) % 25 == 0 or k + 1 == len(files):
            print(f'  {k + 1}/{len(files)}')
    assert i == tot
    return S, G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src-cache', required=True, help='fit here (--cache-full)')
    ap.add_argument('--dst-cache', required=True, help='evaluate here (--cache-full)')
    ap.add_argument('--tau', type=float, required=True, help='published threshold')
    ap.add_argument('--objective', choices=['all', 'real'], default='real')
    ap.add_argument('--src-exclude-re', default=None)
    ap.add_argument('--dst-exclude-re', default=None)
    ap.add_argument('--subsample', type=int, default=40000,
                    help='pixels per SOURCE tile for the w search')
    ap.add_argument('--w-rounds', type=int, default=3)
    ap.add_argument('--tau-rounds', type=int, default=3)
    ap.add_argument('--perms', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--mem-cap', type=float, default=6.0)
    ap.add_argument('--expect', type=float, default=None,
                    help='known published-tau mIoU on the DESTINATION. Aborts on '
                         'a mismatch: a cache holding the wrong tiles, or a label '
                         'convention slip, produces a complete plausible table.')
    ap.add_argument('--expect-tol', type=float, default=0.15)
    ap.add_argument('--deploy-cfg', default=None)
    ap.add_argument('--base-cfg', default=None)
    ap.add_argument('--md', default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # ---- gates -------------------------------------------------------------
    LS, LD = labels.from_cache(args.src_cache), labels.from_cache(args.dst_cache)
    if LS.names != LD.names:
        raise SystemExit(f'class lists differ.\n  src: {LS.names}\n  dst: {LD.names}\n'
                         f'A (w, tau) pair is indexed by class; transferring it '
                         f'across different label spaces is meaningless.')
    if LS.bg != LD.bg:
        raise SystemExit(f'catch-all index differs: {LS.bg} vs {LD.bg}')
    nc, bg = LD.n, LD.bg - 1
    print(f'  classes: {LD}')

    src = pick(args.src_cache, args.src_exclude_re, 'src')
    dst = pick(args.dst_cache, args.dst_exclude_re, 'dst')
    if {f.name for f in src} & {f.name for f in dst}:
        raise SystemExit('src and dst share filenames -- not disjoint splits; the '
                         'transfer number would be a leak')

    # ---- destination: every pixel, exactly ---------------------------------
    DS, DG = load_flat(dst, nc, args.mem_cap)
    ones = np.ones(nc)

    def ev(w, taus):
        return confusion_at(hist_at(DS, DG, w, nc, NBINS), taus, bg, NBINS)

    C_pub = ev(ones, np.full(nc, args.tau))
    m_pub = miou(C_pub)
    print(f'\n  published τ={args.tau} on dst: mIoU {m_pub:.2f}')
    if args.expect is not None:
        d = abs(m_pub - args.expect)
        if d > args.expect_tol:
            raise SystemExit(f'GATE FAILED: {m_pub:.2f} against an expected '
                             f'{args.expect:.2f} (|Δ| {d:.2f} > {args.expect_tol}). '
                             f'Every number below would be void.')
        print(f'  ✅ gate: within {d:.2f} of {args.expect:.2f}')

    # ---- source: fit ---------------------------------------------------------
    print('\n  loading source subsample')
    S, G, _ = load_full(src, args.subsample, nc, NBINS,
                        np.random.default_rng(args.seed + 1000))
    Sf, Gf = S.reshape(-1, nc), G.ravel()
    grid = np.round(np.exp(np.linspace(np.log(0.40), np.log(2.50), 11)), 3)

    print('\n  fitting tau alone on src (arm B)')
    tau_b = fit_tau(hist_at(Sf, Gf, ones, nc, NBINS), bg, NBINS,
                    rounds=args.tau_rounds, objective=args.objective)
    print('  fitting (w, tau) jointly on src (arm C)')
    w_c, tau_c = fit_scale(Sf, Gf, bg, nc, NBINS, grid, args.objective,
                           args.w_rounds, args.tau_rounds)

    # ---- destination oracle (a BOUND) ----------------------------------------
    print('\n  fitting (w, tau) on the DESTINATION — a bound, not a method')
    nsub = min(len(DG), args.subsample * len(dst))
    sel = rng.choice(len(DG), nsub, replace=False)
    w_d, tau_d = fit_scale(DS[sel], DG[sel], bg, nc, NBINS, grid, args.objective,
                           args.w_rounds, args.tau_rounds)

    rows = [('A  published τ, no scale', C_pub),
            ('B  per-class τ from src', ev(ones, tau_b)),
            ('C  + per-class scale from src', ev(w_c, tau_c)),
            ('D  (w, τ) fitted on dst', ev(w_d, tau_d))]
    print()
    for name, C in rows:
        print(f'  {name:<32} {miou(C):6.2f}  ({miou(C) - m_pub:+.2f})')
    d_b, d_c, d_d = (miou(C) - m_pub for _, C in rows[1:])
    print(f'\n  ⭐ lever 2 over lever 1 on the destination: {d_c - d_b:+.2f}')

    # ---- permutation control on w -------------------------------------------
    real = np.array([c for c in range(nc) if c != bg])
    # ⛔ A permutation draw can BE the identity, and then the "shuffle" is the
    # real assignment scored against itself. With 6 real classes that is 1 draw
    # in 720 and it never bit; with 2 it is half of them, and the control
    # reported 75%% of shuffles matching when the only genuine shuffle scored
    # -0.03 against +45.24. It also matters with MANY classes when two of them
    # are fitted to the same value -- UAVid gives `road` and `vegetation` both
    # 0.40, so swapping those two is a no-op dressed as a draw. Compare the
    # resulting VECTOR, not the index permutation.
    pd, degen = [], 0
    for _ in range(args.perms):
        w = None
        for _try in range(64):
            cand = w_c.copy()
            cand[real] = w_c[real[rng.permutation(len(real))]]
            if not np.allclose(cand, w_c):
                w = cand
                break
        if w is None:
            degen += 1
            continue
        pd.append(miou(ev(w, tau_c)) - m_pub)
    if not pd:
        raise SystemExit('every permutation of w was a no-op -- the fitted scales '
                         'are all equal, so there is nothing for this control to '
                         'test. Report the arms without it.')
    if degen:
        print(f'  note: {degen} draws discarded as no-ops (equal fitted scales)')
    pd = np.array(pd)
    beat = float((pd >= d_c).mean())
    print(f'  permutation control ({args.perms}): mean {pd.mean():+.2f}, '
          f'p95 {np.percentile(pd, 95):+.2f}, max {pd.max():+.2f}; '
          f'real {d_c:+.2f}, {100 * beat:.1f}% match or beat it')

    # ---- report --------------------------------------------------------------
    pcA, pcB, pcC = (per_class_iou(C) for C in (rows[0][1], rows[1][1], rows[2][1]))
    md = ['# Lever 2 across splits — fit (w, τ) on one, evaluate exactly on another\n',
          f'- source (fit): `{args.src_cache}` | **{len(src)}** tiles',
          f'- destination (evaluate): `{args.dst_cache}` | **{len(dst)}** tiles, '
          f'**every pixel**, no subsample',
          f'- published τ **{args.tau}** | objective **`{args.objective}`** | '
          f'catch-all `{LD.names[bg]}`\n',
          'The splits share no filenames. The rule is `pred = argmax(w·s)` then keep '
          'if the **raw** score clears τ — verified line-by-line against '
          '`segearthov3_segmentor.py:496-508`, and evaluated with the same `hist_at` '
          'the fit uses, so the two cannot drift.\n']
    if args.expect is not None:
        md.append(f'✅ **Gate:** published-τ reproduces **{m_pub:.2f}** against an '
                  f'expected **{args.expect:.2f}**.\n')
    md += ['## Arms\n', '| arm | mIoU | Δ vs published |', '|---|---|---|']
    for name, C in rows:
        md.append(f'| {name} | **{miou(C):.2f}** | **{miou(C) - m_pub:+.2f}** |')
    md += [f'\n⭐ **Lever 2 over lever 1, on held-out tiles: {d_c - d_b:+.2f} mIoU.**\n',
           '⚠️ **Arm D fits on the evaluation labels.** It bounds what any transfer '
           'could reach; it is not a method.\n',
           '## Fitted on the source\n',
           '| class | w | τ | IoU published | IoU lever 1 | IoU lever 2 | Δ (C−B) |',
           '|---|---|---|---|---|---|---|']
    for c in range(nc):
        tag = ' *(catch-all)*' if c == bg else ''
        md.append(f'| {LD.names[c]}{tag} | **{w_c[c]:.3f}** | '
                  f'{"—" if c == bg else f"{tau_c[c]:.3f}"} | {pcA[c]:.2f} | '
                  f'{pcB[c]:.2f} | {pcC[c]:.2f} | **{pcC[c] - pcB[c]:+.2f}** |')
    md += ['\n## Permutation control — does the argmax REORDERING transfer?\n',
           f'Arm C\'s `w` shuffled among the {len(real)} real classes with τ held at '
           f'arm C\'s values, {args.perms} draws. This isolates the reordering from '
           f'the fact that some classes were scaled at all.\n',
           '| | Δ vs published |', '|---|---|',
           f'| **real assignment (arm C)** | **{d_c:+.2f}** |',
           f'| shuffled, mean | {pd.mean():+.2f} |',
           f'| shuffled, p95 | {np.percentile(pd, 95):+.2f} |',
           f'| shuffled, max | {pd.max():+.2f} |',
           f'| matching or beating the real one | **{100 * beat:.1f}%** |\n',
           '## Verdict\n']
    if d_c - d_b > 0 and beat < 0.05:
        md.append(f'✅ **Lever 2 transfers: {d_c - d_b:+.2f} mIoU over lever 1** on tiles '
                  f'the fit never saw, against a destination bound of '
                  f'{d_d - d_b:+.2f}. Only {100 * beat:.1f}% of `w` shuffles match it, '
                  f'so what transfers is the per-class assignment.')
    elif d_c - d_b > 0:
        md.append(f'⚠️ **Positive ({d_c - d_b:+.2f}) but the permutation control is not '
                  f'beaten** — {100 * beat:.1f}% of shuffles match or exceed it. Consistent '
                  f'with scaling *some* classes helping, not with this particular '
                  f'assignment carrying across.')
    else:
        md.append(f'⛔ **Lever 2 does NOT transfer: {d_c - d_b:+.2f} over lever 1.** The '
                  f'scale is fitted to the source split and does not survive the move.')
    md.append('\n⚠️ Quote the per-class table, never the mean alone.')

    if args.deploy_cfg:
        if not args.base_cfg:
            raise SystemExit('--deploy-cfg needs --base-cfg')
        from mmengine import Config
        cfg = Config.fromfile(str(Path(args.base_cfg).expanduser()))
        pv = [round(float(t), 4) for t in tau_c]
        pv[bg] = float(args.tau)                 # bg's τ has no effect
        wv = [round(float(x), 4) for x in w_c]
        if len(pv) != nc or len(wv) != nc:
            raise SystemExit('vector length does not match the class count')
        if min(wv) <= 0:
            raise SystemExit(f'non-positive scale would delete a class: {wv}')
        cfg.model.prob_thd, cfg.model.class_scale = pv, wv
        out = Path(args.deploy_cfg).expanduser()
        cfg.dump(str(out))
        print(f'\n  deploy config -> {out}')
        for i in range(nc):
            print(f'    {LD.names[i]:<12} w={wv[i]:.3f}  τ={pv[i]:.3f}')
        print(f'  expect eval.py to report mIoU {miou(rows[2][1]):.2f} '
              f'(published-τ gives {m_pub:.2f})')
        md.append(f'\n## Deployed\n\n`{out}`, written from the fit rather than '
                  f'transcribed. `eval.py` should report **{miou(rows[2][1]):.2f}** '
                  f'against the published-τ **{m_pub:.2f}**.\n')

    text = '\n'.join(md)
    print('\n' + text)
    if args.md:
        Path(args.md).expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).expanduser().write_text(text)
        print(f'\nwritten: {args.md}')


if __name__ == '__main__':
    main()
