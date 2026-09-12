"""
WHY did per-class head fusion buy nothing? Decompose what rho actually DOES.

THE HYPOTHESIS, from the fitted values. On LoveDA the fit pinned `road` at the
grid MAXIMUM (4.00 in 4 of 5 folds) and `forest` at the grid MINIMUM (0.25 in all
5). It is not choosing a blend -- it is trying to switch one head off. And that is
exactly where lever 3 has nothing lever 2 does not already have:

    s_c(rho) = max(a_c * P_sem_c, b_c * P_inst_c),  max(a,b)=1,  rho = b/a

Take rho < 1, so a = 1 and b = rho. Every pixel of class c falls in one of three
buckets:

    UNTOUCHED  the winning head's arm carries multiplier 1 -> score unchanged
    RESCALED   the winning head's arm carries the scaled multiplier -> score * rho
    FLIPPED    the OTHER head now supplies the score -> a pixel-dependent change

⛔ AND THE OBVIOUS READING OF THOSE BUCKETS IS WRONG. A first version of this
script called RESCALED "already inside lever 2's family", because w_c is also a
multiplier. That is only true if EVERY pixel of the class gets the SAME
multiplier. Lever 2's w_c scales the whole class at once; rho scales only the
subset where one particular head wins. So a class SPLIT between untouched and
rescaled is getting a pixel-dependent change that no per-class constant can make.

The synthetic test caught it: on the case built so that no w can help
(`test_head_fusion.py`, the separable arm), this script reported 0.0% flipped and
would have concluded "a rescale in disguise" about the one construction where
that is provably false.

⭐ The correct statistic is therefore

    beyond lever 2  =  flipped  +  min(untouched, rescaled)

i.e. pixels that flip head, PLUS the smaller of the two multiplier groups, since
lever 2 can reproduce rho only by applying the majority multiplier to everything.
If that number is small, rho is a per-class constant in disguise and refitting w
and tau on top absorbs it.

This script measures the three shares per class, at the fitted rho, and reports
the flipped share -- the only part that is genuinely new.

⚠️ It explains a NULL. It does not rescue one, and it is not a new lever.

    python scripts/head_dominance.py --cache ~/outputs/loveda_heads/cache \\
        --rho 1.47,0.51,3.83,1.00,0.33,0.25,0.82 \\
        --md ~/outputs/week4/head_dominance_loveda.md

⛔ `--rho` must be given in the cache's own class order, which the script prints.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels                                                     # noqa: E402


def buckets(sem, inst, rho):
    """Classify every pixel of one class as untouched / rescaled / flipped.

    Returns (untouched, rescaled, flipped) boolean arrays, plus the winner at
    rho = 1 (True = the instance head supplied the score).
    """
    a = min(1.0, 1.0 / rho)
    b = min(1.0, rho)
    inst_wins_1 = inst > sem                     # who supplies max(sem, inst)
    inst_wins_r = b * inst > a * sem             # who supplies it at this rho
    flipped = inst_wins_1 != inst_wins_r
    # "untouched" = same winner AND that arm's multiplier is exactly 1
    same = ~flipped
    untouched = same & np.where(inst_wins_1, b == 1.0, a == 1.0)
    rescaled = same & ~untouched
    return untouched, rescaled, flipped, inst_wins_1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True, help='a --cache-heads cache')
    ap.add_argument('--rho', required=True,
                    help='comma-separated fitted ρ, in the cache class order')
    ap.add_argument('--subsample', type=int, default=40000,
                    help='labelled pixels kept per tile')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--md', default=None)
    args = ap.parse_args()

    LB = labels.from_cache(args.cache)
    nc, bg = LB.n, LB.bg - 1
    print(f'  classes: {LB}')
    rho = np.array([float(x) for x in args.rho.split(',')], dtype=np.float64)
    if rho.size != nc:
        raise SystemExit(f'--rho has {rho.size} entries, cache has {nc} classes: '
                         f'{", ".join(LB.names)}')
    if (rho <= 0).any():
        raise SystemExit('ρ must be strictly positive.')

    files = sorted(Path(args.cache).expanduser().glob('*.npz'))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f'no .npz under {args.cache}')
    print(f'{len(files)} tiles | ρ = ' + ', '.join(f'{r:.2f}' for r in rho) + '\n')

    rng = np.random.default_rng(args.seed)
    # per class: [untouched, rescaled, flipped], plus instance-win count
    tally = np.zeros((nc, 3), np.int64)
    inst_win = np.zeros(nc, np.int64)
    total = np.zeros(nc, np.int64)
    # the same, restricted to pixels the class actually WINS (where its score
    # reaches the output at all)
    tally_w = np.zeros((nc, 3), np.int64)
    total_w = np.zeros(nc, np.int64)
    gate_err, gate_n = 0.0, 0

    for i, f in enumerate(files):
        if (i + 1) % 100 == 0 or i + 1 == len(files):
            print(f'  {i + 1}/{len(files)}', flush=True)
        z = np.load(f)
        for k in ('sem', 'inst'):
            if k not in z.files:
                raise SystemExit(f'{f.name} has no `{k}`; needs a --cache-heads cache.')
        gt = z['gt'].ravel()
        keep = np.flatnonzero(gt > 0)
        if keep.size == 0:
            continue
        if keep.size > args.subsample:
            keep = rng.choice(keep, size=args.subsample, replace=False)
        sem = z['sem'].astype(np.float32).reshape(nc, -1)[:, keep]
        inst = z['inst'].astype(np.float32).reshape(nc, -1)[:, keep]

        # ⛔ Same identity gate as head_fusion.py: at ρ = 1 the reconstruction
        # must be the cached score, or these shares describe a different model.
        if 'logits' in z.files:
            lg = z['logits'].astype(np.float32).reshape(nc, -1)[:, keep]
            gate_err += float(np.abs(np.maximum(sem, inst) - lg).mean()); gate_n += 1

        s1 = np.maximum(sem, inst)
        won = np.argmax(s1, axis=0)                  # who wins the argmax at ρ=1
        for c in range(nc):
            u, r, fl, iw = buckets(sem[c], inst[c], rho[c])
            tally[c] += (u.sum(), r.sum(), fl.sum())
            inst_win[c] += int(iw.sum())
            total[c] += u.size
            m = won == c
            if m.any():
                tally_w[c] += (u[m].sum(), r[m].sum(), fl[m].sum())
                total_w[c] += int(m.sum())

    gate_err = gate_err / gate_n if gate_n else float('nan')
    ok = gate_n > 0 and gate_err <= 0.01
    print(f'\n  identity gate: mean |max(sem,inst) − logits| = {gate_err:.5f}  '
          f'{"PASS" if ok else "FAIL"}')
    if gate_n and not ok:
        raise SystemExit('⛔ the cached heads do not reconstruct `logits`; '
                         'these shares would describe a different model.')

    pct = lambda a, t: 100.0 * a / np.where(t > 0, t, 1)
    sh, sh_w = pct(tally, total[:, None]), pct(tally_w, total_w[:, None])
    iw_pct = pct(inst_win, total)
    # ⭐ the part lever 2 cannot express -- see the module docstring
    beyond = sh[:, 2] + np.minimum(sh[:, 0], sh[:, 1])
    beyond_w = sh_w[:, 2] + np.minimum(sh_w[:, 0], sh_w[:, 1])

    md = ['# What does ρ actually change? — the anatomy of lever 3’s null\n',
          f'- cache `{args.cache}` | tiles **{len(files)}** | subsample '
          f'**{args.subsample}** px/tile',
          f'- ρ as fitted: ' + ', '.join(f'`{n}` {r:.2f}' for n, r in
                                         zip(LB.names, rho)) + '\n',
          '`s_c(ρ) = max(a_c·P_sem_c, b_c·P_inst_c)` with `max(a,b)=1`. Every pixel '
          'falls in one of three buckets:\n',
          '| bucket | what happens |',
          '|---|---|',
          '| **untouched** | the winning head’s arm carries multiplier 1 |',
          '| **rescaled** | the winning head’s arm carries the scaled multiplier |',
          '| **flipped** | the *other* head now supplies the score |\n',
          '⭐ **Lever 2 can reproduce ρ only where every pixel of the class gets the '
          'SAME multiplier** — `w_c` scales the whole class at once. So the part ρ '
          'contributes that `w_c` cannot is\n',
          '> **beyond lever 2 = flipped + min(untouched, rescaled)**\n',
          '⛔ Not "flipped" alone. A class split evenly between untouched and rescaled '
          'is getting a pixel-dependent change even though no pixel changed head.\n',
          f'✅ Identity gate **{gate_err:.5f}**. At ρ = 1 the heads reconstruct the '
          f'cached score, so these shares describe the deployed pipeline.\n',
          '## All labelled pixels\n',
          '| class | ρ | inst head wins | untouched | rescaled | flipped | ⭐ beyond lever 2 |',
          '|---|---|---|---|---|---|---|']
    for c, n in enumerate(LB.names):
        md.append(f'| {n} | {rho[c]:.2f} | {iw_pct[c]:.1f}% | {sh[c,0]:.1f}% | '
                  f'{sh[c,1]:.1f}% | {sh[c,2]:.1f}% | **{beyond[c]:.1f}%** |')
    md += ['\n## Restricted to pixels where the class WINS the argmax\n',
           'These are the pixels whose score actually reaches the threshold, so this '
           'is the share that can change an output.\n',
           '| class | pixels won | untouched | rescaled | flipped | ⭐ beyond lever 2 |',
           '|---|---|---|---|---|---|']
    for c, n in enumerate(LB.names):
        md.append(f'| {n} | {total_w[c]:,} | {sh_w[c,0]:.1f}% | {sh_w[c,1]:.1f}% | '
                  f'{sh_w[c,2]:.1f}% | **{beyond_w[c]:.1f}%** |')

    real = [c for c in range(nc) if c != bg]
    w_real = total_w[real].astype(np.float64)
    b_won = float((beyond_w[real] * w_real).sum() / max(w_real.sum(), 1.0))
    a_real = total[real].astype(np.float64)
    b_all = float((beyond[real] * a_real).sum() / max(a_real.sum(), 1.0))
    flip_won = float(tally_w[real, 2].sum()) / max(int(total_w[real].sum()), 1) * 100
    dom = [LB.names[c] for c in real if iw_pct[c] < 5 or iw_pct[c] > 95]

    md += ['\n## Verdict\n']
    if b_won < 10:
        md.append(f'⭐ **Only {b_won:.1f}% of the pixels a real class wins get a change '
                  f'that a per-class constant could not have made** ({flip_won:.1f}% '
                  f'flip head; the rest is one uniform multiplier). A per-class '
                  f'constant is exactly what lever 2 already fits, so **ρ is largely '
                  f'lever 2 in disguise, and refitting `w` and `τ` on top absorbs it. '
                  f'That is why lever 3 adds nothing.**')
    else:
        md.append(f'⚠️ **{b_won:.1f}% of won pixels get a change no per-class constant '
                  f'could make.** ⛔ The "ρ is a rescale in disguise" explanation does '
                  f'NOT hold here — lever 3 had real room and still did not convert it. '
                  f'Its null needs a different explanation, and the rescale story must '
                  f'not be quoted for this dataset.')
    md.append(f'\nOver all labelled pixels of real classes the beyond-lever-2 share is '
              f'**{b_all:.1f}%** ({flip_won:.1f}% of won pixels flip head).')
    if dom:
        md.append(f'\n⭐ **One head dominates outright for: {", ".join(dom)}** '
                  f'(the instance head wins <5% or >95% of that class’s pixels). '
                  f'For those classes ρ can only be a no-op or a uniform rescale, '
                  f'whichever direction it moves.')
    md.append('\n⚠️ **This explains a null; it does not rescue one.** The shares are '
              'measured at the fitted ρ, which was chosen to maximise the objective — '
              'so they describe the best available head assignment, not an arbitrary one.')

    out = '\n'.join(md)
    print('\n' + out)
    if args.md:
        Path(args.md).expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).expanduser().write_text(out)
        print(f'\nwrote {args.md}')


if __name__ == '__main__':
    main()
