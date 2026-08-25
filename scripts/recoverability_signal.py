"""
Week 3 — is there ANY signal that says "this background pixel is really something"?

WHY THIS IS NOW THE WHOLE PROBLEM
---------------------------------
selective_recovery_miou.py, run both ways at the same operating point:

    oracle scope   166,050,845 recovered, 47.2% precision -> mIoU 50.84  (+3.47)
    honest scope   568,900,398 recovered, 11.2% precision -> mIoU 32.00 (-15.37)

The honest run touches 3.4x more pixels and gets FEWER right. The 402,849,553
extra pixels are true background being destroyed. Same code, same matrix, same
neighbour vote -- the ONLY difference is that the oracle is told which
background-assigned pixels are worth touching.

So the labelling problem is effectively solved: given the right pixels, a plain
neighbour vote is worth +3.47 mIoU. The DETECTION problem is untouched, and it is
where all the difficulty is. Margin and purity cannot help -- they describe the
neighbourhood, and a true-background region has confident neighbours too.

ROADMAP Week 6 specified exactly the missing piece and it was never built:

    identified   = P_final >  tau_high
    unidentified = tau_low < P_final <= tau_high     <- the target
    ignored      = P_final <= tau_low                <- true background

This script asks, before any more machinery is written, whether that tau_low
separation exists in the data at all.

THE TEST. Over every pixel the baseline assigned to background, label it
    positive = GT says a real class   (recoverable, 323,084,415 px)
    negative = GT says background     (must not be touched)
and score each cached signal by AUC. 0.50 is coin-flip. Below ~0.65 the signal
cannot support a detection rule and the appearance term (GPU) becomes the only
remaining candidate -- which would invert ANALYSIS 3.3's prediction that the
embedding term would be the weak one.

Signals tested, all already in the .npz cache, all free:
    conf        max P_final. The tau_low candidate.
    conf2       runner-up class score.
    gap         conf - conf2. Ambiguity between the top two classes.
    spres_max   highest presence score over real classes for the tile.
    spres_arg   presence score of the argmax class.

Histogram-based, so it streams 1.7 billion pixels without holding them.

    python scripts/recoverability_signal.py \
        --cache ~/outputs/week2_tau0.5_instrumented/cache --tau 0.5 \
        --md ~/outputs/week3/recoverability_signal.md
"""
import argparse
import warnings
from pathlib import Path

import numpy as np

NB = 512                       # histogram bins over [0,1]
CLASSES = ['background', 'building', 'road', 'water',
           'barren', 'forest', 'agricultural']


def auc_from_hist(pos, neg):
    """AUC of a signal from its positive/negative histograms.

    Trapezoidal over the ROC traced by sweeping the threshold from high to low.
    Equivalent to the rank statistic, ties handled by the trapezoid.
    """
    p = pos[::-1].astype(float); n = neg[::-1].astype(float)
    P, N = p.sum(), n.sum()
    if P == 0 or N == 0:
        return float('nan')
    tpr = np.concatenate([[0.0], np.cumsum(p) / P])
    fpr = np.concatenate([[0.0], np.cumsum(n) / N])
    return float(np.trapz(tpr, fpr))


def best_operating(pos, neg, edges):
    """Threshold maximising recoverable-precision x recall, sweeping a LOWER
    bound (keep everything >= t), which is what tau_low is."""
    p = pos.astype(float); n = neg.astype(float)
    kp = np.cumsum(p[::-1])[::-1]           # positives kept at threshold >= bin
    kn = np.cumsum(n[::-1])[::-1]
    prec = kp / np.maximum(kp + kn, 1)
    rec = kp / max(p.sum(), 1)
    f = np.where((prec + rec) > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-9), 0)
    i = int(np.argmax(f))
    return edges[i], prec[i], rec[i], kp[i], kn[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--md', default=None)
    args = ap.parse_args()
    warnings.filterwarnings('ignore', message='All-NaN slice encountered')

    files = sorted(Path(args.cache).expanduser().glob('*.npz'))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f'no .npz under {args.cache}')

    names = ['conf', 'conf2', 'gap', 'spres_max', 'spres_arg']
    pos = {k: np.zeros(NB, np.int64) for k in names}
    neg = {k: np.zeros(NB, np.int64) for k in names}
    npos = nneg = 0
    edges = np.linspace(0, 1, NB + 1)[:-1]

    print(f'{len(files)} tiles | τ = {args.tau}\n')
    for i, f in enumerate(files):
        z = np.load(f)
        gt = z['gt'].astype(np.uint8)
        conf = z['conf'].astype(np.float32)
        conf2 = z['conf2'].astype(np.float32)
        pred = z['pred'].astype(np.int16)

        assigned_bg = (gt > 0) & ((conf < args.tau) | (pred == 0))
        if not assigned_bg.any():
            continue
        P = assigned_bg & (gt >= 2)          # recoverable
        N = assigned_bg & (gt == 1)          # true background
        npos += int(P.sum()); nneg += int(N.sum())

        sp = z['spres']                       # (n_views, 7)
        with np.errstate(all='ignore'):
            spc = np.nanmax(sp, axis=0) if sp.size else np.full(7, np.nan)
        spmax = np.nanmax(spc[1:]) if np.isfinite(spc[1:]).any() else 0.0
        sparg_lut = np.nan_to_num(spc, nan=0.0)

        sig = {'conf': conf, 'conf2': conf2, 'gap': conf - conf2,
               'spres_max': np.full(gt.shape, float(spmax), np.float32),
               'spres_arg': sparg_lut[np.clip(pred, 0, 6)].astype(np.float32)}

        for k, v in sig.items():
            b = np.clip((v * NB).astype(np.int32), 0, NB - 1)
            pos[k] += np.bincount(b[P], minlength=NB)
            neg[k] += np.bincount(b[N], minlength=NB)

        if (i + 1) % 250 == 0 or i + 1 == len(files):
            print(f'  {i + 1}/{len(files)}')

    base = npos / max(npos + nneg, 1)
    md = ['# Week 3 — can we detect which background pixels are recoverable?\n',
          f'- tiles: **{len(files)}**  |  τ: **{args.tau}**',
          f'- background-assigned pixels: **{npos + nneg:,}**',
          f'- of those, really a real class (**positive**): **{npos:,}**',
          f'- really background (**negative**): **{nneg:,}**',
          f'- **base rate: {100 * base:.1f}%** — a rule that fires everywhere scores '
          'exactly this precision\n',
          '## Signal quality\n',
          '| signal | AUC | best lower bound | precision | recall | kept px |',
          '|---|---|---|---|---|---|']

    res = {}
    for k in names:
        a = auc_from_hist(pos[k], neg[k])
        t, pr, rc, kp, kn = best_operating(pos[k], neg[k], edges)
        res[k] = (a, t, pr, rc, kp)
        md.append(f'| `{k}` | **{a:.3f}** | {t:.3f} | {100 * pr:.1f}% | '
                  f'{100 * rc:.1f}% | {kp + kn:,.0f} |')

    md.append('\nAUC 0.50 is a coin flip. Precision must clearly exceed the '
              f'**{100 * base:.1f}%** base rate for the signal to be worth anything — '
              'a rule that fires everywhere already achieves that.\n')

    best = max(names, key=lambda k: res[k][0] if np.isfinite(res[k][0]) else 0)
    a, t, pr, rc, kp = res[best]
    md += ['## Verdict\n']
    if a >= 0.70:
        md.append(f'✅ **`{best}` separates them (AUC {a:.3f}).** A lower bound at '
                  f'**{t:.3f}** keeps {100 * rc:.0f}% of the recoverable pixels at '
                  f'{100 * pr:.1f}% precision against a {100 * base:.1f}% base rate. '
                  'This is ROADMAP Week 6\'s missing `τ_low`, and it is measured '
                  'rather than hand-tuned. Wire it into the recovery rule as the '
                  '**detection** stage, with the neighbour vote as the **labelling** '
                  'stage, and re-run `selective_recovery_miou.py`.')
    elif a >= 0.60:
        md.append(f'⚠️ **`{best}` carries weak signal (AUC {a:.3f}).** Enough to be '
                  'worth combining with the neighbourhood terms, not enough alone. '
                  'Try it as a gate anyway — the mIoU sweep is the real arbiter, and '
                  'even a weak detector applied to a +3.47 oracle headroom may clear '
                  'the baseline. If it does not, the appearance term is next.')
    else:
        md.append(f'⛔ **No cached signal detects recoverability** (best `{best}`, AUC '
                  f'{a:.3f}). The confidence map cannot tell a suppressed real class '
                  'from genuine background, so `τ_low` does not exist in this data and '
                  'ROADMAP Week 6\'s three-way split is not realisable from `P_final` '
                  'alone.\n\n**This inverts `ANALYSIS §3.3`.** That section predicted '
                  'the embedding term would be the weak one, because a region is '
                  'ambiguous precisely where SAM 3 is unsure. Instead co-occurrence is '
                  'weak (+0.2 over a neighbour vote) and confidence is blind, which '
                  'makes **appearance the only remaining candidate** — and the only '
                  'one that needs a GPU. Test it before re-scoping: pool `F_cond` per '
                  'region, fit real-class prototypes from confident regions, score '
                  'background-assigned regions by similarity, and re-run this AUC.')

    md.append('\n> The two stages are separable and should be reported separately. '
              'Labelling is solved — the oracle run shows a plain neighbour vote is '
              'worth **+3.47 mIoU** given the right pixels. Everything now rides on '
              'detection.')

    text = '\n'.join(md)
    print('\n' + text)
    if args.md:
        p = Path(args.md).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f'\nwritten: {p}')


if __name__ == '__main__':
    main()
