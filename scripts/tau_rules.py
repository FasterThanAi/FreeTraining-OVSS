"""
Label-free per-class thresholds. How much of the oracle bound can we actually reach?

WHERE THIS COMES FROM. tau_oracle.py showed that per-class thresholds are worth
+1.46 mIoU on LoveDA, and unlike every other result in this project the gain is
real: the six land-cover classes gain 8.63 IoU in aggregate, water alone +6.70.
The chosen thresholds span 0.170 to 0.595, so one global value is wrong for
different classes in OPPOSITE directions. That bound was measured with ground
truth. This script asks what a rule without ground truth can get.

THE COMPARISON IS FAIR, AND THAT MATTERS. SegEarth-OV3 tunes tau per dataset
with labels -- 0.5 for LoveDA, 0.1 for OpenEarthMap. So a rule that spends ONE
label-tuned parameter and spreads it across classes by a fixed principle is at
parity with the baseline, not cheating. What would be cheating is the oracle's N
independent parameters. Each rule below has exactly one knob, swept, with the
whole curve reported so the choice is visible rather than buried.

THE RULES

  equal-commitment   tau_c is the q-th percentile of class c's own confidence
                     distribution, so the same FRACTION of each class's argmax
                     pixels survives. A class the model is broadly unsure about
                     stops being punished for it. One knob: q.

  presence-scaled    P_final = P_fused * S_pres, so thresholding P_final at a
                     constant is thresholding P_fused at tau / S_pres -- a
                     different bar for every class, set by a quantity that has
                     nothing to do with how confident the dense heads are.
                     Setting tau_c proportional to S_pres_c undoes exactly that,
                     thresholding the ungated score uniformly. One knob: the
                     overall level. This is the principled one.

  per-class Otsu     the classical two-mode split of each class's confidence
                     histogram. No knob at all, which makes it the honest
                     zero-parameter reference.

  global (reference) one threshold for everything -- what the baseline does.

Everything runs off the same (gt, pred, conf-bin) histogram as tau_oracle.py, so
it is CPU-only and every candidate is evaluated in microseconds.

    python scripts/tau_rules.py --cache ~/outputs/week3_fused/cache --tau 0.5
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels                                    # noqa: E402
from tau_oracle import confusion_at, miou, per_class_iou, NBINS   # noqa: E402


def build(files, nc, nbins):
    """(gt, pred, conf-bin) histogram plus the per-class presence medians."""
    H = np.zeros((nc, nc, nbins), np.int64)
    spres = [[] for _ in range(nc)]
    for i, f in enumerate(files):
        z = np.load(f)
        gt = z['gt'].astype(np.int32)
        conf = z['conf'].astype(np.float32)
        pred = z['pred'].astype(np.int32)
        m = gt > 0
        if m.any():
            b = np.clip((conf[m] * nbins).astype(np.int32), 0, nbins - 1)
            np.add.at(H, (gt[m] - 1, np.clip(pred[m], 0, nc - 1), b), 1)
        sp = z['spres'] if 'spres' in z.files else None
        if sp is not None and sp.size:
            with np.errstate(all='ignore'):
                per = np.nanmax(sp, axis=0)      # max over sliding-window views
            for c in range(min(nc, per.shape[0])):
                if np.isfinite(per[c]):
                    spres[c].append(float(per[c]))
        if (i + 1) % 250 == 0 or i + 1 == len(files):
            print(f'  {i + 1}/{len(files)}')
    med = np.array([np.median(v) if v else np.nan for v in spres])
    return H, med


def percentile_taus(Hp, q, nbins):
    """tau_c at the q-th percentile of class c's own confidence distribution."""
    out = np.zeros(Hp.shape[0])
    for c in range(Hp.shape[0]):
        h = Hp[c]
        tot = h.sum()
        if tot == 0:
            continue
        cum = np.cumsum(h) / tot
        out[c] = np.searchsorted(cum, q) / nbins
    return out


def otsu_taus(Hp, nbins):
    """Classical between-class-variance split, per class. Zero free parameters."""
    out = np.zeros(Hp.shape[0])
    idx = np.arange(nbins)
    for c in range(Hp.shape[0]):
        h = Hp[c].astype(float)
        if h.sum() == 0:
            continue
        w0 = np.cumsum(h)
        w1 = h.sum() - w0
        mu = np.cumsum(h * idx)
        mut = mu[-1]
        with np.errstate(invalid='ignore', divide='ignore'):
            m0 = mu / w0
            m1 = (mut - mu) / w1
            var = w0 * w1 * (m0 - m1) ** 2
        var[(w0 == 0) | (w1 == 0)] = -1
        # Between-class variance is FLAT across every split that separates two
        # well-spaced modes, so argmax returns the leftmost tie -- a threshold
        # below both modes, which keeps everything and defeats the point. Take
        # the middle of the plateau, which is the split Otsu is meant to give.
        tie = np.flatnonzero(var >= var.max() - 1e-12)
        out[c] = int(tie[len(tie) // 2]) / nbins
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--tau', type=float, required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--md', default=None)
    args = ap.parse_args()
    warnings.filterwarnings('ignore')

    LB = labels.from_cache(args.cache)
    nc, bg = LB.n, LB.bg - 1
    print(f'  classes: {LB}')

    files = sorted(Path(args.cache).expanduser().glob('*.npz'))
    if args.limit:
        files = files[:args.limit]
    print(f'{len(files)} tiles | published τ = {args.tau}\n')
    H, spres_med = build(files, nc, NBINS)

    Hp = H.sum(axis=0)                 # (pred, bin) -- each class's own conf dist
    grid = np.arange(NBINS + 1) / NBINS

    def ev(t):
        t = np.asarray(t, float).copy()
        t[bg] = 0.0                    # tau[bg] cannot change any assignment
        return miou(confusion_at(H, t, bg, NBINS)), t

    m_base, _ = ev(np.full(nc, args.tau))
    glob = [(ev(np.full(nc, t))[0], t) for t in grid]
    m_glob, t_glob = max(glob)

    # oracle per-class, same coordinate ascent as tau_oracle.py, for the ceiling
    taus_o = np.full(nc, t_glob)
    best_o = m_glob
    for _ in range(6):
        moved = False
        for c in range(nc):
            if c == bg:
                continue
            cand = [(ev(np.where(np.arange(nc) == c, t, taus_o))[0], t) for t in grid]
            sc, st = max(cand)
            if sc > best_o + 1e-9:
                best_o, taus_o[c], moved = sc, st, True
        if not moved:
            break

    # ---- the label-free rules, one knob each, swept
    qs = np.arange(0.0, 0.71, 0.01)
    pct = [(ev(percentile_taus(Hp, q, NBINS))[0], q) for q in qs]
    m_pct, q_pct = max(pct)
    t_pct = ev(percentile_taus(Hp, q_pct, NBINS))[1]

    if np.isfinite(spres_med).any():
        sp = np.nan_to_num(spres_med, nan=np.nanmedian(spres_med))
        sp = np.clip(sp, 1e-3, None)
        base = np.nanmean(sp[[c for c in range(nc) if c != bg]])
        pres = [(ev(np.clip(lvl * sp / base, 0, 1))[0], lvl) for lvl in grid]
        m_pres, lvl_pres = max(pres)
        t_pres = ev(np.clip(lvl_pres * sp / base, 0, 1))[1]
    else:
        m_pres, lvl_pres, t_pres = float('nan'), float('nan'), np.full(nc, np.nan)

    t_otsu = otsu_taus(Hp, NBINS)
    m_otsu, t_otsu = ev(t_otsu)

    head = best_o - m_base                       # the oracle headroom
    def frac(m):
        """Share of the oracle headroom captured, as a string. When the oracle
        finds nothing there is no headroom to divide by, and printing nan% would
        read as a bug rather than as 'the bound is zero'."""
        if head <= 1e-9:
            return 'n/a — oracle headroom is 0.00'
        return f'{(m - m_base) / head * 100:.0f}%'

    md = ['# Label-free per-class thresholds\n',
          f'- cache: `{args.cache}`  |  tiles: **{len(files)}**  |  classes: **{nc}**',
          f'- published τ: **{args.tau}** → **{m_base:.2f}** mIoU\n',
          'SegEarth-OV3 tunes one τ per dataset with labels, so a rule that spends **one** '
          'label-tuned knob and distributes it across classes by a fixed principle is at '
          'parity with the baseline. The oracle row spends **N** independent parameters '
          'and is a ceiling, not a competitor.\n',
          '| rule | knobs | **mIoU** | Δ | share of oracle headroom |',
          '|---|---|---|---|---|',
          f'| published τ = {args.tau} | 1 (baseline) | **{m_base:.2f}** | — | — |',
          f'| best global τ = {t_glob:.3f} | 1 | **{m_glob:.2f}** | {m_glob - m_base:+.2f} | {frac(m_glob)} |',
          f'| per-class Otsu | **0** | **{m_otsu:.2f}** | {m_otsu - m_base:+.2f} | {frac(m_otsu)} |',
          f'| equal-commitment, q = {q_pct:.2f} | 1 | **{m_pct:.2f}** | {m_pct - m_base:+.2f} | **{frac(m_pct)}** |',
          (f'| presence-scaled, level = {lvl_pres:.3f} | 1 | **{m_pres:.2f}** | '
           f'{m_pres - m_base:+.2f} | **{frac(m_pres)}** |'
           if np.isfinite(m_pres) else '| presence-scaled | 1 | — | — | — |'),
          f'| **ORACLE per-class** | {nc - 1} | **{best_o:.2f}** | **{best_o - m_base:+.2f}** | 100% |']

    best_rule = max([(m_pct, 'equal-commitment', t_pct),
                     (m_pres if np.isfinite(m_pres) else -1, 'presence-scaled', t_pres),
                     (m_otsu, 'per-class Otsu', t_otsu)])
    mb, nameb, tb = best_rule

    md += [f'\n## Thresholds chosen — {nameb} vs the oracle\n',
           '| class | label-free τ | oracle τ | IoU published | label-free | Δ |',
           '|---|---|---|---|---|---|']
    v0 = per_class_iou(confusion_at(H, np.full(nc, args.tau), bg, NBINS))
    v1 = per_class_iou(confusion_at(H, tb, bg, NBINS))
    present = H.sum(axis=(1, 2))
    for k in range(nc):
        if present[k] == 0:
            md.append(f'| {LB.names[k]} | — | — | *absent* | | |')
            continue
        lf = '—' if k == bg else f'{tb[k]:.3f}'
        orc = '—' if k == bg else f'{taus_o[k]:.3f}'
        md.append(f'| {LB.names[k]} | {lf} | {orc} | {v0[k]:.2f} | {v1[k]:.2f} | '
                  f'**{v1[k] - v0[k]:+.2f}** |')

    d = v1 - v0
    realk = [k for k in range(nc) if k != bg and present[k] > 0]
    bgd = d[bg] if present[bg] > 0 and np.isfinite(d[bg]) else 0.0
    reald = float(np.nansum(d[realk]))
    md.append(f'\n`background` **{bgd:+.2f}**, the {len(realk)} real classes '
              f'**{reald:+.2f}** in aggregate.\n')

    md += ['## Verdict\n']
    gain = mb - m_base
    if gain < 0.25:
        md.append(f'⛔ **No label-free rule reaches the bound.** Best is `{nameb}` at '
                  f'{gain:+.2f} mIoU ({frac(mb)} of the oracle\'s {head:+.2f}). '
                  'Per-class thresholds help only when the thresholds are chosen with '
                  'labels, so this is a property of the evaluation rather than a method. '
                  'Report the oracle bound as a limitation of global thresholding and '
                  'stop here.')
    elif reald <= 0:
        md.append(f'⚠️ **`{nameb}` gains {gain:+.2f} mIoU, but the real classes lose '
                  f'{reald:+.2f}** — the same background-unwinding pattern as the '
                  'recovery experiments. A real mIoU gain that is not better land-cover '
                  'classification. Do not report it without this table.')
    else:
        md.append(f'✅ **`{nameb}` reaches {frac(mb)} of the oracle bound '
                  f'({gain:+.2f} mIoU) with one knob — the same budget the baseline '
                  f'already spends on τ — and the real classes gain {reald:+.2f} in '
                  'aggregate. This is a deployable rule, not a bound: nothing in it '
                  'reads a label. Re-run it on the second dataset before claiming it, '
                  'and report the per-class thresholds so a reader can see the spread '
                  'a single τ was forced to average over.')

    text = '\n'.join(md)
    print('\n' + text)
    if args.md:
        p = Path(args.md).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f'\nwritten: {p}')


if __name__ == '__main__':
    main()
