"""
Can a LABEL-FREE statistic predict per-class precision -- and therefore the right
per-class threshold?

THE BOUND THIS ATTACKS. WEEK3 §9a: per-class thresholds are worth +1.46 mIoU on
LoveDA, no label-free rule we tested reaches them (Otsu −0.17, presence-scaled
−0.74, equal-commitment −2.98), and the stated reason is structural -- what sets
the right threshold is per-class PRECISION, and precision is label-derived by
definition. §9b then showed ~200 labelled tiles buy +1.18 of it.

That leaves one question open, and it is the most valuable one left: those three
rules all describe how the MODEL's scores are DISTRIBUTED. None of them asks how
often the model is RIGHT. Cross-head agreement does. SAM 3 carries two
independent heads -- a dense semantic head and a DETR-style instance head -- and
SegEarth-OV3 fuses them with an elementwise max, which destroys the distinction.
Where both heads independently pick the same class, the prediction has two
sources of support; where only one fires, it has one. **That is a consistency
signal, computable with no labels, and consistency is the classical proxy for
correctness.**

BOTH OUTCOMES ARE PUBLISHABLE, which is why this is worth the run:

  it works    -- the paper stops being "here is a bound nobody can reach" and
                 becomes "here is a bound, and here is how to reach part of it
                 without labels". That is a method, not just an analysis.
  it fails    -- the impossibility argument gets STRONGER, because the most
                 plausible remaining candidate has been eliminated rather than
                 left unaddressed. A reviewer would otherwise propose it.

TWO LEVELS OF EVIDENCE, and the second is the one that counts.

  screen    Spearman(proxy, target) across classes. With 6 classes this is a
            weak test, so the p-value is computed by EXACT enumeration of all
            6! = 720 relabelings rather than asserted from a table. A proxy that
            cannot clear this does not get to the second level.

  verdict   Turn the proxy into a threshold rule spending ONE label-tuned knob --
            parity with the baseline, which also tunes its single tau with
            labels -- fit that knob on calibration tiles, and measure delta mIoU
            on disjoint tiles. Correlation is a screen; delta mIoU is the result.

⚠️ A RANDOM-PROXY CONTROL IS INCLUDED AND IS NOT OPTIONAL. One fitted knob over
six classes can buy a gain from a meaningless vector, exactly as an atom's mean
colour bought AUC 0.966 on random images in §9. Any proxy that fails to beat the
random control's spread has shown nothing.

    python scripts/precision_proxy.py --cache ~/outputs/week3_fused/cache \\
        --tau 0.5 --calib 200 --md ~/outputs/week3/precision_proxy.md
"""
import argparse
import itertools
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels                                                    # noqa: E402
from tau_oracle import confusion_at, miou, per_class_iou, NBINS  # noqa: E402
from tau_cv import per_tile_hists, fit                           # noqa: E402


# ---------------------------------------------------------------- proxies
def accumulate(files, nc):
    """Per-predicted-class label-free statistics. No `gt` is read here at all.

    Computed over every pixel, including no-data: masking those out would use an
    annotation artefact, and a rule that needs the annotation file is not
    label-free in the sense that matters.
    """
    n = np.zeros(nc)
    acc = {k: np.zeros(nc) for k in
           ('conf', 'margin', 'fconf', 'stable', 'agree', 'ifires', 'iconf', 'sconf')}
    spres = np.zeros(nc)
    spres_n = np.zeros(nc)
    have = {'fconf': False, 'heads': False}

    for i, f in enumerate(files):
        z = np.load(f)
        pred = z['pred'].astype(np.int32)
        conf = z['conf'].astype(np.float32)
        conf2 = z['conf2'].astype(np.float32) if 'conf2' in z.files else conf
        fconf = z['fconf'].astype(np.float32) if 'fconf' in z.files else None
        fpred = z['fpred'].astype(np.int32) if 'fpred' in z.files else None
        ipred = z['ipred'].astype(np.int32) if 'ipred' in z.files else None
        spred = z['spred'].astype(np.int32) if 'spred' in z.files else None
        iconf = z['iconf'].astype(np.float32) if 'iconf' in z.files else None
        sconf = z['sconf'].astype(np.float32) if 'sconf' in z.files else None
        if fconf is not None:
            have['fconf'] = True
        if ipred is not None and spred is not None:
            have['heads'] = True

        cnt = np.bincount(pred.ravel(), minlength=nc)[:nc]
        n += cnt
        acc['conf'] += np.bincount(pred.ravel(), conf.ravel(), nc)[:nc]
        acc['margin'] += np.bincount(pred.ravel(), (conf - conf2).ravel(), nc)[:nc]
        if fconf is not None:
            acc['fconf'] += np.bincount(pred.ravel(), fconf.ravel(), nc)[:nc]
            acc['stable'] += np.bincount(pred.ravel(),
                                         (fpred == pred).ravel().astype(float), nc)[:nc]
        if ipred is not None and spred is not None:
            # the two heads independently pick the SAME class -> two sources of
            # support for this pixel; only one fires -> one source
            acc['agree'] += np.bincount(pred.ravel(),
                                        (ipred == spred).ravel().astype(float), nc)[:nc]
            acc['ifires'] += np.bincount(pred.ravel(),
                                         (iconf > 0).ravel().astype(float), nc)[:nc]
            acc['iconf'] += np.bincount(pred.ravel(), iconf.ravel(), nc)[:nc]
            acc['sconf'] += np.bincount(pred.ravel(), sconf.ravel(), nc)[:nc]

        if 'spres' in z.files:
            sp = np.asarray(z['spres'], float)
            with np.errstate(all='ignore'):
                v = np.nanmax(sp, axis=0) if sp.ndim == 2 else sp
            ok = np.isfinite(v)
            spres[:len(v)][ok] += v[ok]
            spres_n[:len(v)][ok] += 1
        if (i + 1) % 250 == 0 or i + 1 == len(files):
            print(f'  proxies {i + 1}/{len(files)}')

    d = np.where(n > 0, n, 1.0)
    P = {'mean_conf': acc['conf'] / d,
         'mean_margin': acc['margin'] / d,
         'presence': np.where(spres_n > 0, spres / np.where(spres_n > 0, spres_n, 1), np.nan)}
    if have['fconf']:
        P['gate_ratio'] = acc['conf'] / np.where(acc['fconf'] > 0, acc['fconf'], np.nan)
        P['argmax_stability'] = acc['stable'] / d
    if have['heads']:
        P['sem_inst_agree'] = acc['agree'] / d          # <- the one this is for
        P['inst_fires'] = acc['ifires'] / d
        P['head_conf_ratio'] = acc['iconf'] / np.where(acc['sconf'] != 0, acc['sconf'], np.nan)
    return P, n, have


# ---------------------------------------------------------------- stats
def spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def exact_p(a, b):
    """Two-sided p by enumerating every relabeling. With 6-8 classes this is
    cheap and exact, and it is the only honest way to read a correlation at n=6:
    the smallest attainable two-sided p over 6 classes is 2/720 = 0.003, and
    |rho| = 0.6 arises by chance roughly a fifth of the time."""
    ok = np.isfinite(a) & np.isfinite(b)
    x, y = a[ok], b[ok]
    if len(x) < 3 or len(x) > 9:
        return np.nan
    obs = abs(spearman(x, y))
    hits = tot = 0
    for perm in itertools.permutations(range(len(x))):
        tot += 1
        if abs(spearman(x[list(perm)], y)) >= obs - 1e-12:
            hits += 1
    return hits / tot


def pr_at(C, k):
    tp = C[k, k]
    prec = tp / C[:, k].sum() if C[:, k].sum() else np.nan
    rec = tp / C[k].sum() if C[k].sum() else np.nan
    return 100 * prec, 100 * rec


# ---------------------------------------------------------------- rule
def rule_taus(proxy, tau_pub, b, nc, bg):
    """tau_c = tau_pub + b * z(proxy_c). ONE knob, `b`, which also carries the
    SIGN -- so the rule is not told which direction the proxy should point."""
    v = np.asarray(proxy, float).copy()
    m = np.isfinite(v)
    z = np.zeros(nc)
    if m.sum() > 1 and v[m].std() > 0:
        z[m] = (v[m] - v[m].mean()) / v[m].std()
    t = np.clip(tau_pub + b * z, 0.0, 1.0)
    t[bg] = tau_pub
    return t


def best_knob(Hcal, proxy, tau_pub, nc, bg, objective='real'):
    def sc(t):
        C = confusion_at(Hcal, t, bg, NBINS)
        v = per_class_iou(C)
        if objective == 'real':
            v = np.array([v[c] for c in range(nc) if c != bg])
        return float(np.nanmean(v)) if np.isfinite(v).any() else 0.0
    grid = np.arange(-40, 41) / 100.0            # b in [-0.40, +0.40], step 0.005
    return max(((sc(rule_taus(proxy, tau_pub, b, nc, bg)), b) for b in grid))[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--tau', type=float, required=True)
    ap.add_argument('--calib', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--controls', type=int, default=200,
                    help='random proxies for the negative control')
    ap.add_argument('--proxy-on', choices=['calib', 'all'], default='calib',
                    help="which tiles the label-free proxy is measured on. 'calib' is "
                         "conservative; 'all' is also legitimate (the statistic uses no "
                         "labels) and only makes the proxy less noisy.")
    ap.add_argument('--md', default=None)
    args = ap.parse_args()

    LB = labels.from_cache(args.cache)
    nc, bg = LB.n, LB.bg - 1
    files = sorted(Path(args.cache).expanduser().glob('*.npz'))
    n = len(files)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n)
    cal, held = idx[:args.calib], idx[args.calib:]
    print(f'  classes: {LB}\n{n} tiles | calib {args.calib} | held out {n - args.calib}\n')

    PT = per_tile_hists(files, nc, NBINS)
    Hcal = PT[cal].sum(0).astype(np.int64)
    Hhel = PT[held].sum(0).astype(np.int64)

    src = [files[i] for i in (cal if args.proxy_on == 'calib' else range(n))]
    P, npix, have = accumulate(src, nc)

    # ---- targets, from labels, for SCORING only
    Ccal = confusion_at(Hcal, np.full(nc, args.tau), bg, NBINS)
    prec = np.array([pr_at(Ccal, k)[0] for k in range(nc)])
    rec = np.array([pr_at(Ccal, k)[1] for k in range(nc)])
    gap = prec - rec
    tau_or = fit(Hcal, bg, NBINS, objective='real')

    real = [c for c in range(nc) if c != bg]
    tgt = {'precision': prec, 'P−R gap': gap, 'oracle τ': tau_or}

    md = ['# Cross-head agreement as a label-free precision proxy\n',
          f'- cache: `{args.cache}`  |  {n} tiles  |  calib **{args.calib}**, '
          f'held out **{n - args.calib}**  |  published τ **{args.tau}**',
          f'- proxies measured on the **{args.proxy_on}** tiles, using no labels',
          f'- head-separated arrays present: **{"yes" if have["heads"] else "NO"}**'
          + ('' if have['heads'] else
             '  ⚠️ re-run `measure_discard_rate.py` with the patched segmentor to '
             'record `iconf/ipred/sconf/spred`; without them the cross-head rows '
             'below are simply absent and only the weaker proxies are tested') + '\n',
          '## The classes, and what has to be predicted\n',
          '| class | precision | recall | P−R gap | oracle τ |', '|---|---|---|---|---|']
    for c in real:
        md.append(f'| {LB.names[c]} | {prec[c]:.1f} | {rec[c]:.1f} | {gap[c]:+.1f} | '
                  f'**{tau_or[c]:.3f}** |')
    md.append('')

    md += ['## Level 1 — screen: does any proxy rank the classes correctly?\n',
           'Spearman across the real classes, with a two-sided p from **exact enumeration** '
           f'of all {math.factorial(len(real)):,} relabelings. At this n a correlation is '
           'a hint, never a result.\n',
           '| proxy | ' + ' | '.join(f'ρ vs {k} | p' for k in tgt) + ' |',
           '|---' * (1 + 2 * len(tgt)) + '|']
    rows = []
    for name, v in P.items():
        vr = np.array([v[c] for c in real])
        cells, best = [], 1.0
        for k, t in tgt.items():
            tr = np.array([t[c] for c in real])
            r, p = spearman(vr, tr), exact_p(vr, tr)
            best = min(best, p if np.isfinite(p) else 1.0)
            cells += [f'{r:+.3f}' if np.isfinite(r) else '—',
                      f'{p:.3f}' if np.isfinite(p) else '—']
        star = ' ⭐' if best <= 0.05 else ''
        md.append(f'| `{name}`{star} | ' + ' | '.join(cells) + ' |')
        rows.append((name, best))
    md.append('')

    # ---- Level 2: delta mIoU on held-out tiles
    m_pub = miou(confusion_at(Hhel, np.full(nc, args.tau), bg, NBINS))
    m_or = miou(confusion_at(Hhel, fit(Hhel, bg, NBINS, objective='real'), bg, NBINS))
    oracle_gain = m_or - m_pub

    md += ['## Level 2 — verdict: does it move mIoU on held-out tiles?\n',
           'Each rule is `τ_c = τ_pub + b · z(proxy_c)`, with the single knob `b` — sign '
           'included, so the rule is not told which way the proxy should point — fitted on '
           'the calibration tiles and applied to disjoint tiles. Same one-knob budget as '
           "`tau_rules.py`, and parity with the baseline, which also tunes its single τ "
           'with labels.\n',
           '| rule | fitted `b` | held-out mIoU | Δ | share of oracle |',
           '|---|---|---|---|---|',
           f'| published τ = {args.tau} | — | {m_pub:.2f} | — | — |']
    results = []
    for name, v in P.items():
        b, _ = best_knob(Hcal, v, args.tau, nc, bg)
        t = rule_taus(v, args.tau, b, nc, bg)
        m = miou(confusion_at(Hhel, t, bg, NBINS))
        results.append((name, b, m - m_pub))
        md.append(f'| `{name}` | {b:+.3f} | {m:.2f} | **{m - m_pub:+.2f}** | '
                  f'{100 * (m - m_pub) / oracle_gain:+.0f}% |')

    # ---- the control. One knob over a handful of classes can buy a gain from a
    # meaningless vector; without this the table above cannot be read.
    ctrl = []
    for _ in range(args.controls):
        v = rng.normal(size=nc)
        b, _ = best_knob(Hcal, v, args.tau, nc, bg)
        ctrl.append(miou(confusion_at(Hhel, rule_taus(v, args.tau, b, nc, bg),
                                      bg, NBINS)) - m_pub)
    ctrl = np.array(ctrl)
    md.append(f'| _random proxy_ ×{args.controls} | _fitted_ | — | '
              f'_{ctrl.mean():+.2f} ± {ctrl.std(ddof=1):.2f}_ | '
              f'_p95 {np.percentile(ctrl, 95):+.2f}_ |')
    md.append(f'| **oracle per-class τ** | _N−1 params_ | {m_or:.2f} | '
              f'**{oracle_gain:+.2f}** | 100% |\n')
    md.append(f'⚠️ **A proxy only counts if it beats the random control\'s 95th percentile '
              f'({np.percentile(ctrl, 95):+.2f}).** One fitted knob over {len(real)} classes '
              'can extract a gain from noise — the same confound that gave a colour-novelty '
              'detector AUC 0.966 on random images (§9).\n')

    # ---- verdict
    thr = np.percentile(ctrl, 95)
    winners = sorted([r for r in results if r[2] > thr and r[2] > 0],
                     key=lambda r: -r[2])
    md += ['## Verdict\n']
    if not have['heads']:
        md.append('⚠️ **The cross-head proxy was not tested** — `iconf/ipred/sconf/spred` are '
                  'absent from this cache. Everything above is the weaker set of proxies. '
                  'Re-run `measure_discard_rate.py` with the patched segmentor (~25 min GPU) '
                  'before drawing any conclusion about cross-head agreement.\n')
    if not winners:
        md.append(f'⛔ **No label-free proxy beats the random control.** The best is '
                  f'`{max(results, key=lambda r: r[2])[0]}` at '
                  f'{max(r[2] for r in results):+.2f} against a control p95 of {thr:+.2f}, '
                  f'and an oracle bound of {oracle_gain:+.2f}.\n\n'
                  '**This strengthens §9a rather than weakening it.** The impossibility '
                  'argument previously rested on three rules that describe how the model\'s '
                  'scores are *distributed*. Cross-head agreement asks how often the model '
                  'is *right*, which is the one remaining candidate a reviewer would '
                  'propose — and it does not work either. Report it as an eliminated '
                  'alternative, with this table, not as an untested gap.')
    else:
        nm, b, g = winners[0]
        md.append(f'✅ **`{nm}` reaches {g:+.2f} mIoU**, {100 * g / oracle_gain:.0f}% of the '
                  f'{oracle_gain:+.2f} oracle bound, against a random-control p95 of '
                  f'{thr:+.2f}. Fitted knob `b = {b:+.3f}`.\n\n'
                  '**If this is the cross-head proxy, it changes the paper\'s claim**: §9a '
                  'becomes "per-class thresholds are worth X, labels buy most of it, and a '
                  'label-free consistency signal buys a measurable share with no annotation '
                  'at all". Before claiming it: replicate on the second dataset, check the '
                  'fitted sign is the same on both — a sign flip means the knob is fitting '
                  'the split, not the signal — and report the random control beside it.')

    text = '\n'.join(md)
    print('\n' + text)
    if args.md:
        p = Path(args.md).expanduser(); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text); print(f'\nwritten: {p}')


if __name__ == '__main__':
    main()
