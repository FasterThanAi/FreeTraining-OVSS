"""
WEEK1_RESULTS 9.2b — did presence gating CAUSE the catastrophic tiles, or merely
correlate with them?

9.2a established leg (a): S_pres is low on catastrophic tiles (median 0.273 vs
0.918 healthy, r = -0.750 over 1669 tiles). But that correlation is partly
mechanical -- P_final = P_fused * S_pres and a pixel is discarded iff
P_final < tau, so low presence forces discard by construction.

Leg (b) is the counterfactual: turn gating off (P_final = P_fused) and see
whether those tiles recover. This script compares the two runs.

Usage (CPU only, seconds):

    python scripts/compare_presence_counterfactual.py \
        --baseline ~/outputs/week2_tau0.5_instrumented \
        --counterfactual ~/outputs/week2_tau0.5_nopresence

Reads from each run:  per_image_discard.csv, confusion_matrix.npy
and from the baseline: per_image_presence.csv

Two analyses, and they answer different questions -- report both:

  A. PER-TILE. Do the 198 catastrophic tiles recover? This is the actual 9.2b
     question. Judge it here, NOT on aggregate mIoU.

  B. AGGREGATE PRECISION. Of the pixels gating was suppressing, how many come
     back CORRECT vs how many arrive as new false positives? This mirrors 8.2's
     tau-sweep result (1.73 wrong per 1 right) and is the honest cost column
     required by 12. Recovering pixels carelessly is worse than not recovering.
"""
import argparse
import csv
import os
from pathlib import Path

import numpy as np

CLASSES = ['background', 'building', 'road', 'water',
           'barren', 'forest', 'agricultural']
N = len(CLASSES)
BG = 0                      # background is index 0 in the confusion matrix


def load_discard(run):
    out = {}
    with open(Path(run) / 'per_image_discard.csv') as f:
        for r in csv.DictReader(f):
            out[r['image']] = dict(
                real=int(r['real_px']),
                disc=int(r['discarded_px']),
                pct=float(r['discard_pct_of_real']),
            )
    return out


def load_presence(run):
    p = Path(run) / 'per_image_presence.csv'
    if not p.exists():
        return {}
    out = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                out[r['image']] = float(r['spres_max'])
            except (ValueError, KeyError):
                out[r['image']] = float('nan')
    return out


def miou(cm):
    inter = np.diag(cm).astype(float)
    union = cm.sum(1) + cm.sum(0) - np.diag(cm)
    return float(np.nanmean(np.where(union > 0, inter / np.maximum(union, 1), np.nan)) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', required=True, help='run WITH presence gating')
    ap.add_argument('--counterfactual', required=True, help='run with --no-presence')
    ap.add_argument('--cat-thd', type=float, default=99.0, help='%% discard defining catastrophic')
    ap.add_argument('--healthy-thd', type=float, default=1.0)
    args = ap.parse_args()

    b, c = load_discard(args.baseline), load_discard(args.counterfactual)
    pres = load_presence(args.baseline)
    shared = sorted(set(b) & set(c))
    if not shared:
        raise SystemExit('No overlapping images between the two runs.')
    print(f'{len(shared)} tiles in both runs\n')

    cm_b = np.load(Path(args.baseline) / 'confusion_matrix.npy')
    cm_c = np.load(Path(args.counterfactual) / 'confusion_matrix.npy')

    md = ['# 9.2b — Presence-gating counterfactual\n',
          f'- baseline (gating ON):  `{args.baseline}`',
          f'- counterfactual (OFF):  `{args.counterfactual}`',
          f'- mIoU: **{miou(cm_b):.2f}** -> **{miou(cm_c):.2f}** '
          f'({miou(cm_c)-miou(cm_b):+.2f})',
          '\n> mIoU is EXPECTED to fall: gating helps on average, which is why it exists.',
          '> That is not a refutation. The question is what it costs on the tail. Judge 9.2b',
          '> on section A, not on this number.\n']

    # ---------------- A. per-tile ----------------
    cat = [n for n in shared if b[n]['pct'] >= args.cat_thd]
    hea = [n for n in shared if b[n]['pct'] < args.healthy_thd]
    mid = [n for n in shared if args.healthy_thd <= b[n]['pct'] < args.cat_thd]

    md += ['## A. Per-tile — do the catastrophic tiles recover?\n',
           '| Tile set (by baseline discard) | n | discard before | discard after | change |',
           '|---|---|---|---|---|']
    for lbl, grp in (('**catastrophic** (>=%.0f%%)' % args.cat_thd, cat),
                     ('middle', mid),
                     ('healthy (<%.0f%%)' % args.healthy_thd, hea)):
        if not grp:
            md.append(f'| {lbl} | 0 | - | - | - |'); continue
        bb = np.mean([b[n]['pct'] for n in grp])
        cc = np.mean([c[n]['pct'] for n in grp])
        md.append(f'| {lbl} | {len(grp)} | {bb:.2f}% | {cc:.2f}% | **{cc-bb:+.2f}** |')

    if cat:
        after = np.array([c[n]['pct'] for n in cat])
        rec = [(int((after < t).sum()), t) for t in (50, 25, 10)]
        md += ['\n**Recovery among the catastrophic set:**\n']
        for k, t in rec:
            md.append(f'- {k}/{len(cat)} ({100*k/len(cat):.1f}%) now discard <{t}%')
        # did presence predict which ones recover?
        sp = np.array([pres.get(n, np.nan) for n in cat])
        d = np.array([b[n]['pct'] - c[n]['pct'] for n in cat])
        ok = np.isfinite(sp)
        if ok.sum() > 2 and np.std(sp[ok]) > 0:
            r = float(np.corrcoef(sp[ok], d[ok])[0, 1])
            md.append(f'- correlation(baseline `spres_max`, recovery) = **{r:+.3f}** '
                      f'over {int(ok.sum())} catastrophic tiles')

    # ---------------- B. aggregate precision ----------------
    # Of the pixels gating suppressed, how many return correct vs wrong?
    md += ['\n## B. Aggregate — are the recovered pixels CORRECT?\n',
           '`recovered` = real-class pixels no longer sent to background.',
           '`correct` = of those, how many landed on their true class.',
           '`new FP` = true-background pixels newly claimed by this class.\n',
           '| Class | recovered | of which correct | precision | new FP | wrong per right |',
           '|---|---|---|---|---|---|']
    # NB signs are NOT clamped. A negative "recovered" means the intervention sent
    # MORE real-class pixels to background, which is a real and important outcome --
    # an earlier version clamped at zero and produced an incoherent total row.
    tot_rec = tot_cor = tot_fp = 0
    for k in range(1, N):
        recovered = int(cm_b[k, BG] - cm_c[k, BG])
        correct = int(cm_c[k, k] - cm_b[k, k])
        new_fp = int(cm_c[BG, k] - cm_b[BG, k])
        tot_rec += recovered; tot_cor += correct; tot_fp += new_fp
        prec = f'{100*correct/recovered:.1f}%' if recovered > 0 else 'n/a'
        wpr = f'{new_fp/correct:.2f}' if correct > 0 else 'n/a'
        md.append(f'| {CLASSES[k]} | {recovered:,} | {correct:,} | {prec} | '
                  f'{new_fp:,} | {wpr} |')
    prec = f'{100*tot_cor/tot_rec:.1f}%' if tot_rec > 0 else 'n/a'
    wpr = f'{tot_fp/tot_cor:.2f}' if tot_cor > 0 else 'n/a'
    md.append(f'| **total** | **{tot_rec:,}** | **{tot_cor:,}** | **{prec}** | '
              f'**{tot_fp:,}** | **{wpr}** |')
    if tot_rec > 0 and tot_cor > 0:
        md += [f'\nCompare against 8.2: threshold relaxation (tau 0.5->0.1) bought 1 correct '
               f'pixel per **1.73** wrong. This intervention buys 1 per **{wpr}**.\n']
    else:
        md += ['\n> **`n/a` is the result, not a formatting failure.** A negative `recovered`',
               '> means the intervention sent MORE real-class pixels to background, and a',
               '> negative `correct` means it produced FEWER correct pixels. There is no',
               '> recovery to price, so "wrong per right" is undefined. Compare the totals',
               '> against zero, not against 8.2\'s 1.73:1.\n']

    # ---------------- verdict ----------------
    md += ['## Verdict\n']
    if cat:
        drop = np.mean([b[n]['pct'] for n in cat]) - np.mean([c[n]['pct'] for n in cat])
        harm = 0.0
        if hea:
            harm = np.mean([c[n]['pct'] for n in hea]) - np.mean([b[n]['pct'] for n in hea])
        selective = np.nan
        sp = np.array([pres.get(n, np.nan) for n in cat])
        d = np.array([b[n]['pct'] - c[n]['pct'] for n in cat])
        ok = np.isfinite(sp)
        if ok.sum() > 2 and np.std(sp[ok]) > 0:
            selective = float(np.corrcoef(sp[ok], d[ok])[0, 1])

        # A causal claim needs the drop AND selectivity: it must help the tiles
        # presence was suppressing, and not wreck the ones it was not.
        md.append(f'- catastrophic discard change: **{-drop:+.1f}** points')
        md.append(f'- healthy discard change: **{harm:+.1f}** points')
        md.append(f'- correlation(baseline `spres_max`, recovery): **{selective:+.3f}**\n')

        if np.isfinite(selective) and selective > -0.2 and harm > 10:
            md.append(
                '**Presence gating is a CORRELATE, not a cause.** Two disqualifying facts. '
                f'(1) Recovery is uncorrelated with baseline presence ({selective:+.3f}) -- if '
                'gating were suppressing recoverable evidence, the lowest-presence tiles would '
                'recover most, and that correlation would be strongly negative. (2) The same '
                f'intervention makes healthy, barely-gated tiles worse by {harm:.1f} points, so '
                'whatever it is doing is not "releasing suppressed evidence".\n\n'
                'Scope 9.2 down: tile 3487 illustrates that the mechanism CAN occur, but it does '
                'not generalise to a causal claim. Finding this now costs one run; finding it in '
                'week 11 costs a results section.\n\n'
                'Note the likely mechanism: `background` is itself a gated query, and in '
                'predict() it can win by argmax as well as by threshold. Removing gating lifts '
                'background too, so it wins outright on tiles it previously lost. Check '
                '`spres_background` against the real classes in per_image_presence.csv.')
        elif drop > 30 and np.isfinite(selective) and selective < -0.3 and harm < 5:
            md.append(
                f'**Leg (b) established.** Catastrophic discard falls {drop:.1f} points, recovery '
                f'tracks baseline presence ({selective:+.3f}), and healthy tiles are not harmed '
                f'({harm:+.1f}). The dense evidence was there and gating was suppressing it. '
                '9.2 becomes a causal claim, and it argues FOR the method -- a local '
                'co-occurrence prior is the natural correction for a wrong GLOBAL scalar.')
        else:
            md.append(
                f'**Mixed.** Catastrophic discard moves {-drop:+.1f} points, healthy {harm:+.1f}, '
                f'selectivity {selective:+.3f}. Not a clean causal result. Report presence '
                'collapse as an observed correlate and do not claim causation without a '
                'sharper intervention.')
    text = '\n'.join(md)
    print(text)
    out = Path(args.counterfactual) / 'counterfactual_summary.md'
    out.write_text(text)
    print(f'\nWritten to {out}')


if __name__ == '__main__':
    main()
