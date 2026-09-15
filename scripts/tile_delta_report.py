"""Is a per-tile win/loss count evidence of anything? Usually not — here is why.

⛔ THE QUESTION THIS EXISTS TO ANSWER HONESTLY. `fig_dlrsd_compare.py --all`
reports "improved 1057, worse 769", and a reader is entitled to ask what a
dataset-level gain is worth when 769 tiles got worse. A ratio cannot answer it:
what decides the outcome is whether the wins are LARGER than the losses, and
whether the losses have a nameable cause or are spread at random.

⚠️ AND PER-TILE MEAN IoU IS NOT THE DATASET mIoU. Per-tile averages over the
classes in that tile; dataset mIoU averages each class once over every pixel. A
two-class tile weights each class 50%. The two averages can disagree in SIGN --
on DLRSD `field` gains +8.14 as a class while individual agricultural tiles fall
to zero -- so this report is a diagnostic, never a headline.

    python scripts/tile_delta_report.py --index ~/outputs/dlrsd_compare/index.csv
"""
import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

GROUP_RE = re.compile(r'^([A-Za-z_\-]+?)[\-_]?\d+$')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', required=True)
    ap.add_argument('--md', default=None)
    ap.add_argument('--top', type=int, default=8)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(Path(args.index).expanduser())))
    for r in rows:
        for k in ('baseline', 'ours', 'delta', 'changed_pct'):
            r[k] = float(r[k])
        for k in ('fixed', 'broke', 'newly_discarded', 'all_discarded'):
            r[k] = int(r[k])
    d = np.array([r['delta'] for r in rows])
    win, lose = d[d > 0.05], d[d < -0.05]

    def stat(a, f):
        """nan is not a number a reader can act on; say 'none' instead."""
        return f'{f(a):+.2f}' if a.size else '—'

    L = [f'# Per-tile wins and losses — {len(rows)} tiles\n',
         '| | tiles | mean Δ | median | total mass |', '|---|---|---|---|---|',
         f'| improved | **{len(win)}** | **{stat(win, np.mean)}** | {stat(win, np.median)} | **{win.sum():+,.0f}** |',
         f'| worse | **{len(lose)}** | **{stat(lose, np.mean)}** | {stat(lose, np.median)} | **{lose.sum():+,.0f}** |',
         f'| unchanged | {len(d) - len(win) - len(lose)} | — | — | — |',
         f'| **net** | | **{d.mean():+.2f}** | {np.median(d):+.2f} | **{d.sum():+,.0f}** |\n']

    ratio = win.sum() / abs(lose.sum()) if lose.size and lose.sum() else float('inf')
    L.append(f'⭐ **Win mass ÷ loss mass = {ratio:.2f}x.** The count ratio is '
             f'{len(win) / max(len(lose), 1):.2f}x, and it is the MASS that '
             f'decides the dataset number, not the count.\n')
    if ratio < 1.2:
        L.append('⛔ **The wins barely outweigh the losses per tile.** Check the '
                 'dataset-level metrics before claiming anything: a per-tile view '
                 'weights a 2-class tile the same as a 17-class one.\n')

    # ---- do the losses have a cause, or are they everywhere?
    cat = defaultdict(list)
    for r in rows:
        m = GROUP_RE.match(r['tile'])
        cat[m.group(1) if m else '?'].append(r['delta'])
    stats = sorted(((np.mean(v), k, len(v)) for k, v in cat.items()))
    L += ['## Which scenes lose — a cause, or noise?\n',
          '| scene category | tiles | mean Δ | worse |', '|---|---|---|---|']
    for mu, k, nn in stats[:args.top]:
        w = sum(1 for x in cat[k] if x < -0.05)
        L.append(f'| `{k}` | {nn} | **{mu:+.2f}** | {w}/{nn} |')
    L.append('\n*(best few, for contrast)*\n')
    L += ['| scene category | tiles | mean Δ | worse |', '|---|---|---|---|']
    for mu, k, nn in stats[-3:][::-1]:
        w = sum(1 for x in cat[k] if x < -0.05)
        L.append(f'| `{k}` | {nn} | **{mu:+.2f}** | {w}/{nn} |')

    worst = stats[0]
    conc = sum(1 for x in cat[worst[1]] if x < -0.05) / max(len(lose), 1)
    L.append(f'\n⭐ **`{worst[1]}` alone holds {100 * conc:.1f}% of all losing '
             f'tiles.** A loss concentrated in one scene type has a cause worth '
             f'naming; losses spread evenly across 21 categories would mean the '
             f'fitted vector is simply too coarse.\n'
             if conc > 0.15 else
             f'\n⚠️ **Losses are spread across categories** (worst is '
             f'`{worst[1]}` at {100 * conc:.1f}% of them), so there is no single '
             f'scene type to blame — the fitted vector is too coarse for part of '
             f'the dataset.\n')

    # ---- how much of the damage is the discard sink rather than a wrong label?
    nd = np.array([r['newly_discarded'] for r in rows])
    br = np.array([r['broke'] for r in rows])
    allz = [r for r in rows if r['all_discarded']]
    L += ['## Is the damage a WRONG label, or a discarded one?\n',
          f'- pixels newly sent to the sink: **{nd.sum():,}**',
          f'- pixels that were right and became wrong: **{br.sum():,}**',
          f'- ⛔ tiles discarded **entirely**: **{len(allz)}**'
          + (f' — {Counter(GROUP_RE.match(r["tile"]).group(1) for r in allz if GROUP_RE.match(r["tile"]))}'
             if allz else ''), '']
    if len(allz):
        L.append('⛔ **A tile that goes to zero with nothing mislabelled is a '
                 'different failure from a tile that goes to zero wrongly '
                 'labelled**, and only the second is a segmentation error. Both '
                 'are real costs; they need different fixes.\n')

    L += ['## ⚠️ What this report does NOT settle\n',
          'Nothing here is the dataset metric. **mIoU, `aAcc` and precision/recall '
          'are measured by `eval.py` over every pixel** and are what the claim '
          'rests on; this is a per-tile view whose averaging differs. Quote it to '
          'explain WHERE the method helps and hurts, never to argue whether it '
          'does.\n']
    text = '\n'.join(L)
    print(text)
    if args.md:
        p = Path(args.md).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f'\nwritten: {p}')


if __name__ == '__main__':
    main()
