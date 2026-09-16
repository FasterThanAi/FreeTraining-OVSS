"""Recover the headline numbers from a cache whose summary never got written.

⛔ WHY THIS EXISTS. `measure_discard_rate.py` writes each tile's `.npz` inside
the loop and the summary afterwards, so a crash between the two leaves a
COMPLETE cache and no numbers. That happened on the DLRSD vocabulary arm --
a typo in the sidecar block, in a GPU-only path that had never been executed --
and re-running 2100 tiles on the GPU to recover arithmetic would be waste.

⭐ Everything the summary reports is derivable from the cache, because `pred` is
the PRE-threshold argmax and `conf` is the winning score: the threshold is
applied here rather than baked in. The equivalence is not assumed -- it is the
same cross-check `tau_oracle.py` prints, which read 37.89 against
`measure_discard_rate.py`'s 37.89 on DLRSD.

    python scripts/cache_summary.py --cache ~/outputs/dlrsd_v2/cache --tau 0.1
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels as LB_MOD  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--tau', type=float, required=True)
    ap.add_argument('--bg-idx', type=int, default=None,
                    help='only needed if the cache has no _meta.json sidecar')
    ap.add_argument('--md', default=None)
    args = ap.parse_args()

    LB = LB_MOD.from_cache(args.cache, discard_idx=args.bg_idx)
    n, sink = LB.n, LB.bg - 1
    files = sorted(Path(args.cache).expanduser().glob('*.npz'))
    print(f'  {len(files)} tiles | τ = {args.tau} | {n} classes\n')

    npred = max(n, sink + 1)
    C = np.zeros((n, npred), np.int64)
    lost = np.zeros(n, np.int64)
    total = np.zeros(n, np.int64)
    per_tile = []

    for i, f in enumerate(files, 1):
        z = np.load(f)
        gt = z['gt'].astype(np.int32) - 1            # 0-indexed, -1 = no-data
        conf = z['conf'].astype(np.float32)
        pred = z['pred'].astype(np.int32)
        m = gt >= 0
        if not m.any():
            continue
        g, p, cf = gt[m], pred[m].copy(), conf[m]
        p[cf < args.tau] = sink                       # the segmentor's rule
        np.add.at(C, (g, np.clip(p, 0, npred - 1)), 1)
        real = np.ones(g.shape, bool) if LB.sink else (g != sink)
        np.add.at(total, g[real], 1)
        d = real & (p == sink)
        if d.any():
            np.add.at(lost, g[d], 1)
        per_tile.append(100.0 * d.sum() / max(real.sum(), 1))
        if i % 500 == 0 or i == len(files):
            print(f'    {i}/{len(files)}')

    sq = C[:, :n]
    inter = np.diag(sq).astype(float)
    union = C.sum(1) + sq.sum(0) - inter
    iou = np.where(union > 0, 100.0 * inter / np.maximum(union, 1), np.nan)
    miou = float(np.nanmean(iou))
    pt = np.array(per_tile)

    L = [f'# Cache summary — `{args.cache}`\n',
         f'- tiles **{len(files)}** | τ **{args.tau}** | classes **{n}**'
         + ('  | ⭐ unscored sink, no catch-all' if LB.sink else ''),
         f'- **mIoU {miou:.2f}**',
         f'- real-class pixels **{total.sum():,}**',
         f'- ⭐ **discarded {lost.sum():,} ({100 * lost.sum() / max(total.sum(), 1):.2f}%)**',
         f'- per-tile discard: mean **{pt.mean():.2f}%**, median **{np.median(pt):.2f}%**, '
         f'max {pt.max():.2f}%',
         f'- ⛔ tiles at 100% discard: **{int((pt >= 99.5).sum())}**\n',
         '| class | IoU | GT pixels | lost to the sink | % lost |',
         '|---|---|---|---|---|']
    order = np.argsort(-np.where(total > 0, 100.0 * lost / np.maximum(total, 1), -1))
    for k in order:
        if total[k] == 0:
            continue
        L.append(f'| `{LB.names[k]}` | {iou[k]:.2f} | {total[k]:,} | {lost[k]:,} | '
                 f'**{100 * lost[k] / total[k]:.2f}%** |')
    dead = [LB.names[k] for k in range(n) if total[k] > 0 and iou[k] < 0.5]
    L.append('')
    if dead:
        L.append(f'⛔ **Classes at essentially zero IoU: {", ".join(dead)}** — '
                 f'{len(dead)} of {n} is **{100 * len(dead) / n:.1f}% of the metric** '
                 f'contributing nothing.\n')
    else:
        L.append('⭐ **Every class scores above 0.5 IoU.**\n')
    L.append('⚠️ Computed from the cache, not from the segmentor. It is the same '
             'arithmetic `tau_oracle.py` cross-checks, but if this is the first '
             'number quoted for a cache, confirm it against an `eval.py` pass '
             'before it is load-bearing.\n')

    text = '\n'.join(L)
    print('\n' + text)
    if args.md:
        p = Path(args.md).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f'written: {p}')


if __name__ == '__main__':
    main()
