"""
WEEK1_RESULTS 9.1a — how much of the 323M-pixel residual is ADDRESSABLE?

The headline "29.68% of real-class pixels discarded" is an upper bound on what a
region-level co-occurrence prior can recover, not an estimate of it. The
qualitative panels show two different morphologies:

    docs/2524.png  19.0% discard — whole contiguous regions dropped  -> addressable
    docs/2522.png  21.0% discard — thin seams along class boundaries -> NOT addressable

A region-level prior assigns labels to REGIONS. Boundary-seam pixels are the
seams BETWEEN regions; there is no atom to assign them to, and they are also
where GT annotation is least reliable. Counting them in the headline overstates
the opportunity.

This splits the discard by distance to the nearest class boundary.

    python scripts/discard_boundary_decomposition.py \
        --cache ~/outputs/week2_tau0.5_instrumented/cache --tau 0.5

Method. The boundary seam is defined exactly as in ANALYSIS 4 /
cooccurrence_gt.py: a pixel whose 4-neighbourhood contains a different GT label.
Band of width k = the seam dilated k-1 times. Deliberately NOT erosion of each
class mask, which would count image-edge pixels as boundary and inflate the
result.

The number that matters is the ENRICHMENT ratio: the share of DISCARD falling in
the band, against the share of AREA the band occupies. Enrichment ~1 means
discard is spread uniformly and the band tells us nothing. Enrichment >> 1 means
discard concentrates on seams and the addressable residual is materially smaller
than 29.68%.

Cache layout (measure_discard_rate.py):
    conf float16 HxW  max over classes, PRE-threshold
    pred uint8   HxW  argmax, 0-indexed, 0 = background
    gt   uint8   HxW  0 = no-data, 1 = background, 2..7 = real classes

A pixel is assigned to background iff (conf < tau) or (pred == 0) -- both
mechanisms of 7.7, since either route ends at background.
"""
import argparse
from pathlib import Path

import numpy as np

CLASSES = ['background', 'building', 'road', 'water',
           'barren', 'forest', 'agricultural']
N = len(CLASSES)


def seam(gt):
    """Pixels adjacent (4-conn) to a different GT label. Same definition as
    ANALYSIS 4's shared-boundary-length adjacency."""
    b = np.zeros(gt.shape, dtype=bool)
    d = gt[:-1, :] != gt[1:, :]
    b[:-1, :] |= d
    b[1:, :] |= d
    d = gt[:, :-1] != gt[:, 1:]
    b[:, :-1] |= d
    b[:, 1:] |= d
    return b


def dilate(m, k):
    """Dilate a boolean mask k times, 4-connectivity."""
    for _ in range(k):
        o = m.copy()
        o[:-1, :] |= m[1:, :]
        o[1:, :] |= m[:-1, :]
        o[:, :-1] |= m[:, 1:]
        o[:, 1:] |= m[:, :-1]
        m = o
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--widths', type=int, nargs='+', default=[1, 2, 3, 5, 10])
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default=None, help='write a markdown summary here')
    args = ap.parse_args()

    files = sorted(Path(args.cache).expanduser().glob('*.npz'))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f'no .npz files under {args.cache}')
    W = sorted(args.widths)
    print(f'{len(files)} cached tiles | tau = {args.tau} | band widths {W}\n')

    real_px = 0
    disc_px = 0
    band_area = {k: 0 for k in W}          # real-class pixels inside the band
    band_disc = {k: 0 for k in W}          # discarded pixels inside the band
    per_cls = {k: np.zeros((N, 2), np.int64) for k in W}   # [class] x [band area, band discard]
    cls_tot = np.zeros((N, 2), np.int64)                   # [class] x [gt px, total discard]

    for i, f in enumerate(files, 1):
        z = np.load(f)
        gt, conf, pred = z['gt'], z['conf'].astype(np.float32), z['pred']

        real = gt >= 2
        if not real.any():
            continue
        bg = (conf < args.tau) | (pred == 0)      # both mechanisms of 7.7
        disc = real & bg

        real_px += int(real.sum())
        disc_px += int(disc.sum())

        for c in range(2, N + 1):
            cm = gt == c
            if cm.any():
                cls_tot[c - 1, 0] += int(cm.sum())
                cls_tot[c - 1, 1] += int((disc & cm).sum())

        s = seam(gt)
        m = s
        prev = 0
        for k in W:
            m = dilate(m, k - 1 - prev) if k - 1 - prev > 0 else m
            prev = k - 1
            inb = real & m
            band_area[k] += int(inb.sum())
            band_disc[k] += int((disc & m).sum())
            for c in range(2, N + 1):
                cm = gt == c
                per_cls[k][c - 1, 0] += int((cm & m).sum())
                per_cls[k][c - 1, 1] += int((disc & cm & m).sum())

        if i % 250 == 0 or i == len(files):
            print(f'  {i}/{len(files)}')

    md = ['# 9.1a — Boundary vs interior decomposition of the residual\n',
          f'- tiles: **{len(files)}**  |  τ: **{args.tau}**',
          f'- real-class pixels: **{real_px:,}**',
          f'- assigned to background: **{disc_px:,}** '
          f'({100*disc_px/max(real_px,1):.2f}%)\n',
          '`band` = within k pixels of a GT class boundary (4-conn seam, dilated).',
          '`enrichment` = (share of discard in band) / (share of area in band).',
          'Enrichment ≈ 1 means discard is spread uniformly and the band explains nothing.\n',
          '| band width k | band share of area | band share of discard | enrichment | '
          'interior discard (addressable) |',
          '|---|---|---|---|---|']
    for k in W:
        a = 100 * band_area[k] / max(real_px, 1)
        d = 100 * band_disc[k] / max(disc_px, 1)
        e = (d / a) if a > 0 else float('nan')
        interior = disc_px - band_disc[k]
        md.append(f'| {k} | {a:.2f}% | {d:.2f}% | **{e:.2f}×** | '
                  f'{interior:,} ({100*interior/max(real_px,1):.2f}% of real-class) |')

    kmax = W[-1]
    interior = disc_px - band_disc[kmax]
    md += [f'\n## Per class, at k={kmax}\n',
           'Which classes lose their pixels on seams, and which lose whole regions.\n',
           '| Class | total discard | in band | in interior | band share of its discard | '
           'enrichment |',
           '|---|---|---|---|---|---|']
    for c in range(1, N):
        t_d = int(cls_tot[c, 1])
        b_d = int(per_cls[kmax][c, 1])
        b_a = int(per_cls[kmax][c, 0])
        gt_c = int(cls_tot[c, 0])
        if t_d == 0 or gt_c == 0:
            md.append(f'| {CLASSES[c]} | 0 | — | — | — | — |')
            continue
        share_d = 100 * b_d / t_d
        share_a = 100 * b_a / gt_c
        enr = (share_d / share_a) if share_a > 0 else float('nan')
        md.append(f'| {CLASSES[c]} | {t_d:,} | {b_d:,} | {t_d-b_d:,} | '
                  f'{share_d:.1f}% | **{enr:.2f}×** |')
    md.append('\nA class with high enrichment loses its pixels on seams (annotation-boundary '
              'effects, thin structures). A class near 1.00× loses whole regions — that is the '
              'population a region-level prior can actually recover.')

    a = 100 * band_area[kmax] / max(real_px, 1)
    d = 100 * band_disc[kmax] / max(disc_px, 1)
    e = (d / a) if a > 0 else float('nan')
    md += ['\n## Verdict\n']
    if e > 1.5:
        md.append(f'At k={kmax}, discard is **{e:.2f}× enriched** on class boundaries. The '
                  f'residual is materially seam-shaped, so **{interior:,}** interior pixels '
                  f'({100*interior/max(real_px,1):.2f}% of real-class) is the honest addressable '
                  'figure for a region-level prior — not 29.68%. Lead with the addressable '
                  'number and give 29.68% as the gross residual.')
    elif e > 1.15:
        md.append(f'At k={kmax}, enrichment is **{e:.2f}×** — mild. Boundaries carry somewhat '
                  'more discard than their area share, but most of the residual is interior. '
                  'Report both; the headline survives with a sentence of qualification.')
    else:
        md.append(f'At k={kmax}, enrichment is **{e:.2f}×** — discard is spread essentially '
                  'uniformly. The seam/region distinction visible in docs/2522.png vs 2524.png '
                  'does NOT generalise, and 29.68% stands as the addressable figure. '
                  'A negative result, and it removes a caveat rather than adding one.')
    md += ['\n> Note the band grows fast: at k=10 a large share of a tile is within 10px of '
           'some boundary, so a high band share at large k is partly definitional. Read the '
           'enrichment column, which normalises for exactly that, not the raw share.\n']

    text = '\n'.join(md)
    print('\n' + text)
    if args.out:
        Path(args.out).expanduser().write_text(text)
        print(f'\nWritten to {args.out}')


if __name__ == '__main__':
    main()
