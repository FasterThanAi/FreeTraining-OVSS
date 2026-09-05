"""
Why does a VECTOR threshold miss when a SCALAR one reproduces exactly?

THE OBSERVATION. On Potsdam, `eval.py` reproduces the cached prediction to 0.03
mIoU at the published scalar tau, and every class agrees to within 0.08. Switch to
a fitted per-class vector and three of six classes miss by 2-3 IoU, always low,
while the other three stay exact. The same shape appears under the scale.

⚠️ A scalar threshold is INVARIANT to class ordering; a vector one is not. So a
mismatch that appears only with a vector points at indexing, not at arithmetic --
but that is a hypothesis, and this script exists to test it rather than assume it.

WHAT IT SEPARATES. There are exactly two places the disagreement can live:

  (1) the HISTOGRAM ARITHMETIC -- `confusion_at` summing a binned histogram is not
      the same as applying the rule per pixel on this data. Bin edges, float16
      `conf` near a boundary, or the catch-all's special case.

  (2) the CACHE vs THE PIPELINE -- the arithmetic is right, but the cached
      (gt, pred, conf) is not what eval.py actually computes.

This tests (1) directly: apply the segmentor's rule PER PIXEL in numpy, with no
binning anywhere, and compare against `confusion_at` on the same tiles and the
same tau vector. If they differ, the histogram is the problem and every cached
sweep in the project inherits it. If they agree to the pixel, the histogram is
exonerated and the cache is the place to look.

⭐ Both outcomes matter beyond this run. verify_perclass_tau.py already proves the
two agree on RANDOM data; this asks whether they agree on REAL data, where the
confidence distribution is concentrated rather than uniform.

    python scripts/diagnose_vector_tau.py \\
        --cache ~/outputs/potsdam_full/cache \\
        --split ~/splits/potsdam_reorder_heldout.txt \\
        --tau 0.110 0.130 0.060 0.020 0.490 0.110 --published 0.1
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels                                                     # noqa: E402
from tau_oracle import confusion_at, per_class_iou, miou, NBINS   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--split', help='file of tile stems, one per line; default all')
    ap.add_argument('--tau', type=float, nargs='+', required=True,
                    help='the per-class vector, in class order')
    ap.add_argument('--published', type=float, required=True,
                    help='the scalar baseline, as a control that must agree')
    args = ap.parse_args()

    LB = labels.from_cache(args.cache)
    nc, bg = LB.n, LB.bg - 1
    print(f'  classes: {LB}')
    if len(args.tau) != nc:
        raise SystemExit(f'--tau has {len(args.tau)} entries, need {nc}')

    files = sorted(Path(args.cache).expanduser().glob('*.npz'))
    if args.split:
        keep = set(Path(args.split).expanduser().read_text().split())
        files = [f for f in files if f.stem in keep]
        print(f'  restricted to the split: {len(files)} tiles')
    if not files:
        raise SystemExit('no tiles')

    tv = np.asarray(args.tau, np.float64)
    ts = np.full(nc, args.published, np.float64)

    # direct, per pixel, no binning anywhere
    Cd_v = np.zeros((nc, nc), np.int64)
    Cd_s = np.zeros((nc, nc), np.int64)
    H = np.zeros((nc, nc, NBINS), np.int64)

    for i, f in enumerate(files):
        z = np.load(f)
        gt = z['gt'].astype(np.int32)
        m = gt > 0
        if not m.any():
            continue
        g = gt[m] - 1
        pred = np.clip(z['pred'].astype(np.int32)[m], 0, nc - 1)
        conf = z['conf'].astype(np.float32)[m]

        for C, taus in ((Cd_v, tv), (Cd_s, ts)):
            out = pred.copy()
            # exactly predict(): thd = tau[pred]; pred[conf < thd] = bg
            out[conf < taus[pred]] = bg
            np.add.at(C, (g, out), 1)

        b = np.clip((conf * NBINS).astype(np.int32), 0, NBINS - 1)
        np.add.at(H, (g, pred, b), 1)
        if (i + 1) % 250 == 0 or i + 1 == len(files):
            print(f'  {i + 1}/{len(files)}')

    Ch_v = confusion_at(H, tv, bg, NBINS)
    Ch_s = confusion_at(H, ts, bg, NBINS)

    print('\n' + '=' * 74)
    print(f'{"class":<16}{"direct":>10}{"histogram":>12}{"diff":>9}   '
          f'{"scalar dir":>11}{"scalar hist":>12}{"diff":>9}')
    dv, hv = per_class_iou(Cd_v), per_class_iou(Ch_v)
    ds, hs = per_class_iou(Cd_s), per_class_iou(Ch_s)
    for c in range(nc):
        tag = ' *' if c == bg else ''
        print(f'{LB.names[c] + tag:<16}{dv[c]:>10.2f}{hv[c]:>12.2f}{dv[c] - hv[c]:>+9.2f}   '
              f'{ds[c]:>11.2f}{hs[c]:>12.2f}{ds[c] - hs[c]:>+9.2f}')
    print('-' * 74)
    print(f'{"mIoU":<16}{miou(Cd_v):>10.2f}{miou(Ch_v):>12.2f}'
          f'{miou(Cd_v) - miou(Ch_v):>+9.2f}   '
          f'{miou(Cd_s):>11.2f}{miou(Ch_s):>12.2f}'
          f'{miou(Cd_s) - miou(Ch_s):>+9.2f}')
    print('=' * 74)

    same_v = np.array_equal(Cd_v, Ch_v)
    same_s = np.array_equal(Cd_s, Ch_s)
    dv_px = int(np.abs(Cd_v - Ch_v).sum())
    print()
    if same_v and same_s:
        print('✅ The histogram and the per-pixel rule agree EXACTLY, for both the')
        print('   vector and the scalar. The arithmetic is exonerated, so the cached')
        print('   (gt, pred, conf) is where eval.py and this project part company.')
        print('   Next: the cache is written from the tensors predict() returns, so')
        print('   compare a single tile end to end -- run one image through eval.py')
        print('   with the vector config and diff its pred_sem_seg against the rule')
        print('   applied to that tile\'s cached arrays.')
    else:
        print(f'⛔ The histogram and the per-pixel rule DISAGREE on {dv_px:,} pixels')
        print(f'   (vector: {"same" if same_v else "differs"}, '
              f'scalar: {"same" if same_s else "differs"}).')
        print('   The histogram is the problem, and EVERY cached tau sweep in this')
        print('   project inherits it -- the oracle bounds, the cross-validation and')
        print('   both published gains. Fix before anything else.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
