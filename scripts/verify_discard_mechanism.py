"""
WEEK1_RESULTS 7.6 — is "assigned to background" the same set as "discarded by tau"?

A real-class pixel can end up labelled background two different ways
(segearthov3_segmentor.predict):

    seg_pred = argmax(seg_logits)              # background is one of the 7 queries
    seg_pred[max_vals < prob_thd] = bg_idx     # the tau rule

  (A) THRESHOLD  conf < tau                 -> forced to background
  (B) ARGMAX     conf >= tau and pred == bg -> background genuinely won

7.6 hedged that (A) is a subset of (A|B) and asked for verification.
S_pres(background) has median 0.0220, and since P_fused <= 1 we get
P_final(bg) <= S_pres, so background usually cannot reach tau = 0.5 at all --
but 26/1669 tiles do have S_pres(background) >= 0.5 (max 0.8750), so the
argument is not universal.

This measures it exactly instead of arguing, using the .npz cache:
seconds on CPU, no GPU, no re-inference. If (B) is negligible then
"discarded by tau" is accurate at this operating point and the hedge can go.

    python scripts/verify_discard_mechanism.py \
        --cache ~/outputs/week2_tau0.5_instrumented/cache --tau 0.5

Cache layout (see measure_discard_rate.py):
    conf  float16 HxW   max over classes of seg_logits, PRE-threshold
    pred  uint8   HxW   argmax over classes, 0-indexed, 0 = background
    gt    uint8   HxW   0 = no-data, 1 = background, 2..7 = real classes
"""
import argparse
from pathlib import Path

import numpy as np

CLASSES = ['background', 'building', 'road', 'water',
           'barren', 'forest', 'agricultural']
N = len(CLASSES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    files = sorted(Path(args.cache).expanduser().glob('*.npz'))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f'no .npz files under {args.cache}')
    print(f'{len(files)} cached tiles | tau = {args.tau}\n')

    n_real = n_thr = n_arg = 0
    per_class = np.zeros((N, 2), dtype=np.int64)     # [class] x [threshold, argmax]
    tiles_with_argmax = 0

    for i, f in enumerate(files, 1):
        z = np.load(f)
        gt, conf, pred = z['gt'], z['conf'].astype(np.float32), z['pred']

        real = gt >= 2                                # 1 = background, 0 = no-data
        below = conf < args.tau
        thr = real & below                            # (A) forced by the tau rule
        arg = real & ~below & (pred == 0)             # (B) background won the argmax

        n_real += int(real.sum())
        n_thr += int(thr.sum())
        n_arg += int(arg.sum())
        if arg.any():
            tiles_with_argmax += 1
            for c in range(2, N + 1):
                m = gt == c
                per_class[c - 1, 1] += int((arg & m).sum())
        for c in range(2, N + 1):
            m = gt == c
            per_class[c - 1, 0] += int((thr & m).sum())

        if i % 250 == 0 or i == len(files):
            print(f'  {i}/{len(files)}')

    total_bg = n_thr + n_arg
    print('\n' + '=' * 68)
    print(f'real-class pixels                     {n_real:>16,}')
    print(f'assigned to background (A or B)       {total_bg:>16,}'
          f'  ({100*total_bg/max(n_real,1):.2f}%)')
    print(f'  (A) threshold  conf < tau           {n_thr:>16,}'
          f'  ({100*n_thr/max(total_bg,1):.4f}% of those)')
    print(f'  (B) argmax     background won       {n_arg:>16,}'
          f'  ({100*n_arg/max(total_bg,1):.4f}% of those)')
    print(f'tiles with any (B) pixel              {tiles_with_argmax:>16,} / {len(files)}')
    print('=' * 68)

    if n_arg:
        print('\nper class, mechanism (B) only:')
        for c in range(1, N):
            if per_class[c, 1]:
                print(f'  {CLASSES[c]:<14} {per_class[c,1]:>14,}')

    frac = 100 * n_arg / max(total_bg, 1)
    print()
    if frac < 0.1:
        print(f'VERDICT: mechanism (B) accounts for {frac:.4f}% of background assignments.')
        print('"Discarded by tau" is accurate at this operating point. 7.6\'s hedge can go.')
    elif frac < 2:
        print(f'VERDICT: (B) is {frac:.2f}% -- small but not nil. "Assigned to background" is')
        print('the honest phrasing; quote this number when using "discarded by tau".')
    else:
        print(f'VERDICT: (B) is {frac:.2f}% -- NOT negligible. The two sets differ materially.')
        print('Keep 7.6\'s hedge and report both mechanisms separately.')
    print('\nNote: this argument is tau-specific. It rests on S_pres(background) < tau,')
    print('which fails as tau approaches 0.02 -- do not carry it to the tau=0.1 sweep.')


if __name__ == '__main__':
    main()
