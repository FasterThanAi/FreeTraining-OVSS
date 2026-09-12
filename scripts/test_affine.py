"""Unit tests for affine_reorder.py, on synthetic caches with KNOWN answers.

`separable`: a case no per-class SCALE can solve and a bias solves outright.
`beta` must WIN where scores are small and LOSE where they are large:

    low-score pixels  (gt = beta):  beta 0.10, alpha 0.12  -> beta must win
    high-score pixels (gt = alpha): beta 0.80, alpha 0.90  -> alpha must win

    w needs  w_b*0.10 > w_a*0.12  ->  w_b/w_a > 1.200
    and      w_b*0.80 < w_a*0.90  ->  w_b/w_a < 1.125      CONTRADICTION

    b_beta = +0.05 does both: 0.15 > 0.12 and 0.85 < 0.90.

⚠️ THE SOLUTION IS NOT UNIQUE, and the test must not demand one spelling of it.
b_alpha = -0.05 produces the identical reordering (0.07 < 0.10 and 0.85 > 0.80),
and it is what the fit actually returns, because the coordinate sweep reaches
`alpha` first and `beta` then has nothing left to do. Only the DIFFERENCE
b_beta - b_alpha decides the argmax, so that is what is asserted. A first version
of this test required b_beta > 0 and failed a correct fit.

A threshold cannot help either -- thresholding sends a pixel to the catch-all,
not to `beta`. So a positive result here is attributable to the bias alone.

`null`: scores are cleanly separated, so b must stay at 0.

    python scripts/test_affine.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def write(out, kind, n=90, H=64, NC=3, seed=7):
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    for t in range(n):
        gt = rng.integers(1, NC + 1, size=(H, H)).astype(np.uint8)
        g0 = gt - 1
        lg = rng.uniform(0.01, 0.04, size=(NC, H, H))
        lg[0][g0 == 0] = rng.uniform(0.70, 0.75, size=(g0 == 0).sum())
        if kind == 'separable':
            m1, m2 = g0 == 1, g0 == 2
            lg[1][m1] = 0.90; lg[2][m1] = 0.80      # high regime: alpha must win
            lg[1][m2] = 0.12; lg[2][m2] = 0.10      # low regime:  beta must win
        else:
            lg[1][g0 == 1] = rng.uniform(0.70, 0.75, size=(g0 == 1).sum())
            lg[2][g0 == 2] = rng.uniform(0.70, 0.75, size=(g0 == 2).sum())
        top = np.argsort(-lg, axis=0)
        np.savez_compressed(
            out / f'{t:04d}.npz',
            conf=np.take_along_axis(lg, top[:1], 0)[0].astype(np.float16),
            pred=top[0].astype(np.uint8),
            conf2=np.take_along_axis(lg, top[1:2], 0)[0].astype(np.float16),
            pred2=top[1].astype(np.uint8), gt=gt,
            spres=np.ones((1, NC), np.float32),
            classes=np.array(['background', 'alpha', 'beta']),
            logits=lg.astype(np.float16))


def run(cache):
    r = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent / 'affine_reorder.py'),
         '--cache', str(cache), '--tau', '0.5', '--folds', '3',
         '--subsample', '4096', '--rounds', '2'],
        capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit('affine_reorder.py failed')
    return r.stdout


def main():
    ok = True
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        write(td / 'sep', 'separable')
        out = run(td / 'sep')
        d = float(out.split('D − C =')[1].split('±')[0].strip().replace('**', ''))
        row = lambda n: float([l for l in out.splitlines()
                               if l.startswith(f'| {n} |')][0]
                              .split('|')[-2].replace('**', ''))
        rel = row('beta') - row('alpha')
        print(f'separable: D − C = {d:+.2f}   b(beta) − b(alpha) = {rel:+.2f}   '
              f'(expect large, and the difference > 0)')
        if d < 5:
            ok = False
            print('  ⛔ the fit cannot reach an answer no scale can express. Check '
                  '(a) that tau is refitted INSIDE the b search, and (b) that w is '
                  'gauge-fixed to geometric mean 1 before b is fitted — a uniform '
                  'rescaling of w is argmax-equivalent but changes what b means.')
        if rel <= 0:
            ok = False
            print('  ⛔ beta must be favoured RELATIVE to alpha in the low-score '
                  'regime; either b_beta > 0 or b_alpha < 0 will do.')

        write(td / 'null', 'null')
        out2 = run(td / 'null')
        d2 = float(out2.split('D − C =')[1].split('±')[0].strip().replace('**', ''))
        bs = [float(l.split('|')[-2].replace('**', '')) for l in out2.splitlines()
              if l.startswith(('| background |', '| alpha |', '| beta |'))]
        print(f'null:      D − C = {d2:+.2f}   b = {bs}   (expect ≈ 0 and all 0.00)')
        if abs(d2) > 1.0:
            ok = False; print('  ⛔ gained where no reordering was available')
        if any(abs(x) > 1e-9 for x in bs):
            ok = False
            print('  ⛔ b drifted off 0 on a null. Ties must break toward b = 0.')
    print('\n✅ all affine tests pass' if ok else '\n⛔ FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
