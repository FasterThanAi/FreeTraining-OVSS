"""Unit tests for presence_power.py on synthetic caches with KNOWN answers.

`separable`: one class is systematically over-gated -- its presence is low on
every tile even though its dense evidence is good -- so it loses the argmax it
should win. Only gamma < 1 for THAT class can fix it: a per-class constant (w)
cannot, because the presence value differs tile to tile, and a threshold cannot,
because the pixel never wins the argmax in the first place.

`null`: presence is 1.0 everywhere, so gamma has no effect at all and must stay
at 1.

    python scripts/test_presence_power.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def write(out, kind, n=90, H=64, NC=3, seed=5):
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    for t in range(n):
        gt = rng.integers(1, NC + 1, size=(H, H)).astype(np.uint8)
        g0 = gt - 1
        fused = rng.uniform(0.02, 0.10, size=(NC, H, H))
        for c in range(NC):                       # each class is right on its own
            fused[c][g0 == c] = rng.uniform(0.70, 0.80, size=(g0 == c).sum())
        if kind == 'separable':
            # A 3-cycle that makes a constant per-class scale PROVABLY unable to
            # help, while gamma fixes it outright.
            #
            #   beta's presence is 1.0 on half the tiles and 0.1 on the other
            #   half; its dense evidence is identical on both. So on low tiles
            #   its own pixels are stolen by background.
            #
            #   w_beta must rise 4.27x to win those back  -> but beta also sits
            #   just under alpha on alpha's pixels, so raising w_beta steals
            #   those -> so w_alpha must rise too -> but alpha sits just under
            #   background on background's pixels, so that steals those.
            #     w_beta > 4.27 w_bg  and  w_alpha > 0.857 w_beta > 3.66 w_bg
            #     w_alpha < 1.167 w_bg
            #   which is a contradiction: NO w works.
            #
            #   gamma_beta = 0 removes the gate for beta alone: low tiles get
            #   0.75 back, high tiles are untouched (1.0 ** anything == 1).
            fused[:] = rng.uniform(0.02, 0.06, size=(NC, H, H))
            fused[0][m := (g0 == 0)] = 0.35; fused[1][m] = 0.30
            fused[1][m := (g0 == 1)] = 0.35; fused[2][m] = 0.30
            fused[2][m := (g0 == 2)] = 0.75; fused[0][m] = 0.32
            pres = np.array([1.0, 1.0, 1.0 if t % 2 else 0.1], np.float32)
        else:
            pres = np.ones(NC, np.float32)
        lg = fused * pres[:, None, None]
        top = np.argsort(-lg, axis=0)
        np.savez_compressed(
            out / f'{t:04d}.npz',
            conf=np.take_along_axis(lg, top[:1], 0)[0].astype(np.float16),
            pred=top[0].astype(np.uint8),
            conf2=np.take_along_axis(lg, top[1:2], 0)[0].astype(np.float16),
            pred2=top[1].astype(np.uint8), gt=gt, spres=pres[None, :],
            classes=np.array(['background', 'alpha', 'beta']),
            logits=lg.astype(np.float16))


def run(cache):
    r = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent / 'presence_power.py'),
         '--cache', str(cache), '--tau', '0.5', '--folds', '3',
         '--subsample', '4096', '--rounds', '2'],
        capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit('presence_power.py failed')
    return r.stdout


def num(out, key):
    return float(out.split(key)[1].split('±')[0].strip().replace('**', ''))


def main():
    ok = True
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        write(td / 'sep', 'separable')
        out = run(td / 'sep')
        gate = [l for l in out.splitlines() if 'identity gate' in l][0]
        print(gate)
        if 'PASS' not in gate:
            ok = False; print('  ⛔ the ungate/re-gate must be exact on a single-view cache')
        d = num(out, 'E − C =')
        beta = [l for l in out.splitlines() if l.startswith('| beta |')][0]
        gb = float(beta.split('|')[-2].replace('**', ''))
        print(f'separable: E − C = {d:+.2f}   γ(beta) = {gb:.2f}   (expect large, γ < 1)')
        if d < 5:
            ok = False
            print('  ⛔ the fit cannot reach an answer that exists — check that tau is '
                  'refitted INSIDE the gamma search')
        if gb >= 1.0:
            ok = False; print('  ⛔ the over-gated class must take γ < 1')

        write(td / 'null', 'null')
        out2 = run(td / 'null')
        d2 = num(out2, 'E − C =')
        gam = [float(l.split('|')[-2].replace('**', '')) for l in out2.splitlines()
               if l.startswith(('| background |', '| alpha |', '| beta |'))]
        print(f'null:      E − C = {d2:+.2f}   γ = {gam}   (expect ≈ 0 and all 1.00)')
        if abs(d2) > 1.0:
            ok = False; print('  ⛔ gained where presence carries no information')
        if any(abs(g - 1.0) > 0.01 for g in gam):
            ok = False
            print('  ⛔ γ drifted off the published rule on a null. Ties must break '
                  'toward γ = 1.')
    print('\n✅ all presence-power tests pass' if ok else '\n⛔ FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
