"""Unit tests for head_fusion.py, on synthetic caches with KNOWN answers.

Two bugs were caught by writing these, both of which would have produced a
clean, plausible, wrong table on real data:

  1. tau was FROZEN at rung C's value during the rho search, so the search could
     not see its own answer -- the true rho scores 50.00 with tau frozen and
     100.00 with tau refitted. The fit settled 50 points short.
  2. Ties broke toward the FIRST grid point (rho = 0.25), so classes whose
     fusion is irrelevant reported a confident rho = 0.25. That reads as
     "this class wants the semantic head" and means nothing.

    python scripts/test_head_fusion.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def write(out, kind, n=90, H=64, NC=3, seed=3):
    """`separable`: only per-class fusion can win. `null`: fusion is irrelevant."""
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    for t in range(n):
        gt = rng.integers(1, NC + 1, size=(H, H)).astype(np.uint8)
        g0 = gt - 1
        sem = rng.uniform(0.02, 0.10, size=(NC, H, H))
        inst = rng.uniform(0.02, 0.10, size=(NC, H, H))
        m0, m1, m2 = g0 == 0, g0 == 1, g0 == 2
        sem[0][m0] = rng.uniform(0.74, 0.76, size=m0.sum())
        inst[1][m1] = rng.uniform(0.54, 0.56, size=m1.sum())
        sem[2][m2] = rng.uniform(0.59, 0.61, size=m2.sum())
        if kind == 'separable':
            # class 2 scores 0.60 and class 1 scores 0.55 on BOTH sets, but the
            # right answer differs -- and class 2's 0.60 comes from a different
            # head each time. No scale and no threshold can separate identical
            # numbers; only rho can, because it moves one head and not the other.
            inst[1][m2] = rng.uniform(0.54, 0.56, size=m2.sum())
            inst[2][m1] = rng.uniform(0.59, 0.61, size=m1.sum())
        else:
            # every class's evidence sits in one head and nothing competes, so
            # `max` is already right everywhere and rho must stay at 1.
            inst[2][m2] = rng.uniform(0.20, 0.30, size=m2.sum())
        pres = np.ones(NC, np.float32)
        lg = np.maximum(sem, inst) * pres[:, None, None]
        top = np.argsort(-lg, axis=0)
        np.savez_compressed(
            out / f'{t:04d}.npz',
            conf=np.take_along_axis(lg, top[:1], 0)[0].astype(np.float16),
            pred=top[0].astype(np.uint8),
            conf2=np.take_along_axis(lg, top[1:2], 0)[0].astype(np.float16),
            pred2=top[1].astype(np.uint8), gt=gt, spres=pres[None, :],
            classes=np.array(['background', 'thing', 'stuff']),
            logits=lg.astype(np.float16),
            sem=sem.astype(np.float16), inst=inst.astype(np.float16))


def run(cache):
    r = subprocess.run(
        [sys.executable, str(HERE / 'head_fusion.py'), '--cache', str(cache),
         '--tau', '0.5', '--folds', '3', '--subsample', '4096', '--rho-rounds', '2'],
        capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit('head_fusion.py failed')
    return r.stdout


def main():
    ok = True
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # -------------------------------------------------- the gate itself
        write(td / 'sep', 'separable')
        out = run(td / 'sep')
        gate = [l for l in out.splitlines() if 'identity gate' in l][0]
        print(gate)
        if 'PASS' not in gate:
            ok = False; print('  ⛔ the identity gate must pass on a single-view cache')

        # -------------- 1. the fit must FIND an answer only fusion can reach
        d = float([l for l in out.splitlines() if 'D − C =' in l][0]
                  .split('D − C =')[1].split('±')[0].strip().replace('**', ''))
        print(f'separable: D − C = {d:+.2f}   (expect ≈ +50)')
        if d < 40:
            ok = False
            print('  ⛔ the fit cannot reach an answer that exists. Check that tau '
                  'is refitted INSIDE the rho search -- freezing it caps this at +0.')

        # ------- 2. and it must NOT invent one where `max` is already right
        write(td / 'null', 'null')
        out2 = run(td / 'null')
        d2 = float([l for l in out2.splitlines() if 'D − C =' in l][0]
                   .split('D − C =')[1].split('±')[0].strip().replace('**', ''))
        rows = [l for l in out2.splitlines() if l.startswith('| background |')
                or l.startswith('| thing |') or l.startswith('| stuff |')]
        rhos = [float(r.split('|')[-2].replace('**', '')) for r in rows]
        print(f'null:      D − C = {d2:+.2f}   ρ = {rhos}   (expect ≈ 0 and all 1.00)')
        if abs(d2) > 1.0:
            ok = False; print('  ⛔ fusion gained where none was available')
        if any(abs(np.log(r)) > np.log(1.01) for r in rhos):
            ok = False
            print('  ⛔ ρ drifted off the published rule on a null. Ties must break '
                  'toward ρ = 1, not toward the first grid point.')

    print('\n✅ all head-fusion tests pass' if ok else '\n⛔ FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
