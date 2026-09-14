"""scale_transfer.py on synthetic full caches whose answer is worked out by hand.

The construction. Three classes: catch-all `bg`, plus `a` and `b`. Every tile holds
four pixel groups with hand-chosen score stacks:

  g1  3000 px  gt=a   scores [bg .10, a .60, b .70]   -> a LOSES the argmax to b
  g2  3000 px  gt=b   scores [bg .10, a .20, b .80]   -> b wins, correctly
  g3  2000 px  gt=bg  scores [bg .90, a .10, b .10]   -> catch-all wins
  g4   500 px  gt=a   scores [bg .10, a .90, b .10]   -> a wins at any scale in grid

⭐ g1 is the point. `a` is right and loses anyway, so **no threshold can recover it** --
lowering a's tau cannot take a pixel b already won. Only a scale before the argmax can:
0.60 x 1.201 = 0.721 > 0.700. That is exactly the error mass lever 2 claims.

  SAME      destination drawn the same way -> the source (w, tau) must be clearly POSITIVE
            over lever 1 alone.
  OPPOSITE  g1's label flipped to b, so the same scale now converts 3000 correct pixels
            into false positives -> must be clearly NEGATIVE. A script that cannot
            produce a negative here cannot be trusted with a positive anywhere.
  GATES     differing class list, shared filename, wrong --expect: each must ABORT.
  DEPLOY    the emitted config must carry a 3-entry class_scale and prob_thd.
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CLASSES = ['bg', 'a', 'b']
#       n     gt   bg    a     b
SAME = [(3000, 2, (.10, .60, .70)),
        (3000, 3, (.10, .20, .80)),
        (2000, 1, (.90, .10, .10)),
        (500,  2, (.10, .90, .10))]
OPP = [(3000, 3, (.10, .60, .70)),      # <- only this line differs
       (3000, 3, (.10, .20, .80)),
       (2000, 1, (.90, .10, .10)),
       (500,  2, (.10, .90, .10))]


def write(path, spec, classes=CLASSES):
    tot = sum(n for n, _, _ in spec)
    side = int(np.ceil(np.sqrt(tot)))
    gt = np.zeros(side * side, np.uint8)
    lg = np.zeros((len(classes), side * side), np.float16)
    i = 0
    for n, g, sc in spec:
        gt[i:i + n] = g
        for c, v in enumerate(sc):
            lg[c, i:i + n] = v
        i += n
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, gt=gt.reshape(side, side),
                        logits=lg.reshape(len(classes), side, side),
                        classes=np.array(classes))


def run(src, dst, extra=()):
    return subprocess.run(
        [sys.executable, str(HERE / 'scale_transfer.py'),
         '--src-cache', str(src), '--dst-cache', str(dst), '--tau', '0.3',
         '--objective', 'real', '--subsample', '4000', '--perms', '12', *extra],
        capture_output=True, text=True)


def lever2(out):
    m = re.search(r'Lever 2 over lever 1, on held-out tiles: ([+-][0-9.]+)', out)
    if not m:
        raise AssertionError('lever-2 line missing from output')
    return float(m.group(1))


def main():
    tmp = Path(tempfile.mkdtemp(prefix='scale_transfer_test_'))
    fail = 0

    def check(ok, msg):
        nonlocal fail
        fail += not ok
        print(f'{"ok  " if ok else "FAIL"}  {msg}')

    try:
        for i in range(2):
            write(tmp / 'src' / f's{i}.npz', SAME)
            write(tmp / 'same' / f'd{i}.npz', SAME)
            write(tmp / 'opp' / f'd{i}.npz', OPP)

        r = run(tmp / 'src', tmp / 'same')
        check(r.returncode == 0, f'SAME runs (rc={r.returncode})\n{r.stderr[-500:]}')
        if r.returncode == 0:
            v = lever2(r.stdout)
            check(v > 5.0, f'SAME: lever 2 transfers strongly positive ({v:+.2f}) — '
                           f'the argmax-lost pixels are recovered')
            check('Lever 2 transfers' in r.stdout, 'SAME: verdict is positive')
            check('0.0% of `w` shuffles' in r.stdout or 'Only 0.0%' in r.stdout,
                  'SAME: no permutation of w matches the real assignment')

        r = run(tmp / 'src', tmp / 'opp')
        check(r.returncode == 0, f'OPPOSITE runs (rc={r.returncode})')
        if r.returncode == 0:
            v = lever2(r.stdout)
            check(v < -5.0, f'OPPOSITE: lever 2 is strongly negative ({v:+.2f})')
            check('does NOT transfer' in r.stdout, 'OPPOSITE: verdict says it fails')

        write(tmp / 'other' / 'd0.npz', SAME, classes=['bg', 'a', 'zzz'])
        r = run(tmp / 'src', tmp / 'other')
        check(r.returncode != 0 and 'class lists differ' in r.stdout + r.stderr,
              'GATE: differing class list aborts')

        write(tmp / 'overlap' / 's0.npz', SAME)
        r = run(tmp / 'src', tmp / 'overlap')
        check(r.returncode != 0 and 'share filenames' in r.stdout + r.stderr,
              'GATE: shared filename aborts')

        r = run(tmp / 'src', tmp / 'same', ('--expect', '99.0'))
        check(r.returncode != 0 and 'GATE FAILED' in r.stdout + r.stderr,
              'GATE: wrong --expect aborts')

        # a cache written WITHOUT --cache-full must be refused, not silently skipped
        d = tmp / 'nologits'
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / 'x.npz', gt=np.ones((4, 4), np.uint8),
                            conf=np.zeros((4, 4), np.float16),
                            pred=np.zeros((4, 4), np.uint8),
                            classes=np.array(CLASSES))
        r = run(tmp / 'src', d)
        check(r.returncode != 0 and 'no `logits` key' in r.stdout + r.stderr,
              'GATE: a cache without --cache-full is refused')

        # deploy path: the emitted config must carry both vectors, correct length
        base = tmp / 'base_cfg.py'
        base.write_text("model = dict(type='X', prob_thd=0.3)\n")
        out = tmp / 'deployed.py'
        r = run(tmp / 'src', tmp / 'same',
                ('--base-cfg', str(base), '--deploy-cfg', str(out)))
        try:
            import mmengine  # noqa: F401
            have_mm = True
        except ImportError:
            have_mm = False
        if not have_mm:
            print('skip  DEPLOY: mmengine not installed here (it is in the '
                  'segov3 env on the workstation, where this path runs)')
        check(have_mm is False or (r.returncode == 0 and out.exists()),
              'DEPLOY: config written')
        if have_mm and out.exists():
            txt = out.read_text()
            ns = {}
            exec(compile(txt, str(out), 'exec'), ns)
            w = ns['model']['class_scale']
            t = ns['model']['prob_thd']
            check(len(w) == 3 and len(t) == 3,
                  f'DEPLOY: both vectors have one entry per class ({len(w)}, {len(t)})')
            check(all(x > 0 for x in w), f'DEPLOY: every scale is positive ({w})')
            check(abs(t[0] - 0.3) < 1e-9,
                  f'DEPLOY: the catch-all keeps the published τ ({t[0]})')
            check('expect eval.py to report mIoU' in r.stdout,
                  'DEPLOY: the expected eval.py number is printed')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print('ALL PASS' if not fail else f'{fail} FAILURE(S)')
    return 1 if fail else 0


if __name__ == '__main__':
    sys.exit(main())
