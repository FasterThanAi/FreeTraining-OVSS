"""tau_transfer.py on synthetic caches whose answer is known before the run.

Three constructions, all deterministic:

  SAME      src and dst share a generative process. The source-fitted vector must
            be strongly POSITIVE on dst -- if it is not, the script cannot detect
            transfer even when transfer is trivially available.
  OPPOSITE  dst inverts which pixels are correct, so the source vector must be
            strongly NEGATIVE. A script that reports a gain here would report one
            on anything.
  GATES     a class-list mismatch, an overlapping filename and a wrong --expect
            must each ABORT rather than produce a number.

Confidence values are chosen exactly representable in float16 and away from bin
edges (0.25, 0.625, 0.75, 0.9375), because the cache stores float16 and a value
landing on a grid edge makes the expected thresholds off by one bin.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CLASSES = ['background', 'a', 'b']

#            n     gt  pred  conf      what
SRC = [(3000,  2,   1, 0.25),     # class a, correct, below published tau -> recoverable
       (3000,  2,   1, 0.75),     # class a, correct, kept
       (2000,  1,   2, 0.625),    # class b, FALSE, kept -> removable by raising tau_b
       (2000,  3,   2, 0.9375),   # class b, correct, kept
       (2000,  1,   0, 0.9375)]   # true catch-all
OPP = [(3000,  1,   1, 0.25),     # now FALSE -> lowering tau_a HURTS
       (3000,  2,   1, 0.75),
       (2000,  3,   2, 0.625),    # now CORRECT -> raising tau_b HURTS
       (2000,  3,   2, 0.9375),
       (2000,  1,   0, 0.9375)]


def write(path, spec, classes=CLASSES):
    total = sum(n for n, _, _, _ in spec)
    side = int(np.ceil(np.sqrt(total)))
    gt = np.zeros(side * side, np.uint8)
    pred = np.zeros(side * side, np.uint8)
    conf = np.zeros(side * side, np.float16)
    i = 0
    for n, g, p, c in spec:
        gt[i:i + n], pred[i:i + n], conf[i:i + n] = g, p, c
        i += n
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, gt=gt.reshape(side, side), pred=pred.reshape(side, side),
                        conf=conf.reshape(side, side), classes=np.array(classes))


def run(src, dst, extra=()):
    cmd = [sys.executable, str(HERE / 'tau_transfer.py'),
           '--src-cache', str(src), '--dst-cache', str(dst),
           '--tau', '0.5', '--objective', 'real',
           '--perms', '20', '--repeats', '1', '--sizes', '2', *extra]
    return subprocess.run(cmd, capture_output=True, text=True)


def arm_delta(out, marker):
    for ln in out.splitlines():
        if marker in ln and '|' in ln:
            cells = [c.strip() for c in ln.split('|')]
            return float(cells[4].replace('*', ''))
    raise AssertionError(f'arm {marker!r} not found in output')


def main():
    tmp = Path(tempfile.mkdtemp(prefix='tau_transfer_test_'))
    fail = 0

    def check(ok, msg):
        nonlocal fail
        fail += not ok
        print(f'{"ok  " if ok else "FAIL"}  {msg}')

    try:
        for i in range(2):
            write(tmp / 'src' / f's{i}.npz', SRC)
            write(tmp / 'same' / f'd{i}.npz', SRC)
            write(tmp / 'opp' / f'd{i}.npz', OPP)

        # ---- SAME: transfer must be clearly positive ----------------------
        r = run(tmp / 'src', tmp / 'same')
        check(r.returncode == 0, f'SAME runs (rc={r.returncode})\n{r.stderr[-400:]}')
        if r.returncode == 0:
            c = arm_delta(r.stdout, 'C  per-class τ fitted on src')
            b = arm_delta(r.stdout, 'B  global τ fitted on src')
            d = arm_delta(r.stdout, 'D  per-class oracle on dst')
            check(c > 5.0, f'SAME: per-class transfer is strongly positive ({c:+.2f})')
            check(abs(c - d) < 0.5, f'SAME: transfer reaches the dst oracle '
                                    f'({c:+.2f} vs {d:+.2f})')
            check(c > b, f'SAME: per-class beats a global τ ({c:+.2f} > {b:+.2f})')
            check('Transfer works' in r.stdout or 'Transfer is positive' in r.stdout,
                  'SAME: verdict reports a positive transfer')

        # ---- OPPOSITE: transfer must be clearly negative -------------------
        r = run(tmp / 'src', tmp / 'opp')
        check(r.returncode == 0, f'OPPOSITE runs (rc={r.returncode})')
        if r.returncode == 0:
            c = arm_delta(r.stdout, 'C  per-class τ fitted on src')
            check(c < -5.0, f'OPPOSITE: transfer is strongly negative ({c:+.2f})')
            check('Transfer FAILS' in r.stdout, 'OPPOSITE: verdict says it fails')

        # ---- GATE: class-list mismatch must abort --------------------------
        write(tmp / 'other' / 'd0.npz', SRC, classes=['background', 'a', 'zzz'])
        r = run(tmp / 'src', tmp / 'other')
        check(r.returncode != 0 and 'class lists differ' in (r.stdout + r.stderr),
              'GATE: a differing class list aborts')

        # ---- GATE: overlapping filenames must abort ------------------------
        write(tmp / 'overlap' / 's0.npz', SRC)
        r = run(tmp / 'src', tmp / 'overlap')
        check(r.returncode != 0 and 'share filenames' in (r.stdout + r.stderr),
              'GATE: a shared filename aborts (it would be a leak)')

        # ---- GATE: a wrong --expect must abort -----------------------------
        r = run(tmp / 'src', tmp / 'same', ('--expect', '99.0'))
        check(r.returncode != 0 and 'GATE FAILED' in (r.stdout + r.stderr),
              'GATE: a wrong --expect aborts')
        r = run(tmp / 'src', tmp / 'same', ('--expect', '42.86', '--expect-tol', '0.2'))
        check(r.returncode == 0, 'GATE: the correct --expect passes')

        # ---- exclusion actually removes files ------------------------------
        write(tmp / 'src' / 's_flipped9.npz', OPP)
        r = run(tmp / 'src', tmp / 'same', ('--src-exclude-re', '_flipped'))
        check(r.returncode == 0 and 'dropped 1 of 3' in r.stdout,
              'exclusion drops exactly the matching file')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print('ALL PASS' if not fail else f'{fail} FAILURE(S)')
    return 1 if fail else 0


if __name__ == '__main__':
    sys.exit(main())
