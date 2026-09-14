"""--group-re: are folds really group-disjoint, and is the flag a no-op when absent?

UAVid val is 7 flight sequences of 10 consecutive frames. A frame-level 5-fold puts
near-duplicate frames of the same flight in both calibration and evaluation, so the
fit can memorise the scene it is scored on and the gain is inflated. This checks the
grouping arithmetic that prevents it, and -- equally important -- that omitting the
flag leaves every recorded LoveDA/Potsdam/OEM number untouched.
"""
import re
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent / 'tau_cv.py'


def assign(stems, folds, pattern, seed=0):
    """The grouping path from tau_cv.py, lifted."""
    rng = np.random.default_rng(seed)
    pat = re.compile(pattern)
    keys = []
    for st in stems:
        m = pat.search(st)
        if m is None:
            raise ValueError(st)
        keys.append(m.group(1) if m.groups() else m.group(0))
    uniq = sorted(set(keys))
    pos = {g: i for i, g in enumerate(uniq)}
    gid = np.array([pos[k] for k in keys])
    gperm = rng.permutation(gid.max() + 1)
    gof = np.empty(gid.max() + 1, dtype=int)
    for k, part in enumerate(np.array_split(gperm, folds)):
        gof[part] = k
    owner = gof[gid]
    return gid, [np.where(owner == k)[0] for k in range(folds)], uniq


def main():
    fail = 0

    def check(ok, msg):
        nonlocal fail
        fail += not ok
        print(f'{"ok  " if ok else "FAIL"}  {msg}')

    # --- the no-op guarantee: absent flag must not touch the old code path ---
    src = SRC.read_text()
    check("ap.add_argument('--group-re', default=None," in src,
          '--group-re defaults to None')
    check('    if gid is None:\n        order = rng.permutation(n)\n'
          '        folds = np.array_split(order, args.folds)' in src,
          'ungrouped path is the original permutation split, verbatim')
    check(src.index('gid = None') < src.index('if args.group_re:'),
          'gid is initialised before the grouping branch')
    # the grouping block must consume no randomness when the flag is absent,
    # or every seeded result in the project would shift
    pre = src[src.index('gid = None'):src.index('# ---------------- k-fold')]
    check('rng.' not in pre, 'grouping block draws no randomness when unused')

    # --- UAVid's real shape: 7 sequences x 10 frames ---
    stems = [f'seq{s}_{f:06d}' for s in range(16, 23) for f in range(0, 1000, 100)]
    check(len(stems) == 70, f'70 synthetic UAVid stems ({len(stems)})')
    gid, folds, uniq = assign(stems, 5, r'^(seq[0-9]+)')
    check(len(uniq) == 7, f'7 groups recovered ({len(uniq)})')
    check(sum(len(f) for f in folds) == 70, 'every tile assigned exactly once')
    check(len({i for f in folds for i in f}) == 70, 'no tile in two folds')
    check(all(len(f) > 0 for f in folds), f'no empty fold {[len(f) for f in folds]}')

    # THE property: a group never straddles a fold
    straddle = 0
    for g in range(gid.max() + 1):
        owners = {k for k, f in enumerate(folds) if np.any(gid[f] == g)}
        straddle += len(owners) != 1
    check(straddle == 0, f'no group straddles a fold ({straddle} straddling)')

    # and the ungrouped split DOES straddle -- otherwise the test proves nothing
    rng = np.random.default_rng(0)
    ufolds = np.array_split(rng.permutation(70), 5)
    ustraddle = sum(len({k for k, f in enumerate(ufolds) if np.any(gid[f] == g)}) != 1
                    for g in range(7))
    check(ustraddle == 7, f'frame-level split straddles all 7 groups ({ustraddle}) '
                          f'-- this is the leak being fixed')

    # --- LoveDA-style stems must be rejected loudly, not silently one-group ---
    try:
        assign(['2522', '2523', '2524'], 2, r'^(seq[0-9]+)')
        check(False, 'a non-matching stem raises')
    except ValueError:
        check(True, 'a non-matching stem raises instead of collapsing to one group')

    # --- fewer groups than folds must be refused in the script ---
    check('is fewer than {args.folds} ' in src or 'fewer than' in src,
          'script refuses folds > groups')

    print()
    print('ALL PASS' if not fail else f'{fail} FAILURE(S)')
    return 1 if fail else 0


if __name__ == '__main__':
    sys.exit(main())
