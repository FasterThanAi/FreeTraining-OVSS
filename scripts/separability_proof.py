"""
Is the per-class threshold objective separable? Proved for one objective,
DISPROVED for the other -- on the project's own code, not a re-implementation.

WHY THIS EXISTS. `tau_curves.py` measured max |Δ IoU| of any other real class as
exactly 0.000000 while one threshold sweeps the grid, and observed that
coordinate ascent converges in a single pass under BOTH objectives on LoveDA.
The first is a theorem. The second is a coincidence of that dataset, and stating
it as a property would be wrong.

THE ARGUMENT, for a fixed argmax. A pixel predicted class c either clears
tau_c and stays c, or fails and becomes the catch-all b. It can never become
another real class. So for every real c:

    TP_c = #{gt=c, pred=c, conf >= tau_c}
    FP_c = #{gt!=c, pred=c, conf >= tau_c}
    FN_c = #{gt=c} - TP_c

⭐ All three depend on tau_c ALONE. Hence:

  objective `real`  = mean over real classes
                    = sum of terms each depending on ONE threshold
                    => SEPARABLE. Each tau_c can be maximised independently,
                       one sweep IS the global optimum, and coordinate ascent
                       is EXACT rather than greedy.

  objective `all`   = the above PLUS IoU_b, and b collects every rejected pixel
                     from every class, so IoU_b depends on the WHOLE vector.
                     Maximising over tau_c maximises IoU_c + IoU_b(tau_c; rest),
                     and the second term's shape depends on the rest.
                     => NOT separable in general.

⛔ So the empirical "max |dtau| = 0.0000 under `all` too" on LoveDA does not
generalise. This script finds an explicit counterexample to prove that, so the
paper can state the separability result at exactly the scope it holds.

    python scripts/separability_proof.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tau_oracle import confusion_at                       # noqa: E402
from tau_cv import obj_miou, fit                          # noqa: E402

NB = 20            # coarse grid: the argument is about structure, not resolution


def random_hist(rng, nc, nb, scale=400):
    """A (gt, pred, bin) count table with the shape a real cache produces."""
    H = rng.integers(0, scale, size=(nc, nc, nb)).astype(np.int64)
    for c in range(nc):                       # a diagonal, as any real model has
        H[c, c] += rng.integers(0, 4 * scale, size=nb)
    return H


def best_tau_for(H, c, taus, bg, objective):
    """argmax over tau_c of the objective, with every other threshold held."""
    grid = np.arange(NB + 1) / NB
    best, arg = -1.0, None
    for t in grid:
        tt = taus.copy(); tt[c] = t
        s = obj_miou(confusion_at(H, tt, bg, NB), bg, objective)
        if s > best + 1e-12:
            best, arg = s, t
    return arg


def main():
    rng = np.random.default_rng(0)
    nc, bg = 5, 0
    ok = True

    # ---------------------------------------------------------------- part 1
    print('PART 1 — objective `real`: is one class\'s IoU independent of the others?\n')
    worst = 0.0
    for trial in range(20):
        H = random_hist(rng, nc, NB)
        base = np.full(nc, 0.5)
        for c in range(1, nc):
            ious = []
            for t in (0.0, 0.25, 0.75, 1.0):          # sweep ANOTHER class
                tt = base.copy(); tt[(c % (nc - 1)) + 1] = t
                C = confusion_at(H, tt, bg, NB)
                tp = np.diag(C)[c]
                iou = tp / max(C[c].sum() + C[:, c].sum() - tp, 1)
                ious.append(iou)
            worst = max(worst, max(ious) - min(ious))
    print(f'  max |Δ IoU| of a class while a DIFFERENT class\'s τ sweeps: {worst:.12f}')
    exact = worst == 0.0
    ok &= exact
    print(f'  {"✅ EXACT — separable, as the argument predicts" if exact else "⛔ NOT exact"}\n')

    # ---------------------------------------------------------------- part 2
    print('PART 2 — objective `real`: is coordinate ascent EXACT (one pass = six)?\n')
    same = True
    for trial in range(20):
        H = random_hist(rng, nc, NB)
        t1 = fit(H, bg, NB, rounds=1, objective='real')
        t6 = fit(H, bg, NB, rounds=6, objective='real')
        same &= np.array_equal(t1, t6)
    ok &= same
    print(f'  {"✅ identical on all 20 random tables — one sweep is the optimum" if same else "⛔ differs"}\n')

    # ---------------------------------------------------------------- part 3
    print('PART 3 — objective `all`: separable too, or only on LoveDA?\n')
    found = None
    for trial in range(4000):
        H = random_hist(rng, nc, NB)
        lo = np.full(nc, 0.10); hi = np.full(nc, 0.90)
        lo[bg] = hi[bg] = 0.0
        for c in range(1, nc):
            a = best_tau_for(H, c, lo, bg, 'all')
            b = best_tau_for(H, c, hi, bg, 'all')
            if a is not None and b is not None and abs(a - b) > 1e-9:
                found = (trial, c, a, b); break
        if found:
            break
    if found:
        trial, c, a, b = found
        print(f'  ⛔ COUNTEREXAMPLE at trial {trial}: the best τ for class {c} is '
              f'{a:.2f} when the other thresholds sit at 0.10,')
        print(f'     and {b:.2f} when they sit at 0.90. The optimum for one class '
              f'DEPENDS on the others.')
        print('  → objective `all` is NOT separable. LoveDA\'s 0.0000 is a property '
              'of that dataset, not of the rule.\n')
    else:
        print('  no counterexample found in 4000 tables — surprising; investigate '
              'before claiming non-separability.\n')
    ok &= (found is not None)

    print('=' * 74)
    print('CONCLUSION, at the scope the evidence supports:')
    print('  • objective `real` (what tau_cv and tau_deploy fit): SEPARABLE.')
    print('    One sweep per class is the exact optimum; coordinate ascent is not')
    print('    a heuristic here. Cost N*(bins+1) instead of (bins+1)^N.')
    print('  • objective `all`: NOT separable. The catch-all couples the vector,')
    print('    and a counterexample exists. Any convergence observed under `all`')
    print('    is empirical and dataset-specific.')
    print('=' * 74)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
