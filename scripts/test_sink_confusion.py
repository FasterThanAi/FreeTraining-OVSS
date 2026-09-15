"""`confusion_at` with an unscored sink — and proof it changed nothing else.

`tau_cv.py`, `argmax_reorder.py` and `metric_report.py` all import
`confusion_at` from `tau_oracle.py`, so ONE allocation decides four scripts and
every lever result in the project. It was `np.zeros((nc, nc))`, and where a
dataset has no catch-all the discard target is an unscored sink at index `nc`
itself -- `C[:, bg]` then raises IndexError.

⛔ WHY THE NO-OP HALF MATTERS AS MUCH AS THE FIX. Those four scripts produced
+1.18 on LoveDA, +4.86 on Potsdam, +5.89 on UAVid and +2.51 on ConInfer. If the
widening is not bit-identical wherever `bg < nc`, every one of those is in
question. The fix is `max(nc, bg + 1)` precisely because it cannot fire on a
dataset that has a catch-all.

⭐ Test 4 is a bonus: raising one class's threshold moves that class's column
and the sink, and touches no other column. That is SEPARABILITY_RESULTS.md's
claim -- the real-class objective separates exactly -- falling out of the
matrix construction rather than being measured.

    python scripts/test_sink_confusion.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tau_oracle import confusion_at, per_class_iou, miou  # noqa: E402

NB = 10
ok = True


def chk(c, m):
    global ok
    print(f'  {"✅" if c else "⛔"} {m}')
    ok = ok and bool(c)


rng = np.random.default_rng(0)

print('\n1. No-op guarantee — datasets that HAVE a catch-all')
H = rng.integers(0, 50, size=(7, 7, NB)).astype(np.int64)
C = confusion_at(H, np.full(7, 0.5), 0, NB)                 # LoveDA/UAVid shape
chk(C.shape == (7, 7), f'catch-all first: stays (7,7), got {C.shape}')
chk(C.sum() == H.sum(), f'every pixel accounted: {C.sum()} == {H.sum()}')
C2 = confusion_at(H[:6, :6], np.full(6, 0.5), 5, NB)        # Potsdam: catch-all LAST
chk(C2.shape == (6, 6), f'catch-all last: stays (6,6), got {C2.shape}')
chk(C2.sum() == H[:6, :6].sum(), 'Potsdam-shaped: every pixel accounted')

print('\n2. Sink — 17 classes, bg = 17 (DLRSD)')
H3 = rng.integers(0, 50, size=(17, 17, NB)).astype(np.int64)
C3 = confusion_at(H3, np.full(17, 0.5), 17, NB)
chk(C3.shape == (17, 18), f'widens to (17,18), got {C3.shape}')
chk(C3.sum() == H3.sum(), f'every pixel accounted: {C3.sum()} == {H3.sum()}')
chk(len(per_class_iou(C3)) == 17, 'per_class_iou returns 17 values, not 18')
chk(np.isfinite(miou(C3)), f'miou is finite: {miou(C3):.4f}')

print('\n3. The sink is a false positive for nobody')
C_keep = confusion_at(H3, np.full(17, 0.0), 17, NB)
C_sink = confusion_at(H3, np.full(17, 1.0), 17, NB)
chk(C_sink[:, :17].sum() == 0,
    f'all discarded: the 17 real columns hold {C_sink[:, :17].sum()} (want 0)')
chk(C_sink[:, 17].sum() == H3.sum(), 'all pixels land in the sink column')
chk(np.nansum(per_class_iou(C_sink)) == 0,
    'every class IoU is 0 when everything is discarded')
chk(C_keep[:, 17].sum() == 0, 'tau = 0 discards nothing')

print('\n4. ⭐ Separability — one threshold moves one column, plus the sink')
t = np.full(17, 0.0)
t[3] = 1.0
Cx = confusion_at(H3, t, 17, NB)
moved = [c for c in range(17) if not np.array_equal(Cx[:, c], C_keep[:, c])]
chk(moved == [3], f'only column 3 changed, got {moved}')
chk(Cx[:, 17].sum() == C_keep[:, 3].sum(),
    "exactly class 3's pixels went to the sink, no more and no less")

print('\n5. `real` and `all` objectives coincide with no catch-all (D3)')
v = per_class_iou(C3)
real = np.array([v[c] for c in range(len(v)) if c != 17])
chk(np.array_equal(np.nan_to_num(v), np.nan_to_num(real)),
    'the `real` filter removes nothing, so the two objectives are the same fit')

print('\n' + ('ALL PASS' if ok else '⛔ FAILURES ABOVE'))
sys.exit(0 if ok else 1)
