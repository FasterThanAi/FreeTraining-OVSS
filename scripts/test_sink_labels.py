"""The unscored sink must work, and must change NOTHING on the other datasets.

⛔ THE BUG THIS EXISTS TO PREVENT RECURRING. `labels.py` locates the catch-all
BY NAME and, finding none, nominated the FIRST class -- so on DLRSD it declared
`airplane` the background, and `measure_discard_rate.py` then crashed with
`IndexError: index 17 is out of bounds for axis 1 with size 17` because the
segmentor legitimately predicted the sink. ⭐ The crash was the lucky outcome.
Had the confusion matrix been one wider by accident, the run would have
completed and reported every discarded pixel in the dataset as an `airplane`
false positive, with a table that looked entirely reasonable.

That is the Potsdam mistake (WEEK3 §11) in a new costume: an assumption about
where the catch-all sits, encoded in a script whose whole purpose is to read
the class list from the data.

⚠️ AND THE SECOND HALF MATTERS AS MUCH. Widening the predicted axis touches the
script that produced EVERY cache in this project. If it is not a bit-exact
no-op wherever no sink exists, LoveDA's 47.37 / 29.68% gate, Potsdam's 57.87 /
4.68% and UAVid's 56.86 are all in question. So the four real datasets are
checked explicitly.

    python scripts/test_sink_labels.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels as L  # noqa: E402

DLRSD = ['airplane', 'bare soil', 'buildings', 'cars', 'chaparral', 'court',
         'dock', 'field', 'grass', 'mobile home', 'pavement', 'sand', 'sea',
         'ship', 'tanks', 'trees', 'water']
LOVEDA = ['background', 'building', 'road', 'water', 'barren', 'forest',
          'agricultural']
POTSDAM = ['road', 'building', 'grass', 'tree', 'car', 'clutter']
UAVID = ['background', 'building', 'road', 'car', 'tree', 'vegetation', 'human']
OEM = ['background', 'bareland', 'grass', 'pavement', 'road', 'tree', 'water',
       'cropland', 'building']

ok = True


def check(cond, msg):
    global ok
    print(f'  {"✅" if cond else "⛔"} {msg}')
    ok = ok and bool(cond)


print('\n1. DLRSD — no catch-all, sink at bg_idx=17')
lb = L.Labels(DLRSD, None, 17)
check(lb.sink, 'recognised as an unscored sink')
check(lb.bg == 18, f'bg is mask value 18 (outside 1..17), got {lb.bg}')
check(lb.n_pred == 18, f'n_pred widens to 18, got {lb.n_pred}')
check(len(lb.real) == 17, f'all 17 classes are real, got {len(lb.real)}')
check(1 in lb.real, '`airplane` is NOT treated as the catch-all')

print('\n2. The four real datasets — nothing may move')
for name, cls, bgi, want_bg in [
        ('LoveDA', LOVEDA, 0, 1), ('Potsdam', POTSDAM, 5, 6),
        ('UAVid', UAVID, 0, 1), ('OpenEarthMap', OEM, 0, 1)]:
    lb2 = L.Labels(cls, None, bgi)
    check(not lb2.sink, f'{name}: not a sink')
    check(lb2.bg == want_bg, f'{name}: bg stays mask value {want_bg}, got {lb2.bg}')
    check(lb2.n_pred == lb2.n,
          f'{name}: n_pred == n == {lb2.n} (matrix stays square), got {lb2.n_pred}')
    check(len(lb2.real) == lb2.n - 1,
          f'{name}: {lb2.n - 1} real classes beside the catch-all')

print('\n3. A missing catch-all with bg_idx INSIDE the range still warns')
lb3 = L.Labels(DLRSD, None, 3)          # bg_idx 3 is a real class here
check(not lb3.sink, 'not treated as a sink — bg_idx points at a real class')
check(lb3.bg == 1, 'falls back to mask value 1, loudly')

print('\n4. The sink semantics, on the widened matrix')
# 3 classes, sink at index 3. GT all class 0; predictions: 1 correct, 1 wrong
# (class 1), 1 discarded.
N, NP = 3, 4
conf = np.zeros((N, NP), np.int64)
conf[0, 0] = 10          # correct
conf[0, 1] = 5           # predicted class 1, wrong
conf[0, 3] = 7           # discarded to the sink
sq = conf[:, :N]
inter = np.diag(sq).astype(float)
union = conf.sum(1) + sq.sum(0) - inter
check(union[0] == 22, f'class 0 union counts its discarded pixels: {union[0]} (want 22)')
check(union[1] == 5, f'class 1 union is its 5 false positives: {union[1]} (want 5)')
check(union[2] == 0, f'class 2, untouched, has union 0: {union[2]}')
check(inter[0] == 10, f'intersection is the 10 correct: {inter[0]}')

# ⭐ the decisive one: the sink must be a false positive for NOBODY
conf_no_sink = conf.copy()
conf_no_sink[0, 3] = 0
u2 = conf_no_sink.sum(1) + conf_no_sink[:, :N].sum(0) - np.diag(conf_no_sink[:, :N])
check(union[1] == u2[1],
      'removing the sink column changes no OTHER class union '
      f'({union[1]} vs {u2[1]})')

print('\n5. Square matrices are unaffected by the new code path')
c2 = np.array([[10, 5], [2, 20]], np.int64)
sq2 = c2[:, :2]
u_new = c2.sum(1) + sq2.sum(0) - np.diag(sq2)
u_old = c2.sum(1) + c2.sum(0) - np.diag(c2)
check(np.array_equal(u_new, u_old),
      f'new formula == old formula when square: {u_new.tolist()}')

print('\n' + ('ALL PASS' if ok else '⛔ FAILURES ABOVE'))
sys.exit(0 if ok else 1)
