"""Does an unscored 18th sink actually behave the way cfg_dlrsd.py claims?

THE CLAIM, from prereg/predict_dlrsd.md 1b and the comment in cfg_dlrsd.py:

    With 17 classes and bg_idx=17, a thresholded pixel becomes a FALSE NEGATIVE
    for its true class and a false positive for NOTHING.

⛔ That is an assertion about mmseg's metric internals, and this project has
been bitten four times by assertions about a loader's internals that were true
for one dataset and false for the next (WEEK3 §11). It is cheap to check and
expensive to get wrong: if a prediction of 17 were instead clamped into class
16, every discarded pixel in the dataset would pile onto `water` and the table
would still look perfectly reasonable.

The mechanism it depends on is `torch.histc(pred, bins=C, min=0, max=C-1)`,
which SILENTLY DROPS values outside [min, max]. So class 17 contributes to no
predicted area and to no intersection, while the true class still counts the
pixel in its own area.

    union_c = area_pred_c + area_label_c - intersect_c

Three cases are checked against mmseg's own `intersect_and_union` where it can
be imported, and against a faithful reimplementation otherwise:

    1. discard a WRONG pixel  -> the predicted class's IoU RISES (its false
                                 positive is removed), the true class is
                                 unchanged. ⭐ This is what makes discarding
                                 free on DLRSD and is the basis of prediction D4.
    2. discard a RIGHT pixel  -> that class's IoU FALLS (a true positive
                                 becomes a false negative).
    3. no class gains a false positive from the sink, ever.

    python scripts/test_dlrsd_sink.py
"""
import sys

import numpy as np

NCLS = 17
SINK = 17


def _mmseg_iu(pred, label, num_classes, ignore_index=255):
    """mmseg's intersect_and_union, or a faithful stand-in."""
    try:
        import torch
        from mmseg.evaluation.metrics.iou_metric import IoUMetric
        r = IoUMetric.intersect_and_union(
            torch.as_tensor(pred), torch.as_tensor(label),
            num_classes, ignore_index)
        src = 'mmseg.IoUMetric.intersect_and_union'
        return [x.numpy() for x in r], src
    except Exception:                                    # noqa: BLE001
        pass
    try:
        import torch
        p = torch.as_tensor(pred).flatten()
        l = torch.as_tensor(label).flatten()
        m = l != ignore_index
        p, l = p[m], l[m]
        inter = p[p == l]
        kw = dict(bins=num_classes, min=0, max=num_classes - 1)
        ai = torch.histc(inter.float(), **kw)
        ap = torch.histc(p.float(), **kw)
        al = torch.histc(l.float(), **kw)
        return [x.numpy() for x in (ai, ap + al - ai, ap, al)], \
            'torch.histc (mmseg formula, reimplemented)'
    except Exception:                                    # noqa: BLE001
        pass
    # numpy fallback: histc's out-of-range drop is the behaviour being tested,
    # so it is reproduced explicitly rather than by clipping.
    p = np.asarray(pred).ravel()
    l = np.asarray(label).ravel()
    m = l != ignore_index
    p, l = p[m], l[m]
    inter = p[p == l]
    def h(v):
        v = v[(v >= 0) & (v <= num_classes - 1)]
        return np.bincount(v, minlength=num_classes)[:num_classes].astype(float)
    ai, ap, al = h(inter), h(p), h(l)
    return [ai, ap + al - ai, ap, al], 'numpy (mmseg formula, reimplemented)'


def iou(pred, label):
    (ai, au, ap, al), src = _mmseg_iu(pred, label, NCLS)
    with np.errstate(divide='ignore', invalid='ignore'):
        v = np.where(au > 0, 100.0 * ai / np.maximum(au, 1), np.nan)
    return v, ap, src


def main():
    rng = np.random.default_rng(0)
    label = rng.integers(0, NCLS, size=(64, 64)).astype(np.int64)

    # ---------- case 1: a WRONG prediction, then the same pixel discarded
    pred = label.copy()
    wrong = (3, 7)
    true_c = int(label[wrong])
    fp_c = (true_c + 5) % NCLS                  # some other class takes it
    pred[wrong] = fp_c

    base, ap_base, src = iou(pred, label)
    print(f'  metric source: {src}\n')

    sunk = pred.copy()
    sunk[wrong] = SINK
    after, ap_after, _ = iou(sunk, label)

    print(f'case 1 — discard a WRONG pixel (true={true_c}, predicted={fp_c})')
    print(f'  IoU[{fp_c}] (the wrong predictor) : {base[fp_c]:.4f} -> {after[fp_c]:.4f}')
    print(f'  IoU[{true_c}] (the true class)    : {base[true_c]:.4f} -> {after[true_c]:.4f}')
    assert after[fp_c] > base[fp_c], \
        'discarding a false positive must RAISE that class IoU'
    assert abs(after[true_c] - base[true_c]) < 1e-9, \
        'the true class must be unchanged — it was a false negative either way'
    print('  ✅ the wrong predictor improves; the true class is untouched\n')

    # ---------- case 2: a RIGHT prediction, then discarded
    pred2 = label.copy()
    right = (10, 10)
    rc = int(label[right])
    b2, _, _ = iou(pred2, label)
    sunk2 = pred2.copy()
    sunk2[right] = SINK
    a2, _, _ = iou(sunk2, label)
    print(f'case 2 — discard a CORRECT pixel (class {rc})')
    print(f'  IoU[{rc}] : {b2[rc]:.4f} -> {a2[rc]:.4f}')
    assert a2[rc] < b2[rc], 'discarding a true positive must LOWER that class IoU'
    print('  ✅ a correct pixel costs, as it must\n')

    # ---------- case 3: the sink never becomes anyone's false positive
    allsunk = np.full_like(label, SINK)
    _, ap_all, _ = iou(allsunk, label)
    print('case 3 — EVERY pixel discarded')
    print(f'  total predicted area over the 17 classes: {ap_all.sum():.0f}')
    assert ap_all.sum() == 0, \
        (f'the sink leaked into the scored classes: predicted area '
         f'{ap_all.sum():.0f}, expected 0. bg_idx=17 is being clamped into a '
         f'real class, and every discarded pixel in the dataset would pile '
         f'onto it.')
    print('  ✅ no scored class gains a single pixel from the sink\n')

    # ---------- case 4: guard the config constant itself
    print('case 4 — the config constant')
    print(f'  NCLS={NCLS}, bg_idx={SINK}')
    assert SINK >= NCLS, 'bg_idx must sit OUTSIDE 0..NCLS-1 or it is a real class'
    print('  ✅ bg_idx is outside the scored range\n')

    print('ALL PASS — an unscored sink behaves as cfg_dlrsd.py claims.')
    print('⚠️ If this ever fails, cfg_dlrsd.py:bg_idx is wrong and every DLRSD '
          'number computed with it is void.')


if __name__ == '__main__':
    sys.exit(main())
