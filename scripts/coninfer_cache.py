"""
Run ConInfer and cache (gt, pred, conf) in THIS project's .npz format.

WHY. §7.1a: ConInfer thresholds per-class scores with a single global `prob_thd`
(0.8 on LoveDA). That is exactly the design our per-class calibration replaces, and
our fit needs nothing but per-pixel `(gt, pred, conf)`. If it works on their CLIP
backbone as well as on SAM 3, the claim stops being "worth +1.18 on SAM 3" and
becomes "a property of any pipeline that thresholds per-class scores" -- which
answers "is this a SAM 3 quirk?", a question the paper currently cannot.

⭐ WHY THE REPRODUCTION GAP DOES NOT MATTER HERE. Our ConInfer run gives 36.99
against their published 39.33. This measures a DELTA on our own run, and a delta is
robust to a constant offset: if per-class τ adds +X to their scores as we measured
them, that is a valid statement about the method transferring to a different
backbone regardless of the absolute.

⭐ ZERO EDITS TO THEIR SOURCE. mmseg's `postprocess_result` leaves `seg_logits`,
`pred_sem_seg` and `gt_sem_seg` on each `SegDataSample` at ORIGINAL resolution, so
everything we need is available AFTER their `predict()` returns. We wrap it,
call it unmodified, and read the result. Their file is never touched, which means
the observation-only property is structural rather than something to verify by
diffing.

⚠️ VALIDATION GATE. This must print the SAME mIoU as the un-instrumented run
(LoveDA: 36.99). A wrapper that only reads cannot change it -- if the number moves,
something is wrong and the cache must not be used.

    cd ~/ConInfer && python ~/FreeTraining-OVSS/scripts/coninfer_cache.py \
        --config configs_ConInfer/cfg_loveda_1gpu.py \
        --out ~/outputs/coninfer_loveda/cache
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import Runner

# Their modules. This script must be run with cwd=~/ConInfer, both because their
# configs use paths relative to it and because the repo VENDORS open_clip and
# dinov3 -- the local copies must win over anything pip installed.
sys.path.insert(0, str(Path.cwd()))
import segearth_segmentor          # noqa: F401,E402  (registers SegEarthSegmentation)
import ConInfer_segmentor          # noqa: E402
import custom_datasets             # noqa: F401,E402

STATE = dict(out=None, classes=None, n=0, lo=float('inf'), hi=float('-inf'),
             ignore=0, seen=set())


def dump(ds):
    st = STATE
    lg = ds.seg_logits.data.float()                     # (C, H, W), original size
    if lg.ndim != 3:
        raise RuntimeError(f'expected (C,H,W) seg_logits, got {tuple(lg.shape)}')
    k = min(2, lg.shape[0])
    vals, idxs = torch.topk(lg, k=k, dim=0)

    gt = ds.gt_sem_seg.data
    gt = gt[0] if gt.ndim == 3 else gt

    # ---- convert to THIS project's convention -----------------------------
    # labels.py: mask value 0 = no-data/ignore, i+1 = classes[i].
    # mmseg with reduce_zero_label=True has already shifted (raw 0 -> 255 ignore,
    # raw c -> c-1); with reduce_zero_label=False the values are 0..C-1 directly.
    # Adding 1 and mapping 255 -> 0 lands both cases on our convention, which is
    # exactly what measure_discard_rate.py writes.
    ignore = int((gt == 255).sum())
    st['ignore'] += ignore
    gt_ours = torch.where(gt == 255, torch.zeros_like(gt), gt + 1)

    conf = vals[0]
    st['lo'] = min(st['lo'], float(conf.min()))
    st['hi'] = max(st['hi'], float(conf.max()))

    stem = Path(ds.metainfo.get('img_path', f'tile{st["n"]:05d}')).stem
    if stem in st['seen']:
        raise RuntimeError(f'duplicate tile stem {stem!r} -- would overwrite a cache entry')
    st['seen'].add(stem)

    np.savez_compressed(
        st['out'] / f'{stem}.npz',
        conf=conf.cpu().numpy().astype(np.float16),
        pred=idxs[0].cpu().numpy().astype(np.uint8),
        conf2=(vals[1] if k > 1 else vals[0]).cpu().numpy().astype(np.float16),
        pred2=(idxs[1] if k > 1 else idxs[0]).cpu().numpy().astype(np.uint8),
        gt=gt_ours.cpu().numpy().astype(np.uint8),
        classes=np.array(st['classes']),
    )
    st['n'] += 1
    if st['n'] % 250 == 0:
        print(f'  cached {st["n"]}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--work-dir', default='./out_cache')
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    STATE['out'] = out

    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir
    cfg.launcher = 'none'

    runner = Runner.from_cfg(cfg)

    ds = runner.test_dataloader.dataset
    classes = list(getattr(ds, 'METAINFO', {}).get('classes', []) or
                   ds.metainfo.get('classes', []))
    if not classes:
        raise SystemExit('could not read class names from the dataset metainfo')
    STATE['classes'] = classes
    print(f'  classes ({len(classes)}): {", ".join(classes)}')

    # Wrap AFTER the runner is built, so construction is untouched.
    orig = ConInfer_segmentor.ConInferSegmentation.predict

    def wrapped(self, inputs, data_samples):
        outs = orig(self, inputs, data_samples)      # their code, unmodified
        for d in outs:
            dump(d)
        return outs

    ConInfer_segmentor.ConInferSegmentation.predict = wrapped
    print('  predict() wrapped (read-only)\n')

    runner.test()

    st = STATE
    print(f'\n  cached {st["n"]} tiles -> {out}')
    print(f'  conf range observed: [{st["lo"]:.4f}, {st["hi"]:.4f}]')
    if st['lo'] < -1e-6 or st['hi'] > 1.0 + 1e-6:
        print('  ⚠️  conf is OUTSIDE [0,1]. tau_oracle bins thresholds over [0,1],')
        print('      so a raw-logit score would be binned wrongly. Check whether')
        print('      their postprocess applies a sigmoid/softmax before prob_thd.')
    else:
        print('  ✅ conf lies in [0,1], matching the threshold grid our scripts use.')
    print(f'  ignored (255) pixels: {st["ignore"]:,}')
    print('\n  ⚠️ GATE: the mIoU above must match the un-instrumented run exactly')
    print('     (LoveDA: 36.99). A read-only wrapper cannot change it; if it moved,')
    print('     do not use this cache.')


if __name__ == '__main__':
    main()
