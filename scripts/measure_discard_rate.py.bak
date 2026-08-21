"""
Week 2 diagnostic — how many labelled pixels does SegEarth-OV3 throw into "background"?

Runs SegEarth-OV3 over a LoveDA split and accumulates:
  * a 7x7 confusion matrix (true class x predicted class)
  * per-class counts of pixels that carry a real GT label but are predicted background
  * per-image discard rates

Outputs to  outputs/week2/  :
  confusion_matrix.npy / .csv     raw counts
  discard_per_class.csv           the headline table
  per_image_discard.csv           distribution across images
  discard_summary.md              paste-ready summary
  confusion_matrix.png            figure
  discard_by_class.png            figure

MUST be run from inside the SegEarth-OV-3 clone (it imports their custom classes):

    conda activate segov3
    cd ~/SegEarth-OV-3
    python ~/"final year pro/Final_year_project"/scripts/measure_discard_rate.py --limit 20
    python ~/"final year pro/Final_year_project"/scripts/measure_discard_rate.py

Label convention (LoveDA):
    GT files hold 0..7 where 0 = no-data (ignored), 1 = background, 2..7 = real classes.
    mmseg is configured with reduce_zero_label=True, so predictions come back 0-indexed.
    We add 1 to predictions to put both on the same 1..7 scale. Getting this wrong
    silently invalidates every number, so --limit runs print an mIoU you can check
    against the known baseline of 47.38.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# --- register the authors' custom segmentor/dataset with mmseg's registry ---
sys.path.insert(0, os.getcwd())
try:
    import segearthov3_segmentor  # noqa: F401
    import custom_datasets        # noqa: F401
    import custom_transforms      # noqa: F401
except ImportError as e:
    sys.exit(f"ERROR: run this from inside the SegEarth-OV-3 clone.\n  {e}")

from mmseg.apis import init_model, inference_model  # noqa: E402

CLASSES = ['background', 'building', 'road', 'water',
           'barren', 'forest', 'agricultural']
N = len(CLASSES)
BACKGROUND = 1  # 1-indexed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/cfg_loveda.py')
    ap.add_argument('--img-dir', default='data/LoveDA/img_dir/val')
    ap.add_argument('--ann-dir', default='data/LoveDA/ann_dir/val')
    ap.add_argument('--out', default=os.path.expanduser('~/outputs/week2'))
    ap.add_argument('--limit', type=int, default=0, help='0 = all images')
    ap.add_argument('--tau', type=float, default=None,
                    help='override prob_thd, e.g. 0.3 or 0.1 for the sweep')
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    names = sorted(p.stem for p in Path(args.img_dir).glob('*.png'))
    if args.limit:
        names = names[:args.limit]
    print(f'{len(names)} images | config {args.config} | out {out}')

    cfg_opts = {}
    if args.tau is not None:
        cfg_opts['model.prob_thd'] = args.tau
        print(f'  overriding prob_thd -> {args.tau}')

    model = init_model(args.config, device='cuda')
    if args.tau is not None:
        model.prob_thd = args.tau   # segmentor reads this attribute at inference

    conf = np.zeros((N, N), dtype=np.int64)      # conf[true, pred], 0-indexed
    per_image = []

    for i, name in enumerate(names, 1):
        img_p = f'{args.img_dir}/{name}.png'
        gt = np.array(Image.open(f'{args.ann_dir}/{name}.png'))
        pred = inference_model(model, img_p).pred_sem_seg.data.cpu().numpy().squeeze()
        pred = pred.astype(np.int64) + 1          # 0-indexed -> 1..7

        valid = gt > 0                            # drop no-data
        g, p = gt[valid], pred[valid]
        np.add.at(conf, (g - 1, p - 1), 1)

        real = g > BACKGROUND                     # carries a real land-cover label
        discarded = real & (p == BACKGROUND)
        per_image.append({
            'image': name,
            'valid_px': int(valid.sum()),
            'real_px': int(real.sum()),
            'discarded_px': int(discarded.sum()),
            'discard_pct_of_real': round(100 * discarded.sum() / max(real.sum(), 1), 2),
        })
        if i % 100 == 0 or i == len(names):
            print(f'  {i}/{len(names)}')

    # ---- derived numbers -------------------------------------------------
    total_valid = conf.sum()
    total_real = conf[1:, :].sum()
    total_discarded = conf[1:, BACKGROUND - 1].sum()

    rows = []
    for c in range(1, N):                          # skip background as a true class
        tot = conf[c, :].sum()
        lost = conf[c, BACKGROUND - 1]
        rows.append({
            'class': CLASSES[c],
            'gt_pixels': int(tot),
            'lost_to_background': int(lost),
            'pct_lost': round(100 * lost / max(tot, 1), 2),
        })
    rows.sort(key=lambda r: -r['pct_lost'])

    # mIoU from the confusion matrix — sanity check against the known 47.38
    inter = np.diag(conf).astype(float)
    union = conf.sum(1) + conf.sum(0) - np.diag(conf)
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    miou = float(np.nanmean(iou) * 100)

    # ---- write ------------------------------------------------------------
    np.save(out / 'confusion_matrix.npy', conf)
    with open(out / 'confusion_matrix.csv', 'w') as f:
        f.write('true\\pred,' + ','.join(CLASSES) + '\n')
        for c in range(N):
            f.write(CLASSES[c] + ',' + ','.join(map(str, conf[c])) + '\n')
    with open(out / 'discard_per_class.csv', 'w') as f:
        f.write('class,gt_pixels,lost_to_background,pct_lost\n')
        for r in rows:
            f.write(f"{r['class']},{r['gt_pixels']},{r['lost_to_background']},{r['pct_lost']}\n")
    with open(out / 'per_image_discard.csv', 'w') as f:
        f.write('image,valid_px,real_px,discarded_px,discard_pct_of_real\n')
        for r in per_image:
            f.write(f"{r['image']},{r['valid_px']},{r['real_px']},"
                    f"{r['discarded_px']},{r['discard_pct_of_real']}\n")

    pcts = np.array([r['discard_pct_of_real'] for r in per_image])
    tau_str = args.tau if args.tau is not None else 'config default (0.5)'
    summary = [
        '# Week 2 — Discard-Rate Diagnostic\n',
        f'- Images: **{len(names)}**  |  τ: **{tau_str}**',
        f'- mIoU recomputed from confusion matrix: **{miou:.2f}** '
        f'(baseline reference: 47.38 — if these disagree, the label alignment is wrong)\n',
        '## Headline\n',
        f'- Labelled (non-no-data) pixels: **{total_valid:,}**',
        f'- Pixels with a real class (excl. background): **{total_real:,}** '
        f'({100*total_real/max(total_valid,1):.1f}%)',
        f'- **Of those, discarded to background: {total_discarded:,} '
        f'({100*total_discarded/max(total_real,1):.2f}%)**\n',
        f'- Per-image discard rate: mean **{pcts.mean():.2f}%**, '
        f'median {np.median(pcts):.2f}%, max {pcts.max():.2f}%\n',
        '## Loss by class\n',
        '| Class | GT pixels | Lost to background | % lost |',
        '|---|---|---|---|',
    ]
    for r in rows:
        summary.append(f"| {r['class']} | {r['gt_pixels']:,} | "
                       f"{r['lost_to_background']:,} | **{r['pct_lost']}%** |")
    summary += [
        '\n## Read this\n',
        '- **> 15% of real-class pixels lost** → premise confirmed, proceed with the '
        'co-occurrence prior.',
        '- **5–15%** → real but modest; the gain ceiling is limited, say so explicitly.',
        '- **< 5%** → premise weak. Pivot to the medium-resolution domain gap '
        '(GID: SegEarth-OV3 42.2 vs SegEarth-OV 46.3).\n',
        'Compare against the τ-sweep before concluding: if τ=0.1 recovers these pixels '
        'without hurting precision, the trivial fix suffices and the method needs a '
        'sharper justification.\n',
    ]
    (out / 'discard_summary.md').write_text('\n'.join(summary))

    # ---- figures ----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        cm = conf / np.maximum(conf.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(N)); ax.set_xticklabels(CLASSES, rotation=45, ha='right')
        ax.set_yticks(range(N)); ax.set_yticklabels(CLASSES)
        ax.set_xlabel('predicted'); ax.set_ylabel('true')
        ax.set_title('Row-normalised confusion matrix (LoveDA val)')
        for a in range(N):
            for b in range(N):
                if cm[a, b] > 0.01:
                    ax.text(b, a, f'{cm[a,b]*100:.0f}', ha='center', va='center',
                            fontsize=8, color='white' if cm[a, b] > 0.5 else 'black')
        fig.colorbar(im); plt.tight_layout()
        plt.savefig(out / 'confusion_matrix.png', dpi=140); plt.close()

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh([r['class'] for r in rows][::-1],
                [r['pct_lost'] for r in rows][::-1], color='#c0392b')
        ax.set_xlabel('% of class pixels discarded to background')
        ax.set_title(f'What SegEarth-OV3 throws away (τ={tau_str})')
        plt.tight_layout(); plt.savefig(out / 'discard_by_class.png', dpi=140); plt.close()
    except Exception as e:
        print(f'(figures skipped: {e})')

    print('\n' + '\n'.join(summary[:14]))
    print(f'\nWritten to {out}')


if __name__ == '__main__':
    main()
