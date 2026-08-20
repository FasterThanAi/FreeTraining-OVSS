"""Four-panel qualitative figure: image | GT | prediction | pixels discarded to background.

Run from inside the SegEarth-OV-3 clone (needs its configs + model on the path):
    python /path/to/scripts/visualize_predictions.py 2522 3144 4276
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mmseg.apis import init_model, inference_model
sys.path.insert(0, os.getcwd())
import segearthov3_segmentor   # noqa: F401  registers SegEarthOV3Segmentation
import custom_datasets         # noqa: F401  registers LoveDADataset variants
CLASSES = ['background', 'building', 'road', 'water',
           'barren', 'forest', 'agricultural']
OUT = os.path.expanduser('~/vis_out')
os.makedirs(OUT, exist_ok=True)

model = init_model('configs/cfg_loveda.py', device='cuda')

for name in sys.argv[1:]:
    img_p = f'data/LoveDA/img_dir/val/{name}.png'
    gt_p  = f'data/LoveDA/ann_dir/val/{name}.png'

    gt = np.array(Image.open(gt_p))                    # 0=no-data, 1..7=classes
    pred = inference_model(model, img_p).pred_sem_seg.data.cpu().numpy().squeeze()
    pred = pred + 1                                    # reduce_zero_label shifts by 1

    valid    = gt > 0                                  # ignore no-data
    real     = gt > 1                                  # a real (non-background) class
    discarded = real & (pred == 1) & valid             # <-- what our method must recover

    pct = 100 * discarded.sum() / max(valid.sum(), 1)

    fig, ax = plt.subplots(1, 4, figsize=(22, 5.5))
    ax[0].imshow(Image.open(img_p));                       ax[0].set_title('image')
    ax[1].imshow(gt,   vmin=0, vmax=7, cmap='tab10');      ax[1].set_title('ground truth')
    ax[2].imshow(pred, vmin=0, vmax=7, cmap='tab10');      ax[2].set_title('SegEarth-OV3 (τ=0.5)')
    ax[3].imshow(discarded, cmap='Reds');
    ax[3].set_title(f'discarded to background: {pct:.1f}%')
    for a in ax: a.axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUT}/{name}.png', dpi=110, bbox_inches='tight')
    plt.close()

    # which classes are being lost?
    lost = {CLASSES[c-1]: int((discarded & (gt == c)).sum())
            for c in range(2, 8)}
    lost = {k: v for k, v in sorted(lost.items(), key=lambda x: -x[1]) if v > 0}
    print(f'{name}: {pct:5.1f}% discarded | {lost}')
