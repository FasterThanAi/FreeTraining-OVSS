"""Bar plots of pixel accuracy, from the JSON that pixel_accuracy.py writes.

Two figures, both datasets side by side:
  <out>_overall    aAcc for A / B / C per dataset
  <out>_per_class  per-class accuracy (recall) for A / B / C, one panel per dataset

Needs no cache -- it reads only the committed JSON, so it runs on either machine.
⛔ Refuses a JSON whose eval.py gate failed.

    python scripts/fig_pixel_accuracy.py \\
        --json results/dlrsd/pixel_accuracy.json results/potsdam/pixel_accuracy.json \\
        --out docs/fig_pixel_accuracy
    # baseline vs ours only (both levers together):
    python scripts/fig_pixel_accuracy.py --rungs A,C \\
        --json results/dlrsd/pixel_accuracy.json results/potsdam/pixel_accuracy.json \\
        --out docs/fig_pixel_accuracy_baseline_vs_ours
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

RUNGS = ['A', 'B', 'C']
LABEL = {'A': 'A  baseline', 'B': 'B  + per-class τ', 'C': 'C  + per-class scale'}
COLOR = {'A': '#9e9e9e', 'B': '#6baed6', 'C': '#2171b5'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', nargs='+', required=True)
    ap.add_argument('--out', required=True, help='path prefix, no extension')
    ap.add_argument('--rungs', default='A,B,C',
                    help="which steps to draw. 'A,C' = baseline vs ours (both levers "
                         "together), without the lever-1 step in between.")
    args = ap.parse_args()
    rungs = [r.strip() for r in args.rungs.split(',') if r.strip()]
    if not rungs or any(r not in RUNGS for r in rungs):
        raise SystemExit(f'⛔ --rungs must be drawn from A,B,C, got {args.rungs}')
    label = dict(LABEL)
    if rungs == ['A', 'C']:
        label = {'A': 'baseline (SegEarth-OV3)', 'C': 'ours (both levers)'}

    data = []
    for p in args.json:
        d = json.loads(Path(p).expanduser().read_text())
        if not d.get('gate_passed'):
            raise SystemExit(f'⛔ {p}: the eval.py gate did not pass '
                             f'(worst gap {d.get("gate_worst_diff")}). Not plotting it.')
        data.append(d)
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    # ---- figure 1: overall pixel accuracy
    fig, ax = plt.subplots(figsize=(2.6 + 2.2 * len(data), 4.2))
    x = np.arange(len(data))
    bw = 0.78 / len(rungs)
    for j, r in enumerate(rungs):
        vals = [d['rungs'][r]['aAcc'] for d in data]
        bars = ax.bar(x + (j - (len(rungs) - 1) / 2) * bw, vals, bw, label=label[r],
                      color=COLOR[r])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f'{v:.1f}',
                    ha='center', va='bottom', fontsize=8)
        print(f'  overall {r}: ' + ', '.join(f'{d["dataset"]} {v:.2f}' for d, v in zip(data, vals)))
    lo = min(d['rungs'][r]['aAcc'] for d in data for r in rungs)
    hi = max(d['rungs'][r]['aAcc'] for d in data for r in rungs)
    ax.set_ylim(max(0, lo - 10), min(100, hi + 5))
    ax.set_xticks(x)
    ax.set_xticklabels([f'{d["dataset"]}\n({d["n_tiles"]} held-out tiles)' for d in data])
    ax.set_ylabel('pixel accuracy (% of labelled pixels correct)')
    ax.set_title('Overall pixel accuracy')
    ax.legend(fontsize=8, loc='upper left')      # empty corner: no bar underneath
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}_overall.{ext}', dpi=200)
    plt.close(fig)
    print(f'  wrote {out}_overall.png/.pdf')

    # ---- figure 2: per-class accuracy
    heights = [max(3.0, 0.42 * len(d['classes']) + 1.2) for d in data]
    fig, axes = plt.subplots(1, len(data), figsize=(6.2 * len(data), max(heights)),
                             squeeze=False)
    for ax, d in zip(axes[0], data):
        acc = {r: [np.nan if v is None else v for v in d['rungs'][r]['acc']] for r in RUNGS}
        n = len(d['classes'])
        order = sorted(range(n), key=lambda c: np.nan_to_num(acc['C'][c] - acc['A'][c]))
        y = np.arange(n)
        bh = 0.8 / len(rungs)
        for j, r in enumerate(rungs):
            ax.barh(y + ((len(rungs) - 1) / 2 - j) * bh, [acc[r][c] for c in order], bh,
                    label=label[r], color=COLOR[r])
        ax.set_yticks(y)
        ax.set_yticklabels([f'{d["classes"][c]} ({d["gt_share"][c]:.1f}%)' for c in order],
                           fontsize=8)
        for yy, c in zip(y, order):
            dd = acc['C'][c] - acc['A'][c]
            if not np.isnan(dd):
                ax.text(101, yy, f'{dd:+.1f}', va='center', fontsize=7,
                        color='#1a7f37' if dd >= 0 else '#c62828')
            print(f'  {d["dataset"]} {d["classes"][c]}: '
                  + '  '.join(f'{r} {acc[r][c]:.2f}' for r in rungs))
        ax.set_xlim(0, 110)
        ax.set_xlabel('per-class accuracy (% of the class\'s pixels labelled correctly)')
        ax.set_title(f'{d["dataset"]} — per class  (right: change from baseline to ours; brackets: share of pixels)',
                     fontsize=9)
        ax.grid(axis='x', alpha=0.3)
    # One legend for the whole figure, above the panels -- inside a panel it
    # covered the C - A label of the bottom class.
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=len(rungs), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}_per_class.{ext}', dpi=200)
    plt.close(fig)
    print(f'  wrote {out}_per_class.png/.pdf')


if __name__ == '__main__':
    main()
