"""Figure 11 — one word against the whole method. Three panels, one argument.

    (a) UAVid: a single word is worth +3.53 mIoU, and it is SURGICAL
    (b) the same rule on Potsdam COSTS 2.71 -- and tree's recall does not move
    (c) the word and the multiplier are SUBSTITUTES: two routes, one endpoint

⭐ Panel (b) is the one that turns this from an anecdote into a finding. UAVid and
Potsdam look identical in a results table -- a scale gain carried by `tree` -- and
panel (b) shows they have different causes. On UAVid the rename lifts tree's recall
16.9 points; on Potsdam it moves it by 0.00, so Potsdam's tree deficit is visual
confusion at 5 cm GSD rather than a naming error.

⚠️ Panel (a) deliberately plots ALL seven classes including the four that do not
move. Showing only tree and vegetation would make the effect look broad when its
whole force is that it is narrow.

EVERY NUMBER IS CITED to VOCABULARY_RESULTS.md / UAVID_RESULTS.md and printed on
render, so a figure that has drifted from its source table is visible rather than
silent.

    python scripts/fig_vocab_lever.py --outdir docs
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

C_UP, C_DOWN, C_FLAT, C_WORD, C_LEVER = '#2e8b57', '#c0392b', '#95a5a6', '#c47f17', '#1f6f8b'

# ---- VOCABULARY_RESULTS.md §1 : UAVid, `vegetation` -> `low vegetation` -----
UAV_CLASSES = ['tree', 'low vegetation', 'human', 'car', 'road', 'building', 'background']
UAV_DELTA = [+14.66, +11.14, +0.02, +0.01, -0.02, -0.04, -1.07]
UAV_MIOU = (56.86, 60.39)              # §1  baseline before / after

# ---- VOCABULARY_RESULTS.md §2 : Potsdam, `grass` -> `low vegetation` --------
POTS_MIOU = (57.83, 55.12)             # §2
# tree, the class both datasets' scale gain is carried by
TREE_REC = {'UAVid': (55.90, 72.76),   # §1  recall before / after the rename
            'Potsdam': (38.63, 38.63)} # §2  IDENTICAL to a hundredth
TREE_PREC = {'UAVid': (91.96, 84.65),  # UAVID §18 (after both levers, for context)
             'Potsdam': (93.34, 92.17)}

# ---- UAVID_RESULTS.md §20 : the two routes -------------------------------
ROUTE_A = [('baseline\n(shipped prompt)', 56.86), ('+ per-class\nthreshold', 57.92),
           ('+ per-class\nscale', 63.55)]
ROUTE_B = [('baseline\n(shipped prompt)', 56.86), ('one word', 60.40),
           ('+ per-class\nthreshold', 61.59), ('+ per-class\nscale', 63.62)]

# ---- for the scale bar: what a word is worth against what a method is worth
COMPARE = [('one word,\nUAVid', 3.53, C_WORD),
           ('two words,\nLoveDA `barren`', 4.94, C_WORD),
           ('our method,\nLoveDA', 2.32, C_LEVER),
           ('our method,\nUAVid (corrected)', 3.22, C_LEVER)]


def verify():
    """Print every plotted number beside its source. A drifted figure is visible."""
    print('  VOCABULARY_RESULTS.md §1  UAVid  56.86 -> 60.39  (+3.53)')
    print(f'     plotted: {UAV_MIOU[0]} -> {UAV_MIOU[1]}  '
          f'(+{UAV_MIOU[1]-UAV_MIOU[0]:.2f})')
    s = sum(UAV_DELTA) / 7
    print(f'     per-class deltas sum/7 = {s:+.2f}  (must equal +3.53)')
    assert abs(s - 3.53) < 0.005, 'per-class deltas do not reconstruct the headline'
    print('  VOCABULARY_RESULTS.md §2  Potsdam  57.83 -> 55.12  (-2.71)')
    print(f'     plotted: {POTS_MIOU[0]} -> {POTS_MIOU[1]}  '
          f'({POTS_MIOU[1]-POTS_MIOU[0]:+.2f})')
    assert abs((POTS_MIOU[1] - POTS_MIOU[0]) + 2.71) < 0.005
    print(f'  tree recall  UAVid {TREE_REC["UAVid"][0]} -> {TREE_REC["UAVid"][1]} '
          f'({TREE_REC["UAVid"][1]-TREE_REC["UAVid"][0]:+.2f})')
    print(f'               Potsdam {TREE_REC["Potsdam"][0]} -> '
          f'{TREE_REC["Potsdam"][1]} '
          f'({TREE_REC["Potsdam"][1]-TREE_REC["Potsdam"][0]:+.2f})  <- the finding')
    assert TREE_REC['Potsdam'][0] == TREE_REC['Potsdam'][1], 'Potsdam tree recall moved'
    print(f'  UAVID_RESULTS.md §20  routes end {ROUTE_A[-1][1]} vs {ROUTE_B[-1][1]} '
          f'({abs(ROUTE_B[-1][1]-ROUTE_A[-1][1]):.2f} apart)')
    assert abs(ROUTE_B[-1][1] - ROUTE_A[-1][1]) < 0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='docs')
    args = ap.parse_args()
    verify()

    fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.9))

    # ---- (a) UAVid, surgical -------------------------------------------
    y = np.arange(len(UAV_CLASSES))[::-1]
    cols = [C_UP if d > 0.5 else (C_DOWN if d < -0.5 else C_FLAT) for d in UAV_DELTA]
    ax[0].barh(y, UAV_DELTA, color=cols, height=0.66)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(UAV_CLASSES, fontsize=9)
    ax[0].axvline(0, color='0.3', lw=0.8)
    for yy, d in zip(y, UAV_DELTA):
        ax[0].text(d + (0.45 if d >= 0 else -0.45), yy, f'{d:+.2f}',
                   va='center', ha='left' if d >= 0 else 'right', fontsize=8.5)
    ax[0].set_xlim(-4.0, 18.5)
    ax[0].set_xlabel(r'$\Delta$ IoU from one word', fontsize=9)
    ax[0].set_title(f'(a) UAVid: {UAV_MIOU[0]} $\\rightarrow$ {UAV_MIOU[1]} mIoU\n'
                    '"vegetation" $\\rightarrow$ "low vegetation"',
                    fontsize=9.5)
    ax[0].text(0.97, 0.06, 'four classes move\nby $\\leq$ 0.04', transform=ax[0].transAxes,
               ha='right', va='bottom', fontsize=8, color='0.35', style='italic')

    # ---- (b) the same rule, opposite outcome; and tree's recall ---------
    lbl = ['UAVid', 'Potsdam']
    x = np.arange(2)
    before = [TREE_REC[k][0] for k in lbl]
    after = [TREE_REC[k][1] for k in lbl]
    ax[1].bar(x - 0.18, before, 0.34, label='shipped prompt', color='0.72')
    ax[1].bar(x + 0.18, after, 0.34, label='"low vegetation"', color=C_WORD)
    for i, k in enumerate(lbl):
        d = TREE_REC[k][1] - TREE_REC[k][0]
        ax[1].text(i, max(before[i], after[i]) + 3.2, f'{d:+.2f}',
                   ha='center', fontsize=10,
                   fontweight='bold', color=C_UP if d > 1 else C_DOWN)
        ax[1].text(i, -9.5, f'mIoU {(UAV_MIOU if k == "UAVid" else POTS_MIOU)[1] - (UAV_MIOU if k == "UAVid" else POTS_MIOU)[0]:+.2f}',
                   ha='center', fontsize=9, color=C_UP if k == 'UAVid' else C_DOWN)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(lbl, fontsize=9.5)
    ax[1].set_ylim(0, 92)
    ax[1].set_ylabel('tree recall (%)', fontsize=9)
    ax[1].set_title('(b) the same rule, opposite outcomes\n'
                    'Potsdam tree does not move at all', fontsize=9.5)
    ax[1].legend(fontsize=8, loc='upper right', frameon=False)

    # ---- (c) the two routes, one endpoint -------------------------------
    # ⚠️ The two routes have a DIFFERENT number of steps -- route A never touches
    # the word. Plotting them on a shared "step 1/2/3" axis would imply they do the
    # same things in the same order. They are placed on the stages they actually
    # occupy, so the lines converge on the shared endpoint, which is the point.
    STAGE = ['baseline', 'one word', '+ per-class\nthreshold', '+ per-class\nscale']
    xa, ya = [0, 2, 3], [v for _, v in ROUTE_A]          # no word step
    xb, yb = [0, 1, 2, 3], [v for _, v in ROUTE_B]
    ax[2].plot(xa, ya, marker='o', color=C_LEVER, lw=2, ms=6,
               label='keep the shipped prompt,\nlet the levers repair it')
    ax[2].plot(xb, yb, marker='s', color=C_WORD, lw=2, ms=6,
               label='correct the word first,\nthen the levers')
    # both routes start from the same point; label it once, on route B
    for xx, yy in list(zip(xa, ya))[1:]:
        ax[2].annotate(f'{yy:.2f}', (xx, yy), textcoords='offset points',
                       xytext=(0, -14), ha='center', fontsize=8.5, color=C_LEVER)
    for xx, yy in zip(xb, yb):
        ax[2].annotate(f'{yy:.2f}', (xx, yy), textcoords='offset points',
                       xytext=(0, 8), ha='center', fontsize=8.5, color=C_WORD)
    ax[2].annotate('', xy=(3, ROUTE_A[-1][1]), xytext=(3, ROUTE_B[-1][1]),
                   arrowprops=dict(arrowstyle='<->', color='0.35', lw=1.0))
    ax[2].text(2.88, (ROUTE_A[-1][1] + ROUTE_B[-1][1]) / 2,
               f'{abs(ROUTE_B[-1][1]-ROUTE_A[-1][1]):.2f}', ha='right', va='center',
               fontsize=9, color='0.25', fontweight='bold')
    ax[2].set_xticks(range(4))
    ax[2].set_xticklabels(STAGE, fontsize=8)
    ax[2].set_xlim(-0.3, 3.45)
    ax[2].set_ylim(55.0, 65.8)
    ax[2].set_ylabel('mIoU', fontsize=9)
    ax[2].set_title(f'(c) substitutes: the two routes end '
                    f'{abs(ROUTE_B[-1][1]-ROUTE_A[-1][1]):.2f} mIoU apart',
                    fontsize=9.5)
    ax[2].legend(fontsize=7.5, loc='upper left', frameon=False)

    for a in ax:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.tick_params(labelsize=8.5)

    fig.tight_layout()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        p = out / f'fig11_vocab_lever.{ext}'
        fig.savefig(p, dpi=200, bbox_inches='tight')
        print(f'  wrote {p}')


if __name__ == '__main__':
    main()
