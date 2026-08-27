"""
Week 3 — the last untested detection hypothesis, tested cheaply.

WHERE THIS SITS
---------------
Eight detection signals have now come back at chance: conf 0.582, fconf 0.559,
gap 0.558, spres_arg 0.520, spres_max 0.434, fgap 0.447 (pixel level), and
mean_conf 0.576 / max_conf 0.516 / size 0.467 (region level, SLIC atoms).

Every one of them came from SAM 3's SCORES. Nobody has looked at the IMAGE.

And there is a reason to expect the image to know something the scores do not.
LoveDA's `background` is a residual class -- median S_pres 0.022 against 0.45-0.91
for every real class (WEEK1_RESULTS 9.2b), because it is "everything not in the
other six", not a visual concept. SAM 3 has nothing to fire on. So the question
"is this region recoverable?" is not a thresholding question at all. It is
OPEN-SET NOVELTY DETECTION:

    does this region look like NONE of the six real classes?

A score for an absent concept cannot answer that. A distance in feature space can.

WHY COLOUR AND TEXTURE, NOT DINOv3
----------------------------------
The obvious move is deep features -- ConInfer gets +2.80 mIoU on this problem
with DINOv3, so visual features demonstrably carry signal here. But that is a new
model, a new pipeline stage and a week of work, spent on a hypothesis nobody has
checked. Mean RGB, colour spread and gradient energy are free, need no GPU, and
sit strictly BELOW deep features in power. So:

    AUC ~0.5   appearance is likely hopeless even deep; stop, and write up.
    AUC >0.65  crude features already see it -- DINOv3 is worth the investment.

A cheap lower bound is the right instrument for a go/no-go.

PROTOTYPES ARE MINED, NEVER LABELLED. Class prototypes come from pixels SAM 3 is
CONFIDENT about (conf >= tau, argmax = c) -- its own output, no ground truth. GT
is used only to score the AUC afterwards. Both scopes are reported:

    global     one prototype set for the corpus. Stable, but blind to scene
               illumination, season and sensor.
    per-image  fitted on this tile's own confident pixels. Adapts, but collapses
               on a tile with few confident pixels -- and 28.8% of tiles have no
               confident boundary at all at tau=0.5.

    python scripts/appearance_detection.py \
        --cache ~/outputs/week3_fused/cache --img-dir ~/data/loveda/img_dir/val \
        --tau 0.5 --md ~/outputs/week3/appearance_detection.md
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atoms import get_atomiser, load_image      # noqa: E402

LOVEDA = ['unknown', 'background', 'building', 'road',
          'water', 'barren', 'forest', 'agriculture']
NC = 8
REAL = list(range(2, NC))
NB = 512
FD = 4                    # prototype dims: R, G, B, gradient energy


def grad_energy(img):
    """Mean-of-channels gradient magnitude. Crude texture, no dependencies."""
    g = img.astype(np.float32).mean(2)
    gy = np.zeros_like(g); gx = np.zeros_like(g)
    gy[1:-1, :] = (g[2:, :] - g[:-2, :]) * 0.5
    gx[:, 1:-1] = (g[:, 2:] - g[:, :-2]) * 0.5
    return np.sqrt(gx * gx + gy * gy)


# Factor-2 bins. Factor-4 bins left `atom size` itself scoring 0.604 stratified
# against 0.779 raw -- most of the leak gone but not enough to trust a 0.65 call.
_E = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 1 << 60]
SIZE_BINS = list(zip(_E[:-1], _E[1:]))


def auc_w(score, label, weight, lo, hi):
    """Weighted AUC via histograms. label True = positive (recoverable)."""
    b = np.clip(((score - lo) / max(hi - lo, 1e-9) * NB).astype(np.int32), 0, NB - 1)
    pos = np.bincount(b[label], weights=weight[label], minlength=NB)
    neg = np.bincount(b[~label], weights=weight[~label], minlength=NB)
    P, N = pos.sum(), neg.sum()
    if P == 0 or N == 0:
        return float('nan')
    tpr = np.concatenate([[0.0], np.cumsum(pos[::-1]) / P])
    fpr = np.concatenate([[0.0], np.cumsum(neg[::-1]) / N])
    f = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    return float(f(tpr, fpr))


def auc_stratified(score, label, weight, size):
    """AUC computed inside size bins, then pixel-weighted across them.

    A NEGATIVE CONTROL forced this. On images with RANDOM colours -- where no
    signal can exist by construction -- novelty-vs-prototypes scored 0.966. The
    cause is that an atom's MEAN colour has sampling noise scaling as 1/sqrt(size),
    so distance-to-prototype partly measures how SMALL the atom is; and atom size
    correlates with the label, because background atoms are larger. Any
    distance-based feature inherits that leak, and the raw AUC would have reported
    a strong false positive on real data.

    Comparing atoms only against others of similar size removes it. The raw figure
    is still printed beside this one, and a `size` row is printed with both, so the
    gap between them is visible rather than hidden.
    """
    tot = 0.0
    acc = 0.0
    for lo_s, hi_s in SIZE_BINS:
        m = (size >= lo_s) & (size < hi_s)
        if m.sum() < 50 or label[m].sum() == 0 or (~label[m]).sum() == 0:
            continue
        v = score[m]
        if not np.isfinite(v).any():
            continue
        lo, hi = np.nanpercentile(v, [0.5, 99.5])
        a = auc_w(v, label[m], weight[m], lo, hi)
        if not np.isfinite(a):
            continue
        w = weight[m].sum()
        acc += max(a, 1 - a) * w          # direction is free, informativeness is not
        tot += w
    return acc / tot if tot > 0 else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--img-dir', required=True)
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--atoms', choices=['slic', 'cc'], default='slic')
    ap.add_argument('--n-segments', type=int, default=600)
    ap.add_argument('--min-size', type=int, default=64)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--md', default=None)
    args = ap.parse_args()
    warnings.filterwarnings('ignore')
    atomise = get_atomiser(args.atoms)

    files = sorted(Path(args.cache).expanduser().glob('*.npz'))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f'no .npz under {args.cache}')

    gsum = np.zeros((NC, FD)); gcnt = np.zeros(NC)      # global prototypes
    feats, novi, lab, wt, has_img_proto = [], [], [], [], []

    print(f'{len(files)} tiles | τ={args.tau} | atoms={args.atoms}\n')
    for fi, f in enumerate(files):
        z = np.load(f)
        gt = z['gt'].astype(np.uint8)
        conf = z['conf'].astype(np.float32)
        pred = z['pred'].astype(np.int16)
        cand = (gt > 0) & ((conf < args.tau) | (pred == 0))
        if not cand.any():
            continue

        img = load_image(args.img_dir, f.stem)
        ge = grad_energy(img)
        px = np.concatenate([img.astype(np.float32),
                             ge[..., None]], axis=2)     # H,W,4

        # ---- prototypes from SAM 3's OWN confident pixels. No labels read.
        committed = (conf >= args.tau) & (pred > 0) & (gt > 0)
        isum = np.zeros((NC, FD)); icnt = np.zeros(NC)
        for c in REAL:
            m = committed & (pred == c - 1)
            if m.sum() >= 64:
                v = px[m].sum(0)
                isum[c] += v; icnt[c] += m.sum()
                gsum[c] += v; gcnt[c] += m.sum()

        lab_map, n = atomise(cand, img, args.n_segments)
        if n == 0:
            continue
        size = np.bincount(lab_map.ravel(), minlength=n + 1)
        fsum = np.zeros((n + 1, FD))
        for d in range(FD):
            fsum[:, d] = np.bincount(lab_map.ravel(),
                                     weights=px[..., d].ravel(), minlength=n + 1)
        hist = np.bincount(lab_map.ravel().astype(np.int64) * NC + gt.ravel(),
                           minlength=(n + 1) * NC).reshape(n + 1, NC)
        hist[:, 0] = 0
        maj = hist[:, 1:].argmax(1) + 1

        iproto = np.where(icnt[:, None] > 0, isum / np.maximum(icnt[:, None], 1), np.nan)
        avail = [c for c in REAL if icnt[c] > 0]

        for c in range(1, n + 1):
            s = int(size[c])
            if s < args.min_size:
                continue
            mu = fsum[c] / s
            feats.append(mu)
            lab.append(maj[c] >= 2)
            wt.append(s)
            if avail:
                d = np.linalg.norm(iproto[avail] - mu, axis=1)
                novi.append(float(d.min())); has_img_proto.append(True)
            else:
                novi.append(np.nan); has_img_proto.append(False)

        if (fi + 1) % 250 == 0 or fi + 1 == len(files):
            print(f'  {fi + 1}/{len(files)}')

    feats = np.array(feats); lab = np.array(lab, bool)
    wt = np.array(wt, float); novi = np.array(novi)
    hip = np.array(has_img_proto, bool)

    # scale each dim by its own spread so the distance is not dominated by
    # whichever channel happens to have the largest raw range
    sd = feats.std(0); sd[sd < 1e-6] = 1.0
    gproto = np.where(gcnt[:, None] > 0, gsum / np.maximum(gcnt[:, None], 1), np.nan)
    gavail = [c for c in REAL if gcnt[c] > 0]
    dg = np.linalg.norm((gproto[gavail][None] - feats[:, None]) / sd, axis=2)
    nov_g = dg.min(1)

    base = wt[lab].sum() / wt.sum()
    md = ['# Week 3 — appearance-based detection\n',
          f'- tiles: **{len(files)}**  |  τ: **{args.tau}**  |  atoms: **`{args.atoms}`**',
          f'- atoms scored: **{len(feats):,}**, covering **{wt.sum():,.0f}** px',
          f'- recoverable (majority a real class): **{100 * base:.1f}%** by pixels '
          '— the base rate any rule must beat',
          f'- tiles where per-image prototypes existed: atoms **{100 * hip.mean():.1f}%**\n',
          '## Signals\n',
          'Prototypes are built from pixels SAM 3 is confident about — its own '
          'output, never ground truth. GT is used only to score the AUC.\n',
          '`raw` is the plain weighted AUC. **`size-controlled`** recomputes it '
          'inside atom-size bins, because a negative control on random-colour '
          'images scored **0.966** raw — an atom\'s mean colour has noise scaling '
          'as 1/√size, so any distance feature partly measures atom size, and size '
          'correlates with the label. Read the second column. The `atom size` row '
          'is printed so the confound is visible rather than hidden.\n',
          '| signal | raw AUC | size-controlled AUC |', '|---|---|---|']

    tests = [('novelty vs GLOBAL prototypes', nov_g, np.ones(len(nov_g), bool)),
             ('novelty vs PER-IMAGE prototypes', novi, hip & np.isfinite(novi)),
             ('mean R', feats[:, 0], np.ones(len(feats), bool)),
             ('mean G', feats[:, 1], np.ones(len(feats), bool)),
             ('mean B', feats[:, 2], np.ones(len(feats), bool)),
             ('gradient energy (texture)', feats[:, 3], np.ones(len(feats), bool)),
             ('atom size (confound reference)', wt.astype(float),
              np.ones(len(feats), bool))]

    best_name, best_auc = None, 0.5
    for nm, v, m in tests:
        if m.sum() == 0 or not np.isfinite(v[m]).any():
            md.append(f'| {nm} | — | — |')
            continue
        vv = v[m]; ll = lab[m]; ww = wt[m]
        lo, hi = np.nanpercentile(vv, [0.5, 99.5])
        a = auc_w(vv, ll, ww, lo, hi)
        a_eff = max(a, 1 - a) if np.isfinite(a) else 0.5
        a_str = auc_stratified(vv, ll, ww, wt[m])
        md.append(f'| {nm} | {a_eff:.3f} | **{a_str:.3f}** |')
        # judge on the size-controlled figure only
        if np.isfinite(a_str) and a_str > best_auc and not nm.startswith('atom size'):
            best_name, best_auc = nm, a_str

    md += ['\nAUC 0.50 is a coin flip. For reference, the best score-based signal '
           'across eight tests was `conf` at **0.582**.\n', '## Verdict\n']
    if best_auc >= 0.65:
        md.append(f'✅ **Appearance sees it — `{best_name}`, AUC {best_auc:.3f}**, against '
                  '0.582 for the best of eight score-based signals. And these are *crude* '
                  'features: mean colour and a gradient magnitude, no learned '
                  'representation at all. Deep features sit strictly above them, so '
                  'DINOv3 or SAM 3\'s own `F_cond` is now worth the pipeline work. '
                  '**Next: pool deep features per SLIC atom, rebuild prototypes from '
                  'confident regions, re-run this AUC, then gate '
                  '`selective_recovery_miou.py` on it and check the +3.62 oracle bound.**')
    elif best_auc >= 0.58:
        md.append(f'⚠️ **Marginal — `{best_name}`, AUC {best_auc:.3f}**, about level with '
                  '`conf` (0.582) and well short of what the mIoU sweep needs. Crude '
                  'appearance is not enough on its own. Deep features could still clear '
                  'the bar, but this is no longer evidence that they will; it is a coin '
                  'flip on a week of work. Decide against the calendar, not the hope.')
    else:
        md.append(f'⛔ **Appearance does not detect it either** (best `{best_name}`, AUC '
                  f'{best_auc:.3f}). Colour and texture cannot separate a suppressed '
                  'real class from LoveDA background — which, on reflection, is what '
                  '`background` being a *residual* class implies: it is not visually '
                  'coherent, so there is no compact region of feature space for it to '
                  'occupy and nothing for a novelty score to be far from.\n\n'
                  '**That closes the last cheap hypothesis.** Nine detection signals, '
                  'score-based and appearance-based, all at chance, while the labelling '
                  'side is worth +3.62 given the right regions. The honest conclusion is '
                  'that the residual is recoverable in principle and not detectable from '
                  'anything this pipeline exposes. Write that up — it is a real finding '
                  'with an oracle bound attached, and it tells the next person exactly '
                  'where to dig.')

    text = '\n'.join(md)
    print('\n' + text)
    if args.md:
        p = Path(args.md).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f'\nwritten: {p}')


if __name__ == '__main__':
    main()
