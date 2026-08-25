"""
Week 3 — the two validation gates on M_global. Run BEFORE building anything on it.

M_global is a 7x7 table mined from SAM 3's own confident predictions. Two things
can be wrong with it, and both are cheap to test and expensive to discover in
Week 11.

GATE 1 -- CIRCULARITY  (ANALYSIS 3.2)
-------------------------------------
M is estimated from SAM 3's confident output. If SAM 3 is systematically unsure
about a class, M under-represents every relationship that class has -- and we
then use M to label the regions SAM 3 missed. The prior would be blind in
exactly the places we need it to see.

The naive test is "is M_pred close to M_gt?". That is too weak: a uniformly
noisy M is harmless, whereas a small error concentrated on the discarded classes
is fatal. So the sharp test here is the CORRELATION between a class's PMI error
and that class's discard rate. If the classes SAM 3 throws away most are also
the classes M gets most wrong, circularity is real AND targeted -- the worst
case. If the error is flat across classes, M is merely noisy and usable.

GATE 2 -- IS M USEFUL?  (WEEK1_RESULTS 8.1)
-------------------------------------------
A correct M can still be useless. For the prior to fix a confusion i -> j it
must say j is IMPLAUSIBLE in the neighbourhoods where i lives. If M instead says
i and j are strongly compatible, the prior REINFORCES the baseline's dominant
errors rather than correcting them.

The baseline's top confusions are forest->agricultural (23.8M), water->
agricultural (19.3M), agricultural->barren (16.1M). This gate reads off what M
says about exactly those pairs. It is a go/no-go signal on the method's ceiling.

    python scripts/validate_m_global.py \
        --pred ~/outputs/week3/M_global_pred.npz \
        --gt   ~/outputs/week3/M_global_gt.npz \
        --confusion ~/outputs/week2_tau0.5_instrumented/confusion_matrix.npy \
        --md ~/outputs/week3/validation.md
"""
import argparse
from pathlib import Path

import numpy as np

# confusion_matrix.npy is in measure_discard_rate.py's own order; entry i maps
# to LoveDA label i+1, so position -- never name -- is the join key.
CACHE_ORDER = ['background', 'building', 'road', 'water',
               'barren', 'forest', 'agricultural']


def rank(x):
    """Ranks with ties averaged. Avoids a scipy dependency in the pinned env."""
    x = np.asarray(x, float)
    order = np.argsort(x)
    r = np.empty(len(x), float)
    r[order] = np.arange(len(x), dtype=float)
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


def kl(p, q, eps=1e-12):
    """KL(p || q) in bits over the flattened off-diagonal distribution."""
    p = np.clip(p, eps, None); p = p / p.sum()
    q = np.clip(q, eps, None); q = q / q.sum()
    return float(np.sum(p * np.log2(p / q)))


def joint(counts, valid):
    M = counts[np.ix_(valid, valid)].astype(float)
    np.fill_diagonal(M, 0.0)
    return M / max(M.sum(), 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True, help='M_global built from predictions')
    ap.add_argument('--gt', required=True, help='M_global built from ground truth')
    ap.add_argument('--confusion', default=None, help='confusion_matrix.npy (7x7)')
    ap.add_argument('--md', default=None)
    args = ap.parse_args()

    P = np.load(Path(args.pred).expanduser(), allow_pickle=True)
    G = np.load(Path(args.gt).expanduser(), allow_pickle=True)

    vp, vg = list(P['valid']), list(G['valid'])
    valid = [c for c in vg if c in vp]
    if len(valid) != len(vg) or len(valid) != len(vp):
        print(f'note: intersecting class sets — pred {vp}, gt {vg} -> {valid}')
    names = [str(x) for x in G['class_names']]
    names = [names[vg.index(c)] for c in valid]
    n = len(valid)
    off = ~np.eye(n, dtype=bool)

    # Recompute PMI on the intersected class set so both sides are directly
    # comparable even if a class is missing from one source.
    def pmi_sub(Z):
        idx = [list(Z['valid']).index(c) for c in valid]
        return np.asarray(Z['pmi_bnd'])[np.ix_(idx, idx)]

    pmi_p, pmi_g = pmi_sub(P), pmi_sub(G)
    Jp, Jg = joint(P['counts'], valid), joint(G['counts'], valid)

    diff = pmi_p - pmi_g
    md = ['# Week 3 — M_global validation\n',
          f'- classes compared: {", ".join(names)}',
          f"- pred source: τ={float(P['tau']):.2f}, α={float(P['alpha'])}",
          f"- tiles: pred {len(P['names'])}, gt {len(G['names'])}\n",
          '## Gate 1 — circularity (ANALYSIS §3.2)\n',
          '| statistic | value |', '|---|---|',
          f'| KL(pred ‖ gt) over the joint boundary distribution | **{kl(Jp, Jg):.4f} bits** |',
          f'| KL(gt ‖ pred) | {kl(Jg, Jp):.4f} bits |',
          f'| mean \\|ΔPMI_bnd\\| off-diagonal | **{np.abs(diff[off]).mean():.3f} bits** |',
          f'| max \\|ΔPMI_bnd\\| | {np.abs(diff[off]).max():.3f} bits |',
          f'| pairs flipping sign | **{int(((pmi_p[off] > 0) != (pmi_g[off] > 0)).sum())} / {int(off.sum())}** |',
          f'| Spearman(PMI_pred, PMI_gt) over pairs | **{spearman(pmi_p[off], pmi_g[off]):+.3f}** |',
          '\nSpearman is the one to read: it asks whether the mined matrix ranks '
          'class pairs the same way ground truth does. Rank agreement is what the '
          'scoring function actually consumes; absolute bits are not.\n',
          '### Per-class error vs how much SAM 3 discards that class\n',
          '| class | mean \\|ΔPMI\\| in its row | discard rate | boundary share pred / gt |',
          '|---|---|---|---|']

    rowerr = np.array([np.abs(diff[k, off[k]]).mean() for k in range(n)])
    bs_p = Jp.sum(1); bs_g = Jg.sum(1)

    discard = np.full(n, np.nan)
    C = None
    if args.confusion:
        C = np.load(Path(args.confusion).expanduser()).astype(float)
        for k, c in enumerate(valid):
            i = c - 1                               # LoveDA label -> cache index
            tot = C[i].sum()
            if tot > 0 and CACHE_ORDER[i] != 'background':
                discard[k] = C[i, 0] / tot          # column 0 = background

    for k in np.argsort(-rowerr):
        d = '—' if not np.isfinite(discard[k]) else f'{100 * discard[k]:.1f}%'
        md.append(f'| {names[k]} | {rowerr[k]:.3f} | {d} | '
                  f'{100 * bs_p[k]:.1f}% / {100 * bs_g[k]:.1f}% |')

    # A targeted-bias test is only meaningful once M is accurate AT ALL. Reporting
    # "error is not concentrated on the discarded classes" for a matrix that ranks
    # pairs no better than chance is true and useless -- it was the first verdict
    # this script printed on real data, and it was misleading. Overall rank
    # agreement is therefore checked FIRST and short-circuits the gate.
    rho_all = spearman(pmi_p[off], pmi_g[off])
    fin = np.isfinite(discard)
    if rho_all < 0.35:
        md.append(f'\n> ⛔ **GATE 1 FAILED — accuracy, before any question of bias.** '
                  f'Spearman(PMI_pred, PMI_gt) = **{rho_all:+.3f}**: the mined matrix '
                  f'does not rank class pairs the way ground truth does, so the '
                  f'targeted-bias test below is not yet meaningful. Fix fidelity '
                  f'first, then re-run this gate. Read the per-class boundary-share '
                  f'column — a class whose pred share is far from its gt share is not '
                  f'being observed, and the most common cause is that it never clears '
                  f'τ (WEEK1_RESULTS §9.2b for `background`).\n')
        if fin.sum() >= 3:
            md.append(f'_(For the record, Spearman(row error, discard rate) = '
                      f'{spearman(rowerr[fin], discard[fin]):+.3f} — not interpretable '
                      f'until the above is fixed.)_')
    elif fin.sum() >= 3:
        r = spearman(rowerr[fin], discard[fin])
        md.append(f'\n**Spearman(row error, discard rate) = {r:+.3f}** '
                  f'over {int(fin.sum())} real classes.\n')
        if r > 0.6:
            md.append('> ⛔ **Circularity is real and targeted.** M is least accurate '
                      'for exactly the classes SAM 3 discards most — the prior is '
                      'blind where it is needed. Mitigate before Week 4: weight '
                      'contributions by confidence, add a class-frequency '
                      'correction, or mine M at a lower τ than the one used at '
                      'inference. Report the mitigation and this figure.')
        elif r > 0.3:
            md.append('> ⚠️ **Mild targeted bias.** Present, not dominant. Try '
                      'confidence-weighted accumulation and re-measure; if the '
                      'correlation drops, keep the weighting and say why.')
        else:
            md.append('> ✅ **Circularity is not targeted.** Error does not '
                      'concentrate on the discarded classes, so M is noisy rather '
                      'than biased against the classes that need it. State this '
                      'with the number — §3.2 raised the risk, and this retires it.')

    md += ['\n## Gate 2 — does M predict the baseline\'s confusions? (§8.1)\n']
    if C is None:
        md.append('_skipped — pass `--confusion` to run this gate._')
    else:
        pairs = []
        for i in range(len(CACHE_ORDER)):
            for j in range(len(CACHE_ORDER)):
                if i == j or CACHE_ORDER[i] == 'background' or CACHE_ORDER[j] == 'background':
                    continue
                if (i + 1) in valid and (j + 1) in valid and C[i, j] > 0:
                    pairs.append((C[i, j], valid.index(i + 1), valid.index(j + 1)))
        pairs.sort(reverse=True)
        top = pairs[:8]

        md += ['For the prior to *fix* a confusion, it must call that pair '
               'implausible. A **positive** `PMI_bnd` on a top confusion means M '
               'would **reinforce** the baseline\'s error.\n',
               '| rank | true → predicted | confusion px | `PMI_bnd` | M would… |',
               '|---|---|---|---|---|']
        bad = 0
        for r_, (cnt, i, j) in enumerate(top, 1):
            v = pmi_p[i, j]
            if v > 0.15:
                verdict, flag = 'reinforce it ⛔', 1
            elif v < -0.15:
                verdict, flag = 'suppress it ✅', 0
            else:
                verdict, flag = 'say nothing (≈chance)', 0
            bad += flag
            md.append(f'| {r_} | {names[i]} → {names[j]} | {int(cnt):,} | '
                      f'**{v:+.2f}** | {verdict} |')

        cnts = np.array([p[0] for p in pairs])
        pmis = np.array([pmi_p[p[1], p[2]] for p in pairs])
        rc = spearman(cnts, pmis)
        md.append(f'\n**Spearman(confusion count, PMI_bnd) = {rc:+.3f}** over '
                  f'{len(pairs)} ordered real-class pairs.\n')
        if rc > 0.3:
            md.append('> ⛔ **The pairs the baseline confuses are the pairs M says '
                      'belong together.** That is expected — things that touch get '
                      'confused — but it means a co-occurrence prior alone cannot '
                      'arbitrate them, because adjacency and confusability point '
                      'the same way. The method needs the *exclusion* half of the '
                      'signal (negative PMI) and the appearance term to do the '
                      'discriminating. Do not claim the prior fixes §8.1\'s top '
                      'confusions without showing it.')
        elif rc < -0.3:
            md.append('> ✅ **M is anti-correlated with the confusions** — it calls '
                      'the baseline\'s worst pairs implausible. That is the '
                      'strongest possible result for this gate and belongs in the '
                      'method section.')
        else:
            md.append('> ➖ **No systematic relationship.** M neither reinforces nor '
                      'suppresses the confusions globally, so any gain will come '
                      'from per-neighbourhood evidence rather than the global '
                      'matrix. Fine, but it means `M_image` and the λ blend carry '
                      'more weight than `M_global` — check that in the λ sweep.')
        md.append(f'\n{bad} of the top {len(top)} confusions would be **reinforced** by M.')

    text = '\n'.join(md)
    print(text)
    if args.md:
        p = Path(args.md).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f'\nwritten: {p}')


if __name__ == '__main__':
    main()
