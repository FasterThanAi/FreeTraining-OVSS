"""
Is the calibration gain governed by how much of the residual a threshold can REACH?

THE OBSERVATION THAT PROMPTED THIS. Ordered by the operating threshold each
pipeline ships with, every (pipeline, dataset) pair measured in this project is
monotone in the gain:

    ConInfer  LoveDA        tau 0.8    +2.51
    SAM 3     LoveDA        tau 0.5    +1.18
    SAM 3     Potsdam       tau 0.1    +0.60
    SAM 3     OpenEarthMap  tau 0.1    +0.30
    ConInfer  OpenEarthMap  tau 0.1    -0.09

That ordering has never been reported, and there is a mechanism waiting for it
that was measured in WEEK1 §7.7 and then never connected to the method.

THE MECHANISM. A pixel is assigned to the catch-all two ways
(segearthov3_segmentor.predict, and ConInfer's postprocess_result is the same
shape):

    seg_pred = argmax(logits)
    seg_pred[max_vals < tau[seg_pred]] = bg_idx

so the final label is background iff `pred == bg` OR `conf < tau[pred]`. Only the
second is a threshold decision. **Lowering a threshold cannot change an argmax**,
so any pixel whose argmax already went to the catch-all is outside what per-class
tau can ever address -- at any tau vector, not just this one.

    REACHABLE    pred != bg and conf < tau[pred]   a threshold move recovers it
    UNREACHABLE  pred == bg                        no tau vector touches it

At tau = 0.5 on LoveDA the threshold is doing most of the work. At tau = 0.1 it
is barely doing any, so most of what is left must be argmax loss -- which the
method cannot fix BY CONSTRUCTION. If that is what the ordering above is made of,
the gain has a mechanism instead of a list.

⭐ WHY IT WOULD BE WORTH REAL MONEY. The reachable share is computable with NO
GROUND TRUTH AT ALL: it is just "of the pixels the model sent to the catch-all,
what fraction were below the threshold rather than argmax losses". §9f went
looking for a label-free rule predicting WHEN calibration pays, tested the
DISCARD RATE, and found a U-shape on LoveDA and the opposite sign on OEM. It
concluded no such rule exists.

⚠️ **Reachable share is not the discard rate, and this script must prove that
before anything else it prints means anything.** If the two are collinear, this
is a restatement of a statistic §9f already killed, and the verdict says so. The
collinearity check is reported first, on purpose, so it cannot be skipped past.

⚠️ THE TAUTOLOGY RISK, stated plainly. A higher tau mechanically discards more AND
makes a larger fraction of that discard threshold-driven. So part of the ordering
above is arithmetic rather than a finding. What makes the test non-trivial is
that discard rate and reachable share come apart -- §9f measured the first and
found nothing. Reporting both against the same gains is the whole design.

TWO THINGS IT ALSO SETTLES, per class:

  §9g found the precision-recall gap ranks which classes move (rho +0.713,
  p 0.013 over 12 cells) -- then Potsdam produced `tree` at a +54.7 gap, LARGER
  than LoveDA's `water` at +34.8 that drove the entire LoveDA result, and it
  gained +0.32, i.e. nothing. A class can be badly under-firing in a way no
  threshold reaches. The prediction here is that SELF-REACHABLE share separates
  `tree` from `water` where the P-R gap does not.

  The sharpest per-class statistic is self-reachability: of class c's pixels sent
  to the catch-all, the fraction with `pred == c` and `conf < tau`. Those are the
  ones lowering c's OWN threshold recovers CORRECTLY. A pixel of class c whose
  argmax is a different real class comes back wearing the wrong label and does
  not help c's IoU.

USAGE. One pass over each cache builds the (gt, pred, conf-bin) histogram, which
is a sufficient statistic for both halves -- the reachability arithmetic and the
5-fold gain -- so this costs one cache read per dataset and no GPU.

    # one row per (pipeline, dataset), appended to a shared summary
    python scripts/reachability.py --cache ~/outputs/week3_fused/cache --tau 0.5 \\
        --tag "SAM3/LoveDA" --append ~/outputs/week4/reach_rows.json \\
        --md ~/outputs/week4/reach_loveda.md

    python scripts/reachability.py --cache ~/outputs/oem_tau0.1/cache --tau 0.1 \\
        --tag "SAM3/OEM" --append ~/outputs/week4/reach_rows.json \\
        --md ~/outputs/week4/reach_oem.md
    ... Potsdam, ConInfer LoveDA (--tau 0.8), ConInfer OEM (--tau 0.1) ...

    # then the cross-dataset test
    python scripts/reachability.py --summarize ~/outputs/week4/reach_rows.json \\
        --md ~/outputs/week4/reachability.md

⚠️ `conf` is cached as float16, so pixels within ~0.03% of a bin edge can land on
the wrong side. Immaterial at these magnitudes and stated for the record; it is
the same caveat that governs every tau sweep in this project.
"""
import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels                                                    # noqa: E402
from tau_oracle import confusion_at, per_class_iou, miou, NBINS  # noqa: E402
from tau_cv import per_tile_hists, fit                           # noqa: E402


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def _rank(x):
    """Ranks with TIES AVERAGED.

    ⚠️ The obvious `argsort(argsort(x))` is wrong the moment a value repeats: it
    hands tied entries distinct ranks in array order, so a CONSTANT vector gets
    ranks 0..n-1 and correlates with whatever the caller happened to pass. That
    is not hypothetical here -- ConInfer/OpenEarthMap has self-reachable = 0 for
    every class, and the first version of this script reported rho = +0.214 for
    it, which was the class ordering and nothing else.
    """
    x = np.asarray(x, float)
    n = len(x)
    order = np.argsort(x, kind='mergesort')
    r = np.empty(n, float)
    r[order] = np.arange(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(a, b):
    """nan when either side is constant -- a rank correlation is undefined there,
    and returning a number invites it to be read as evidence."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float('nan')
    ra, rb = _rank(a), _rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = math.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float('nan')


def fmt_rho(r, p=None, how=None):
    """A constant variable prints as undefined, never as a number."""
    if not np.isfinite(r):
        # keep the cell count -- the caller may be filling a "rho | p" pair
        return '— *(undefined: constant)*' + (' | —' if p is not None else '')
    return f'{r:+.3f}' + (f' | {p:.3f} *({how})*' if p is not None else '')


def perm_p(a, b, exact_cap=8, draws=20000, seed=0):
    """Two-sided p by permutation, exact where affordable. Same discipline as
    §9d/§9g: at this many points a table lookup is misleading."""
    n, r = len(a), abs(spearman(a, b))
    if not np.isfinite(r):
        return float('nan'), 'undefined'
    bl = list(b)
    if n <= exact_cap:
        hits = sum(1 for pm in itertools.permutations(range(n))
                   if abs(spearman(a, [bl[i] for i in pm])) >= r - 1e-12)
        return hits / math.factorial(n), 'exact'
    rng = np.random.default_rng(seed)
    arr = np.asarray(bl, float)
    hits = sum(1 for _ in range(draws)
               if abs(spearman(a, rng.permutation(arr))) >= r - 1e-12)
    return (hits + 1) / (draws + 1), 'sampled'


def reach_stats(H, bg, tau, nbins):
    """Partition every catch-all assignment into REACHABLE and UNREACHABLE.

    H is the summed (gt, pred, conf-bin) histogram over 0-indexed class ids.
    All quantities are returned per GROUND-TRUTH class; marginalising over that
    axis gives the label-free version, since the split itself never consults gt.

    The bin edge uses np.rint, matching confusion_at exactly -- truncation was a
    real bug there (see tau_oracle) and the two must agree or the reachability
    numbers would describe a different threshold than the gain does.
    """
    e = int(np.clip(np.rint(float(tau) * nbins), 0, nbins))
    nc = H.shape[0]

    at_all = H.sum(2)                       # (gt, pred) any confidence
    below = H[:, :, :e].sum(2)              # (gt, pred) conf < tau

    unreach = at_all[:, bg].astype(np.int64)                 # argmax lost: untouchable
    reach = (below.sum(1) - below[:, bg]).astype(np.int64)   # a real argmax, below tau
    to_bg = unreach + reach                                  # assigned to the catch-all

    # WEEK1 §7.7's A/B split, kept so this reconciles with that section.
    # Note reach is a STRICT SUBSET of mech_A: a pixel with pred == bg and
    # conf < tau is mechanism (A) yet still unreachable, because the argmax is
    # what condemns it. That gap is exactly why "reachable" is the sharper
    # statistic, and it is reported rather than assumed to be zero.
    mech_a = below.sum(1).astype(np.int64)
    mech_b = (at_all[:, bg] - below[:, bg]).astype(np.int64)

    # self-reachable: gt == pred == c and below tau. Lowering c's OWN threshold
    # recovers these with the CORRECT label. A pixel of class c predicted as some
    # other real class comes back wrong and does not help c's IoU.
    self_reach = np.zeros(nc, np.int64)
    for c in range(nc):
        if c != bg:
            self_reach[c] = int(below[c, c])

    return dict(gt_tot=H.sum(axis=(1, 2)).astype(np.int64), to_bg=to_bg,
                reach=reach, unreach=unreach, mech_a=mech_a, mech_b=mech_b,
                self_reach=self_reach, edge=e)


def pct(num, den):
    return 100.0 * num / den if den else float('nan')


# --------------------------------------------------------------------------- #
# per-cache mode
# --------------------------------------------------------------------------- #
def run_cache(args):
    LB = labels.from_cache(args.cache)
    nc, bg = LB.n, LB.bg - 1
    print(f'  classes: {LB}')

    files = sorted(Path(args.cache).expanduser().glob('*.npz'))
    if args.limit:
        files = files[:args.limit]
    n = len(files)
    if not n:
        raise SystemExit(f'no .npz under {args.cache}')
    tag = args.tag or Path(args.cache).expanduser().parent.name
    print(f'{n} tiles | published τ = {args.tau} | tag "{tag}"\n')

    PT = per_tile_hists(files, nc, NBINS)
    H = PT.sum(0).astype(np.int64)
    R = reach_stats(H, bg, args.tau, NBINS)

    real = [c for c in range(nc) if c != bg]

    # ---- label-free: marginalise the gt axis away. No ground truth is consulted.
    lf_to_bg = int(R['to_bg'].sum())
    lf_reach = int(R['reach'].sum())
    lf_share = pct(lf_reach, lf_to_bg)
    # denominator for a discard-style rate: all labelled pixels
    lf_all = int(R['gt_tot'].sum())
    lf_bg_frac = pct(lf_to_bg, lf_all)

    # ---- labelled: the residual proper, restricted to real-class ground truth
    res_to_bg = int(R['to_bg'][real].sum())
    res_reach = int(R['reach'][real].sum())
    res_self = int(R['self_reach'][real].sum())
    real_tot = int(R['gt_tot'][real].sum())
    discard_rate = pct(res_to_bg, real_tot)

    print('=' * 72)
    print(f'LABEL-FREE (no ground truth used)')
    print(f'  assigned to catch-all            {lf_to_bg:>16,}  '
          f'({lf_bg_frac:.2f}% of labelled px)')
    print(f'  reachable  (real argmax, < τ)    {lf_reach:>16,}  '
          f'({lf_share:.2f}% of those)')
    print(f'  unreachable (argmax lost)        {lf_to_bg - lf_reach:>16,}  '
          f'({100 - lf_share:.2f}%)')
    print('-' * 72)
    print(f'LABELLED residual (real-class GT only)')
    print(f'  real-class px assigned to bg     {res_to_bg:>16,}  '
          f'({discard_rate:.2f}% of real-class px)')
    print(f'  reachable                        {res_reach:>16,}  '
          f'({pct(res_reach, res_to_bg):.2f}% of the residual)')
    print(f'  self-reachable (correct label)   {res_self:>16,}  '
          f'({pct(res_self, res_to_bg):.2f}%)')
    print(f'  §7.7 (A) threshold               {int(R["mech_a"][real].sum()):>16,}')
    print(f'  §7.7 (B) argmax at conf ≥ τ      {int(R["mech_b"][real].sum()):>16,}')
    print('=' * 72)
    inert = int(R['mech_a'].sum()) == 0
    if inert:
        print('\n⛔ THE PUBLISHED THRESHOLD IS INERT ON THIS CACHE.\n'
              '   Mechanism (A) is EXACTLY zero: not one pixel scores below τ, so the\n'
              '   threshold never fires and every catch-all assignment is an argmax\n'
              '   loss. τ is below this pipeline\'s score floor on this dataset.\n'
              '   This row cannot carry a reachability correlation -- reachable share\n'
              '   is 0 by construction, not by measurement. It IS a real observation\n'
              '   about the published method and worth reporting as one.\n')
    print()

    # ---- the gain, computed here rather than transcribed from another table
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n)
    folds = np.array_split(order, args.folds)
    gains, pcs = [], []
    for k in range(args.folds):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(args.folds) if j != k])
        taus = fit(PT[tr].sum(0).astype(np.int64), bg, NBINS, objective=args.objective)
        Cb = confusion_at(PT[te].sum(0).astype(np.int64), np.full(nc, args.tau), bg, NBINS)
        Cf = confusion_at(PT[te].sum(0).astype(np.int64), taus, bg, NBINS)
        b, g = miou(Cb), miou(Cf)
        gains.append(g - b)
        pcs.append(per_class_iou(Cf) - per_class_iou(Cb))
        print(f'  fold {k + 1}: {b:.2f} -> {g:.2f}  ({g - b:+.2f})')
    gains = np.array(gains)
    d_iou = np.nanmean(np.array(pcs), axis=0)

    # ---- per-class statistics at the published threshold
    C0 = confusion_at(H, np.full(nc, args.tau), bg, NBINS)
    prec = np.full(nc, np.nan)
    rec = np.full(nc, np.nan)
    for c in range(nc):
        if C0[:, c].sum():
            prec[c] = 100.0 * C0[c, c] / C0[:, c].sum()
        if C0[c, :].sum():
            rec[c] = 100.0 * C0[c, c] / C0[c, :].sum()
    pr_gap = prec - rec

    md = [f'# Reachability — is the gain bounded by what a threshold can touch?\n',
          f'- cache: `{args.cache}`  |  tag: **{tag}**  |  tiles: **{n}**',
          f'- published τ: **{args.tau}**  |  classes: **{nc}**, '
          f'catch-all `{LB.names[bg]}`',
          f'- fit objective: **`{args.objective}`**, {args.folds}-fold\n',
          '`reachable` = the argmax named a real class and only the threshold blocked it. ',
          '`unreachable` = the argmax went to the catch-all, which **no τ vector can '
          'change**.\n',
          '## Label-free — computed without any ground truth\n',
          '| quantity | pixels | share |', '|---|---|---|',
          f'| assigned to catch-all | {lf_to_bg:,} | {lf_bg_frac:.2f}% of labelled px |',
          f'| **reachable** | {lf_reach:,} | **{lf_share:.2f}%** of those |',
          f'| unreachable | {lf_to_bg - lf_reach:,} | {100 - lf_share:.2f}% |',
          '\n**This is the deployable statistic** — a practitioner can compute it from a '
          'forward pass over unlabelled tiles.\n',
          '## Labelled — the residual proper\n',
          '| quantity | pixels | share of the residual |', '|---|---|---|',
          f'| real-class px assigned to catch-all | {res_to_bg:,} | '
          f'({discard_rate:.2f}% of real-class px) |',
          f'| reachable | {res_reach:,} | **{pct(res_reach, res_to_bg):.2f}%** |',
          f'| self-reachable (recovered with the *right* label) | {res_self:,} | '
          f'**{pct(res_self, res_to_bg):.2f}%** |',
          f'| §7.7 (A) below threshold | {int(R["mech_a"][real].sum()):,} | '
          f'{pct(int(R["mech_a"][real].sum()), res_to_bg):.2f}% |',
          f'| §7.7 (B) argmax at conf ≥ τ | {int(R["mech_b"][real].sum()):,} | '
          f'{pct(int(R["mech_b"][real].sum()), res_to_bg):.2f}% |',
          '\n⚠️ `reachable` is a **strict subset** of §7.7\'s mechanism (A): a pixel below '
          'the threshold whose argmax was already the catch-all is (A) and still '
          'untouchable. The gap between those two rows is the reason this statistic is '
          'sharper than the A/B split.\n',
          f'## The gain, {args.folds}-fold on this cache\n',
          f'**Δ = {gains.mean():+.2f} mIoU, sd {gains.std(ddof=1):.2f}**, '
          f'{int((gains > 0).sum())}/{args.folds} folds positive, '
          f'range {gains.min():+.2f} to {gains.max():+.2f}.\n',
          'Computed here rather than transcribed, so the reachability columns and the '
          'gain describe the same tiles and the same threshold.\n',
          '## Per class\n',
          '| class | GT px | discarded | reachable | **self-reachable** | P−R gap | '
          '**Δ IoU** |', '|---|---|---|---|---|---|---|']

    rows = []
    for c in real:
        tb, gtt = int(R['to_bg'][c]), int(R['gt_tot'][c])
        sr = pct(int(R['self_reach'][c]), tb)
        md.append(f'| {LB.names[c]} | {gtt:,} | {pct(tb, gtt):.1f}% | '
                  f'{pct(int(R["reach"][c]), tb):.1f}% | **{sr:.1f}%** | '
                  f'{pr_gap[c]:+.1f} | **{d_iou[c]:+.2f}** |')
        if np.isfinite(d_iou[c]) and np.isfinite(pr_gap[c]) and tb:
            rows.append((sr, float(pr_gap[c]), float(d_iou[c])))
    md.append(f'| *{LB.names[bg]}* *(catch-all)* | {int(R["gt_tot"][bg]):,} | — | — | — | '
              f'{pr_gap[bg]:+.1f} | *{d_iou[bg]:+.2f}* |')

    # ---- which per-class statistic ranks the movers?
    md.append('\n## Which per-class statistic ranks the movers?\n')
    if len(rows) >= 4:
        sr_v = [r[0] for r in rows]
        pg_v = [r[1] for r in rows]
        di_v = [r[2] for r in rows]
        r1, (p1, h1) = spearman(sr_v, di_v), perm_p(sr_v, di_v)
        r2, (p2, h2) = spearman(pg_v, di_v), perm_p(pg_v, di_v)
        md += [f'Over the {len(rows)} real classes, against Δ IoU:\n',
               '| statistic | ρ | p |', '|---|---|---|',
               f'| **self-reachable share** | {fmt_rho(r1, p1, h1)} |',
               f'| precision−recall gap *(§9g\'s statistic)* | {fmt_rho(r2, p2, h2)} |',
               f'\n⚠️ {len(rows)} points. This is a direction, not a law — the '
               'cross-dataset test in `--summarize` is what the claim rests on.\n']
        percls = dict(self_reach_rho=r1, self_reach_p=p1, prgap_rho=r2, prgap_p=p2)
    else:
        md.append(f'Only {len(rows)} real classes — too few to rank. Skipped.\n')
        percls = {}

    row = dict(tag=tag, cache=str(args.cache), tau=float(args.tau), tiles=int(n),
               classes=int(nc), catch_all=LB.names[bg],
               lf_reach_share=float(lf_share), lf_bg_frac=float(lf_bg_frac),
               res_reach_share=float(pct(res_reach, res_to_bg)),
               res_self_share=float(pct(res_self, res_to_bg)),
               discard_rate=float(discard_rate),
               gain=float(gains.mean()), gain_sd=float(gains.std(ddof=1)),
               folds_positive=int((gains > 0).sum()), folds=int(args.folds),
               objective=args.objective, per_class=percls)

    if args.append:
        path = Path(args.append).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads(path.read_text()) if path.exists() else []
        data = [d for d in data if d.get('tag') != tag] + [row]
        path.write_text(json.dumps(data, indent=2))
        print(f'\nappended "{tag}" -> {path}  ({len(data)} rows)')

    write_md(args.md, md)
    return row


# --------------------------------------------------------------------------- #
# cross-dataset mode
# --------------------------------------------------------------------------- #
def run_summary(args):
    path = Path(args.summarize).expanduser()
    data = json.loads(path.read_text())
    data.sort(key=lambda d: -d['lf_reach_share'])
    k = len(data)
    print(f'{k} (pipeline, dataset) rows from {path}\n')

    reach = [d['lf_reach_share'] for d in data]
    disc = [d['discard_rate'] for d in data]
    gain = [d['gain'] for d in data]
    taus = [d['tau'] for d in data]

    md = ['# Reachability across pipelines and datasets\n',
          f'- rows: **{k}**  |  source: `{path}`\n',
          'Every row was produced by one pass over that pipeline\'s own cache, at its '
          'own published τ, with the gain computed on the same tiles. Nothing is '
          'transcribed between tables.\n',
          '| tag | τ | classes | **reachable (label-free)** | discard rate | '
          '**Δ mIoU** | folds + |', '|---|---|---|---|---|---|---|']
    for d in data:
        md.append(f'| {d["tag"]} | {d["tau"]:g} | {d["classes"]} | '
                  f'**{d["lf_reach_share"]:.1f}%** | {d["discard_rate"]:.1f}% | '
                  f'**{d["gain"]:+.2f}** ± {d["gain_sd"]:.2f} | '
                  f'{d["folds_positive"]}/{d["folds"]} |')

    if k < 3:
        md.append('\n⛔ Fewer than three rows. No correlation is reportable.\n')
        write_md(args.md, md)
        return

    # ---- the collinearity check comes FIRST, on purpose
    keep = [i for i, d in enumerate(data) if d['lf_reach_share'] > 0.0]
    sub = lambda v: [v[i] for i in keep]

    r_rd, (p_rd, h_rd) = spearman(reach, disc), perm_p(reach, disc)
    inert = [d['tag'] for d in data if d['lf_reach_share'] == 0.0]
    if inert:
        md += ['\n⛔ **Rows with an INERT published threshold: '
               + ', '.join(f'`{x}`' for x in inert) + '.**\n',
               'Reachable share is 0 there *by construction* — τ sits below the score '
               'floor, so the threshold never fires on a single pixel and every '
               'catch-all assignment is an argmax loss. Such a row is a real finding '
               'about that published configuration, but it **cannot carry a '
               'correlation**, and a ρ computed with it in is measuring one point. '
               'Every ρ below is reported twice: with, and without.\n']

    md += ['\n## First: is this just the discard rate again?\n',
           '§9f already tested the **discard rate** as a label-free predictor of when '
           'calibration pays, and found a U-shape on LoveDA and the opposite sign on '
           'OpenEarthMap. If reachable share is collinear with it, this experiment is a '
           'restatement of a closed negative.\n',
           f'ρ(reachable share, discard rate) = **{fmt_rho(r_rd)}**, '
           f'p = {p_rd:.3f} *({h_rd})*\n']
    collinear = np.isfinite(r_rd) and abs(r_rd) >= 0.9
    md.append('⛔ **They are collinear.** Treat everything below as a restatement of '
              '§9f, not a new statistic.\n' if collinear else
              '✅ **They come apart.** The two statistics rank the rows differently, so '
              'this is not §9f\'s measurement under another name.\n')

    r_g, (p_g, h_g) = spearman(reach, gain), perm_p(reach, gain)
    r_d, (p_d, h_d) = spearman(disc, gain), perm_p(disc, gain)
    r_t, (p_t, h_t) = spearman(taus, gain), perm_p(taus, gain)

    kn = len(keep)
    rg2 = spearman(sub(reach), sub(gain)) if kn >= 3 else float('nan')
    rd2 = spearman(sub(disc), sub(gain)) if kn >= 3 else float('nan')
    rt2 = spearman(sub(taus), sub(gain)) if kn >= 3 else float('nan')
    md += ['## Against the gain\n',
           f'| predictor | ρ (all {k} rows) | p | ρ (the {kn} live rows) |',
           '|---|---|---|---|',
           f'| **reachable share** | **{fmt_rho(r_g)}** | {p_g:.3f} *({h_g})* '
           f'| **{fmt_rho(rg2)}** |',
           f'| discard rate *(§9f, closed negative)* | {fmt_rho(r_d)} | {p_d:.3f} '
           f'*({h_d})* | {fmt_rho(rd2)} |',
           f'| published τ | {fmt_rho(r_t)} | {p_t:.3f} *({h_t})* | {fmt_rho(rt2)} |',
           '\n"Live rows" excludes any row whose threshold is inert, where reachable '
           'share is 0 by construction. ⭐ **The right-hand column is the honest one** '
           '— if the correlation only exists in the left, it is one degenerate point.\n',
           f'\n⚠️ **{k} points.** A correlation over {k} rows is an ordering, not a law; '
           f'the smallest attainable two-sided p is '
           f'{2 / math.factorial(k):.3f}. What this table can do is say whether the '
           'ordering is there and whether the rival explanation is better — it cannot '
           'establish the relationship on its own.\n']

    # ---- verdict. Every branch is computed from THIS run; nothing quoted from
    # another dataset. (tau_cv printed a LoveDA figure on four foreign caches
    # before that was caught — WEEK3 §11.)
    md.append('## Verdict\n')
    r_use = rg2 if np.isfinite(rg2) else r_g
    d_use = rd2 if np.isfinite(rd2) else r_d
    if not np.isfinite(r_use):
        md.append('⛔ **No usable correlation.** Too few live rows once inert '
                  'thresholds are excluded. Add rows before reading anything here.\n')
    elif collinear:
        md.append(f'⛔ **Reachable share is collinear with the discard rate** '
                  f'(ρ {r_rd:+.3f}). §9f measured that and found no rule. This adds a '
                  'name, not a finding. Report it as a negative and do not build the '
                  'pre-registration on it.\n')
    elif abs(r_use) >= 0.8 and abs(r_use) > abs(d_use) + 0.2:
        md += [f'⭐ **Reachable share orders the gain (ρ {r_use:+.3f}, live rows) and '
               f'beats the discard rate (ρ {d_use:+.3f}) by a clear margin, while not '
               f'being collinear with it (ρ {r_rd:+.3f}).**\n',
               'That is the discriminating outcome this script was built to test: a '
               '**label-free** statistic that orders the gain where §9f\'s did not.\n',
               '⛔ **It is not established yet, and must not be written up as if it '
               f'were.** {k} points, chosen after the gains were known. The next step is '
               'the only one that converts it into evidence:\n',
               '1. Commit the predicted Δ for a held-out dataset **before running it**, '
               'from its reachable share alone.\n2. Run Vaihingen.\n3. Report the '
               'prediction against the measurement, held or failed.\n',
               'That is the Potsdam protocol, applied to a claim that would be new.\n']
    elif abs(d_use) > abs(r_use) + 0.2:
        md.append(f'⛔ **The discard rate orders the gain better (ρ {d_use:+.3f}) than '
                  f'reachable share does (ρ {r_use:+.3f}).** The reachability framing adds '
                  'nothing over the statistic §9f already closed. Report as a bounded '
                  'negative and move on.\n')
    else:
        md.append(f'⚠️ **Inconclusive.** Reachable share ρ {r_use:+.3f}, discard rate '
                  f'ρ {d_use:+.3f} — neither separates from the other at {kn} live points. '
                  'Either add rows (Vaihingen, and the ConInfer arms if any are '
                  'missing) or report that the ordering has no single predictor.\n')

    if np.isfinite(r_use) and np.isfinite(rt2) and abs(rt2) >= 0.8:
        md.append(f'⚠️ **The published τ alone orders the gain at ρ {rt2:+.3f} on the '
                  'live rows.** A '
                  'higher τ mechanically discards more *and* makes more of that discard '
                  'threshold-driven, so part of any reachability result is arithmetic. '
                  'Say this beside the ρ, and prefer the per-class evidence — which is '
                  'within a single operating point and therefore free of it.\n')

    # ---- per-class evidence pooled across datasets
    pcs = [d for d in data if d.get('per_class')]
    if pcs:
        md += ['\n## Per-class evidence, one row per dataset\n',
               'Within a dataset the operating point is fixed, so these are free of the '
               'τ confound above.\n',
               '| tag | ρ self-reachable | p | ρ P−R gap *(§9g)* | p |',
               '|---|---|---|---|---|']
        for d in pcs:
            q = d['per_class']
            sr, pg = q['self_reach_rho'], q['prgap_rho']
            md.append(f'| {d["tag"]} | **{fmt_rho(sr)}** | '
                      + (f'{q["self_reach_p"]:.3f}' if np.isfinite(sr) else '—')
                      + f' | {fmt_rho(pg)} | '
                      + (f'{q["prgap_p"]:.3f}' if np.isfinite(pg) else '—') + ' |')
        cmp_ok = [d for d in pcs
                  if np.isfinite(d['per_class']['self_reach_rho'])
                  and np.isfinite(d['per_class']['prgap_rho'])]
        wins = sum(1 for d in cmp_ok
                   if abs(d['per_class']['self_reach_rho'])
                   > abs(d['per_class']['prgap_rho']))
        pcs = cmp_ok
        md.append(f'\nSelf-reachability ranks the classes better than the P−R gap in '
                  f'**{wins} of {len(pcs)}** datasets where both are defined.\n')
        if wins == len(pcs) and len(pcs) >= 3:
            md.append('⭐ It wins everywhere. §9g\'s P−R gap is then the *symptom* — a '
                      'class under-fires — and reachability is the *constraint*: whether '
                      'a threshold can do anything about it. That would explain Potsdam\'s '
                      '`tree`, which has the larger P−R gap and does not move.\n')
        elif wins == 0:
            md.append('⛔ It loses everywhere. §9g\'s P−R gap stands as the better '
                      'per-class statistic, and Potsdam\'s `tree` stays unexplained. '
                      'Report that.\n')

    write_md(args.md, md)


def write_md(dest, md):
    text = '\n'.join(md) + '\n'
    print('\n' + text)
    if dest:
        p = Path(dest).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f'wrote {p}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', help='a measure_discard_rate.py / coninfer_cache.py cache')
    ap.add_argument('--tau', type=float, help="this pipeline's PUBLISHED threshold")
    ap.add_argument('--tag', help='e.g. "SAM3/LoveDA"; defaults to the cache\'s parent dir')
    ap.add_argument('--append', help='JSON file to accumulate one row per cache')
    ap.add_argument('--summarize', help='run the cross-dataset test on that JSON instead')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--objective', choices=['all', 'real'], default='real',
                    help="what the FIT maximises. 'real' excludes the catch-all, which "
                         "is what every headline in WEEK3 §9b onward uses.")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0,
                    help='⚠️ smoke test only — takes the FIRST n filenames, which on '
                         'LoveDA is essentially rural-only. Never use it for a result.')
    ap.add_argument('--md', default=None)
    args = ap.parse_args()

    if args.summarize:
        run_summary(args)
    elif args.cache and args.tau is not None:
        if args.limit:
            print('⚠️  --limit is a SMOKE TEST. The subset is filename-ordered, not '
                  'representative. Do not quote anything from this run.\n')
        run_cache(args)
    else:
        ap.error('give --cache with --tau, or --summarize')


if __name__ == '__main__':
    main()
