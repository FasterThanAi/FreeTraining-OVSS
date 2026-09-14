"""Does a per-class threshold vector fitted on one split work on another?

WEEK3 §9e is blunt: thresholds do NOT transfer across LoveDA's domains, and the
mismatched arm lands *below* the published tau. §9b found the same across LoveDA's
own train/val boundary (-0.12). The diagnosis in both cases was distribution shift,
and for LoveDA it is measurable: train discards 14.54% of real-class pixels and val
discards 29.68% -- 2.04x -- at an identical background share.

⭐ UAVid is the first split pair in this project where that diagnosis predicts SUCCESS:
train discards 6.73% and val 6.81%, a gap of 0.08 percentage points. So this is a test
of the DIAGNOSIS, not just of the method. If transfer fails here, "calibrate on the
distribution you will evaluate on" needs a sharper statement than discard-rate matching.

Four arms, every one evaluated on the SAME destination tiles:

  A  published tau                                  the baseline
  B  best GLOBAL tau fitted on the source           ⭐ the control that matters:
                                                    if B == C, the per-class part
                                                    transfers nothing and only the
                                                    level does
  C  per-class tau fitted on the source             THE TEST
  D  per-class oracle on the destination            an upper bound, NOT a method

plus a permutation control (C's thresholds shuffled among the real classes), which
asks whether the class ASSIGNMENT transfers or merely the spread of values.

Example:

  python scripts/tau_transfer.py \
    --src-cache ~/outputs/uavid_train/cache --dst-cache ~/outputs/uavid_val/cache \
    --tau 0.3 --objective real \
    --src-exclude-re '_(flipped|shifted)' \
    --src-group-re '([a-z]+[0-9]*)(?=[-_][0-9]+$)' \
    --expect 56.87 --md ~/outputs/uavid_val/tau_transfer.md
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels                                                    # noqa: E402
from tau_oracle import confusion_at, miou, per_class_iou, NBINS   # noqa: E402
from tau_cv import fit, per_tile_hists                            # noqa: E402


def load(cache, exclude_re, what):
    files = sorted(Path(cache).expanduser().glob('*.npz'))
    if not files:
        raise SystemExit(f'no .npz under {cache}')
    if exclude_re:
        ex = re.compile(exclude_re)
        before = len(files)
        files = [f for f in files if not ex.search(f.stem)]
        if not files:
            raise SystemExit(f'--{what}-exclude-re {exclude_re!r} removed all '
                             f'{before} files')
        print(f'  {what}: --exclude-re {exclude_re!r} dropped '
              f'{before - len(files)} of {before}')
    print(f'  {what}: {len(files)} tiles from {cache}')
    return files


def groups_of(files, group_re, what):
    """Scene id per file, so budget draws take whole scenes."""
    if not group_re:
        return None, None
    pat = re.compile(group_re)
    keys = []
    for f in files:
        m = pat.search(f.stem)
        if m is None:
            raise SystemExit(f'--{what}-group-re {group_re!r} matched nothing on '
                             f'{f.stem!r} -- every tile must have a group')
        keys.append(m.group(1) if m.groups() else m.group(0))
    uniq = sorted(set(keys))
    pos = {g: i for i, g in enumerate(uniq)}
    gid = np.array([pos[k] for k in keys])
    print(f'  {what}: {len(uniq)} groups (sizes '
          f'{np.bincount(gid).min()}-{np.bincount(gid).max()})')
    return gid, uniq


def excl_miou(C, bg):
    """mIoU over the real classes only -- reported BESIDE full mIoU, never instead."""
    v = per_class_iou(C)
    v = np.array([v[c] for c in range(len(v)) if c != bg])
    return float(np.nanmean(v)) if np.isfinite(v).any() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src-cache', required=True, help='fit the thresholds here')
    ap.add_argument('--dst-cache', required=True, help='evaluate them here')
    ap.add_argument('--tau', type=float, required=True,
                    help="the dataset's published tau, for the baseline arm")
    ap.add_argument('--objective', choices=['all', 'real'], default='real')
    ap.add_argument('--src-exclude-re', default=None)
    ap.add_argument('--dst-exclude-re', default=None)
    ap.add_argument('--src-group-re', default=None,
                    help='scene id regex, so budget draws take whole scenes '
                         'rather than correlated frames')
    ap.add_argument('--sizes', type=int, nargs='+',
                    default=[25, 50, 100, 200],
                    help='source-tile budgets for the transfer curve')
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--perms', type=int, default=200,
                    help='permutation-control draws')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--expect', type=float, default=None,
                    help='known published-tau mIoU on the DESTINATION. Aborts on a '
                         'mismatch beyond --expect-tol. This is the gate: the two '
                         'caches may be at different --cache-stride settings, and a '
                         'label-convention slip would otherwise pass silently.')
    ap.add_argument('--expect-tol', type=float, default=0.15)
    ap.add_argument('--deploy-cfg', default=None,
                    help='write an mmseg config carrying arm C\'s threshold '
                         'vector, so the number can be verified end-to-end by '
                         'eval.py instead of only in cached arithmetic (WEEK3 '
                         '§9c). Needs --base-cfg.')
    ap.add_argument('--base-cfg', default=None,
                    help='the config to copy for --deploy-cfg (e.g. '
                         '~/SegEarth-OV-3/configs/cfg_uavid_val.py)')
    ap.add_argument('--md', default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # ---- gate 1: the two caches must describe the SAME label space ----------
    LS = labels.from_cache(args.src_cache)
    LD = labels.from_cache(args.dst_cache)
    if LS.names != LD.names:
        raise SystemExit(f'class lists differ between caches.\n'
                         f'  src: {LS.names}\n  dst: {LD.names}\n'
                         f'A threshold vector is indexed by class; transferring it '
                         f'across different label spaces is meaningless.')
    if LS.bg != LD.bg:
        raise SystemExit(f'catch-all index differs: src {LS.bg}, dst {LD.bg}')
    nc, bg = LD.n, LD.bg - 1
    print(f'  classes: {LD}')

    src_files = load(args.src_cache, args.src_exclude_re, 'src')
    dst_files = load(args.dst_cache, args.dst_exclude_re, 'dst')
    if {f.name for f in src_files} & {f.name for f in dst_files}:
        raise SystemExit('src and dst caches share filenames -- these are not '
                         'disjoint splits and the transfer number would be a leak')
    gid, _ = groups_of(src_files, args.src_group_re, 'src')

    print('  building source histograms')
    HS = per_tile_hists(src_files, nc, NBINS)
    print('  building destination histograms')
    HD = per_tile_hists(dst_files, nc, NBINS).sum(0).astype(np.int64)

    def ev(taus):
        return confusion_at(HD, taus, bg, NBINS)

    # ---- gate 2: the published-tau row must reproduce the known number -----
    C_pub = ev(np.full(nc, args.tau))
    m_pub, e_pub = miou(C_pub), excl_miou(C_pub, bg)
    print(f'\n  published τ={args.tau} on dst: mIoU {m_pub:.2f} '
          f'(catch-all-excluded {e_pub:.2f})')
    if args.expect is not None:
        d = abs(m_pub - args.expect)
        if d > args.expect_tol:
            raise SystemExit(
                f'GATE FAILED: published-τ mIoU on dst is {m_pub:.2f}, expected '
                f'{args.expect:.2f} (|Δ| {d:.2f} > {args.expect_tol}). Something '
                f'about the cache, the label convention or τ is wrong; every '
                f'number below would be void.')
        print(f'  ✅ gate: within {d:.2f} of the expected {args.expect:.2f}')

    # ---- arms ---------------------------------------------------------------
    HS_all = HS.sum(0).astype(np.int64)

    # B: best GLOBAL tau on the source
    from tau_cv import obj_miou
    grid = np.arange(NBINS + 1) / NBINS
    _, t_glob = max((obj_miou(confusion_at(HS_all, np.full(nc, t), bg, NBINS),
                              bg, args.objective), t) for t in grid)
    C_glob = ev(np.full(nc, t_glob))

    # C: per-class tau on the source
    taus_src = fit(HS_all, bg, NBINS, objective=args.objective)
    C_tr = ev(taus_src)

    # D: per-class oracle on the destination (a BOUND, not a method)
    taus_orc = fit(HD, bg, NBINS, objective=args.objective)
    C_orc = ev(taus_orc)

    rows = [('A  published τ', f'{args.tau:.3f} (global)', C_pub),
            ('B  global τ fitted on src', f'{t_glob:.3f} (global)', C_glob),
            ('C  per-class τ fitted on src', 'per class', C_tr),
            ('D  per-class oracle on dst', 'per class ⚠️ bound', C_orc)]

    print()
    for name, tau_s, C in rows:
        print(f'  {name:<30} {miou(C):6.2f}  ({miou(C) - m_pub:+.2f})   '
              f'excl {excl_miou(C, bg):6.2f}  ({excl_miou(C, bg) - e_pub:+.2f})')

    # ---- permutation control ------------------------------------------------
    # Does the class ASSIGNMENT transfer, or only the spread of values? Shuffle
    # C's thresholds among the real classes and re-evaluate.
    real = [c for c in range(nc) if c != bg]
    perm_d = []
    for _ in range(args.perms):
        t = taus_src.copy()
        t[real] = taus_src[np.array(real)[rng.permutation(len(real))]]
        perm_d.append(miou(ev(t)) - m_pub)
    perm_d = np.array(perm_d)
    real_d = miou(C_tr) - m_pub
    beat = float((perm_d >= real_d).mean())
    print(f'\n  permutation control ({args.perms} draws): mean {perm_d.mean():+.2f}, '
          f'p95 {np.percentile(perm_d, 95):+.2f}, max {perm_d.max():+.2f}')
    print(f'  the real assignment scores {real_d:+.2f}; '
          f'{100 * beat:.1f}% of shuffles match or beat it')

    # ---- transfer curve -----------------------------------------------------
    curve = []
    for sz in args.sizes:
        if sz > len(src_files):
            continue
        ds = []
        for _ in range(args.repeats):
            if gid is None:
                sel = rng.permutation(len(src_files))[:sz]
            else:
                take, cnt = [], 0
                for g in rng.permutation(gid.max() + 1):
                    if cnt >= sz:
                        break
                    take.append(g)
                    cnt += int((gid == g).sum())
                sel = np.where(np.isin(gid, take))[0]
            t = fit(HS[sel].sum(0).astype(np.int64), bg, NBINS,
                    objective=args.objective)
            ds.append(miou(ev(t)) - m_pub)
        ds = np.array(ds)
        curve.append((sz, ds.mean(), ds.std(ddof=1) if len(ds) > 1 else 0.0, ds.min()))
        print(f'  src n={sz}: {ds.mean():+.2f} ± '
              f'{ds.std(ddof=1) if len(ds) > 1 else 0:.2f} (worst {ds.min():+.2f})')

    # ---- report -------------------------------------------------------------
    pc_pub, pc_tr = per_class_iou(C_pub), per_class_iou(C_tr)
    md = [f'# Threshold transfer — fit on one split, evaluate on another\n',
          f'- source (fit): `{args.src_cache}`  |  **{len(src_files)}** tiles',
          f'- destination (evaluate): `{args.dst_cache}`  |  **{len(dst_files)}** tiles',
          f'- published τ: **{args.tau}**  |  fit objective **`{args.objective}`**  |  '
          f'classes: **{nc}**, catch-all `{LD.names[bg]}`\n',
          'The two splits share no filenames, so the source tiles are disjoint from '
          'the evaluation tiles by construction.\n']
    if args.expect is not None:
        md.append(f'✅ **Gate:** the published-τ row reproduces **{m_pub:.2f}** against '
                  f'the expected **{args.expect:.2f}**, so the two caches agree on the '
                  f'label convention despite any difference in `--cache-stride`.\n')
    md += ['## Arms — all evaluated on the same destination tiles\n',
           '| arm | τ | full mIoU | Δ | catch-all-excluded | Δ |', '|---|---|---|---|---|---|']
    for name, tau_s, C in rows:
        md.append(f'| {name} | {tau_s} | **{miou(C):.2f}** | **{miou(C) - m_pub:+.2f}** '
                  f'| {excl_miou(C, bg):.2f} | {excl_miou(C, bg) - e_pub:+.2f} |')
    md.append('\n⚠️ **Arm D chooses its thresholds on the evaluation labels.** It bounds '
              'what any transfer could achieve; it is not a method and must never be '
              'quoted as a result.\n')

    md += ['## Fitted thresholds and per-class effect (arm C)\n',
           '| class | τ fitted on src | IoU published | IoU transferred | Δ |',
           '|---|---|---|---|---|']
    for c in range(nc):
        t = '— *(no effect)*' if c == bg else f'{taus_src[c]:.3f}'
        md.append(f'| {LD.names[c]}{" *(catch-all)*" if c == bg else ""} | {t} | '
                  f'{pc_pub[c]:.2f} | {pc_tr[c]:.2f} | **{pc_tr[c] - pc_pub[c]:+.2f}** |')
    md.append('')

    md += ['## Permutation control — does the class ASSIGNMENT transfer?\n',
           f'C\'s thresholds shuffled among the {len(real)} real classes, '
           f'{args.perms} draws, evaluated the same way.\n',
           '| | Δ mIoU |', '|---|---|',
           f'| **real assignment (arm C)** | **{real_d:+.2f}** |',
           f'| shuffled, mean | {perm_d.mean():+.2f} |',
           f'| shuffled, p95 | {np.percentile(perm_d, 95):+.2f} |',
           f'| shuffled, max | {perm_d.max():+.2f} |',
           f'| shuffles matching or beating the real one | **{100 * beat:.1f}%** |\n']

    if curve:
        md += ['## How much source data does the transfer need?\n',
               ('Fit on whole SCENES until *n* source tiles are reached'
                if gid is not None else 'Fit on *n* source tiles at random') +
               f', {args.repeats} draws each, always evaluated on all '
               f'{len(dst_files)} destination tiles.\n',
               '| source tiles | mean Δ | sd | worst draw |', '|---|---|---|---|']
        for sz, mu, sd, wo in curve:
            md.append(f'| {sz} | **{mu:+.2f}** | {sd:.2f} | {wo:+.2f} |')
        md.append('')

    # ---- verdict, computed from this run and nothing else -------------------
    md.append('## Verdict\n')
    captured = (real_d / (miou(C_orc) - m_pub) * 100
                if miou(C_orc) - m_pub > 1e-9 else float('nan'))
    if real_d > 0 and beat < 0.05:
        md.append(f'✅ **Transfer works: {real_d:+.2f} full mIoU** '
                  f'({excl_miou(C_tr, bg) - e_pub:+.2f} catch-all-excluded), against a '
                  f'destination oracle of {miou(C_orc) - m_pub:+.2f} — '
                  f'**{captured:.0f}% of the available headroom**. Only '
                  f'{100 * beat:.1f}% of threshold shuffles match it, so what '
                  f'transfers is the per-class assignment, not the level.')
    elif real_d > 0:
        md.append(f'⚠️ **Transfer is positive ({real_d:+.2f}) but the permutation '
                  f'control is not beaten** — {100 * beat:.1f}% of shuffles match or '
                  f'exceed it. The gain is consistent with a change in the overall '
                  f'threshold LEVEL rather than with per-class structure carrying '
                  f'across the splits. Compare arm B, which is exactly that.')
    else:
        md.append(f'⛔ **Transfer FAILS: {real_d:+.2f} full mIoU.** A vector fitted on '
                  f'the source is worse on the destination than the published τ, so '
                  f'the "calibrate on the distribution you will evaluate on" rule '
                  f'holds here too.')
    if miou(C_glob) - m_pub >= real_d - 1e-9:
        md.append(f'\n⛔ **Arm B matches or beats arm C** ({miou(C_glob) - m_pub:+.2f} '
                  f'vs {real_d:+.2f}): a single global threshold fitted on the source '
                  f'transfers as well as the per-class vector, so the per-class part '
                  f'earns nothing across this split pair.')
    md.append('\n⚠️ Quote the per-class table, never the mean alone — a per-class rule '
              'can hurt a per-class result.')

    # ---- optional: emit a deployable config -------------------------------
    # ⛔ The vector is written straight from the fit. Retyping seven numbers out
    # of a markdown table is exactly how a silently misaligned threshold vector
    # gets deployed -- the segmentor's own length check would not catch a
    # PERMUTATION, and the permutation control above shows a shuffled vector
    # costs 3.08 mIoU while still producing a perfectly plausible table.
    if args.deploy_cfg:
        if not args.base_cfg:
            raise SystemExit('--deploy-cfg needs --base-cfg')
        from mmengine import Config
        cfg = Config.fromfile(str(Path(args.base_cfg).expanduser()))
        vec = [round(float(t), 4) for t in taus_src]
        vec[bg] = float(args.tau)        # bg's tau has no effect; keep it honest
        if len(vec) != nc:
            raise SystemExit(f'vector length {len(vec)} != {nc} classes')
        cfg.model.prob_thd = vec
        out = Path(args.deploy_cfg).expanduser()
        cfg.dump(str(out))
        print(f'\n  deploy config -> {out}')
        print('  prob_thd = ' + ', '.join(
            f'{LD.names[i]}={v:.3f}' for i, v in enumerate(vec)))
        print(f'  expect eval.py to report mIoU {miou(C_tr):.2f} '
              f'(published-τ run gives {m_pub:.2f})')
        md.append(f'\n## Deployed\n\n`{out}` carries arm C\'s vector, written '
                  f'from the fit rather than transcribed. Running `eval.py` on it '
                  f'should report **{miou(C_tr):.2f}** against the published-τ '
                  f'**{m_pub:.2f}**.\n\n| class | τ |\n|---|---|\n' +
                  '\n'.join(f'| {LD.names[i]} | {v:.3f}'
                             f'{" *(no effect)*" if i == bg else ""} |'
                             for i, v in enumerate(vec)))

    text = '\n'.join(md)
    print('\n' + text)
    if args.md:
        Path(args.md).expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).expanduser().write_text(text)
        print(f'\nwritten: {args.md}')


if __name__ == '__main__':
    main()
