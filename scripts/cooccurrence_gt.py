"""
Ground-truth co-occurrence analysis for land-cover masks (LoveDA).

THE question this answers
-------------------------
Your method assumes land cover has strong spatial-adjacency structure - that
knowing a patch borders "road" genuinely constrains what it can be. If adjacency
is close to what you'd get by arranging classes at random, the co-occurrence
prior carries no information and the whole approach collapses.

So this does not just plot a pretty matrix. It compares observed adjacency
against an independence baseline and reports pointwise mutual information:

    PMI(i,j) = log2( P_observed(i,j) / (P(i) * P(j)) )

    PMI  > 0   i and j border each other MORE than chance  -> signal
    PMI == 0   indistinguishable from random               -> no signal
    PMI  < 0   they avoid each other                       -> also signal

If |PMI| is near zero across the board, stop and rethink before writing
pipeline code. Better to learn that in week 2 than week 9.

Adjacency is measured as SHARED BOUNDARY LENGTH: every horizontally or
vertically neighbouring pixel pair with two different class labels counts once.
That is more meaningful for land cover than centroid distance, and it is the
same definition your Step 3 should use.

Usage
-----
    python scripts/cooccurrence_gt.py --root ~/data/loveda/Val
    python scripts/cooccurrence_gt.py --root ~/data/loveda/Val --split-domains
"""

import argparse
import glob
import os
import sys

import numpy as np

# LoveDA: 0 is no-data/ignore, 1..7 are the real classes.
LOVEDA_CLASSES = [
    "ignore", "background", "building", "road",
    "water", "barren", "forest", "agriculture",
]


def find_masks(root):
    pats = [
        os.path.join(root, "**", "masks_png", "*.png"),
        os.path.join(root, "**", "masks", "*.png"),
    ]
    out = []
    for p in pats:
        out.extend(glob.glob(p, recursive=True))
    return sorted(out)


def accumulate(mask_paths, n_classes, verbose=True):
    """Return (adjacency counts n x n, pixel counts n)."""
    from PIL import Image

    M = np.zeros((n_classes, n_classes), dtype=np.int64)
    pix = np.zeros(n_classes, dtype=np.int64)

    for i, p in enumerate(mask_paths):
        lbl = np.array(Image.open(p))
        if lbl.ndim == 3:
            lbl = lbl[..., 0]
        lbl = lbl.astype(np.int64)
        if lbl.max() >= n_classes:
            raise ValueError(
                f"{p} has label {lbl.max()} but n_classes={n_classes}. "
                "Check the class mapping."
            )

        pix += np.bincount(lbl.ravel(), minlength=n_classes)

        # horizontal and vertical neighbour pairs
        a = np.concatenate([lbl[:, :-1].ravel(), lbl[:-1, :].ravel()])
        b = np.concatenate([lbl[:, 1:].ravel(), lbl[1:, :].ravel()])

        # keep only cross-class pairs, drop the ignore label
        keep = (a > 0) & (b > 0) & (a != b)
        if keep.any():
            flat = a[keep] * n_classes + b[keep]
            c = np.bincount(flat, minlength=n_classes * n_classes)
            c = c.reshape(n_classes, n_classes)
            M += c + c.T          # symmetric: boundaries have no direction

        if verbose and (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(mask_paths)} masks", flush=True)

    return M, pix


def analyse(M, pix, names, drop=()):
    """Row-normalised adjacency, and PMI against an independence baseline.

    drop: class names to exclude entirely. Use this to test whether the signal
    survives without catch-all classes like LoveDA's "background" - if mean
    |PMI| collapses once background is removed, the prior is mostly recording
    "everything touches the leftover class", which cannot help you label an
    ambiguous patch.
    """
    n = len(names)
    drop = {d.lower() for d in drop}
    valid = [i for i in range(1, n)                  # skip ignore
             if pix[i] > 0                           # skip absent
             and names[i].lower() not in drop]       # skip explicitly dropped
    if len(valid) < 2:
        raise SystemExit(f"Only {len(valid)} class(es) left after --drop.")

    Mv = M[np.ix_(valid, valid)].astype(float)
    pv = pix[valid].astype(float)
    nv = [names[i] for i in valid]

    total = Mv.sum()
    if total == 0:
        raise SystemExit("No class boundaries found. Check the mask format.")

    P_obs = Mv / total                       # observed joint boundary distribution
    p = pv / pv.sum()                        # marginal class area frequency
    P_exp = np.outer(p, p)
    np.fill_diagonal(P_exp, 0.0)
    if P_exp.sum() > 0:
        P_exp = P_exp / P_exp.sum()          # renormalise, diagonal excluded

    with np.errstate(divide="ignore", invalid="ignore"):
        PMI = np.log2(np.where(P_exp > 0, P_obs / P_exp, np.nan))
    PMI[~np.isfinite(PMI)] = 0.0

    rows = Mv.sum(axis=1, keepdims=True)
    rows[rows == 0] = 1
    M_row = Mv / rows                        # P(neighbour = j | patch = i)

    return dict(names=nv, M_raw=Mv, M_row=M_row, PMI=PMI,
                P_obs=P_obs, freq=p)


def report(res, title):
    names, PMI, M_row, p = res["names"], res["PMI"], res["M_row"], res["freq"]
    w = max(len(x) for x in names) + 1

    print("\n" + "=" * 64)
    print(f"{title}")
    print("=" * 64)

    print("\nClass area share:")
    for nm, f in sorted(zip(names, p), key=lambda t: -t[1]):
        print(f"  {nm:<{w}} {f * 100:5.1f}%")

    print("\nP(neighbour = column | patch = row)   [row-normalised adjacency]")
    print(" " * w + "".join(f"{nm[:6]:>8}" for nm in names))
    for i, nm in enumerate(names):
        print(f"{nm:<{w}}" + "".join(f"{M_row[i, j]:8.3f}" for j in range(len(names))))

    off = ~np.eye(len(names), dtype=bool)
    strength = float(np.abs(PMI[off]).mean())

    print("\nPMI vs independence  (>0 = touch more than chance)")
    print(" " * w + "".join(f"{nm[:6]:>8}" for nm in names))
    for i, nm in enumerate(names):
        print(f"{nm:<{w}}" + "".join(
            ("   .    " if i == j else f"{PMI[i, j]:8.2f}") for j in range(len(names))))

    pairs = [(PMI[i, j], names[i], names[j])
             for i in range(len(names)) for j in range(i + 1, len(names))]
    pairs.sort()
    print("\nStrongest attractions:")
    for v, a, b in pairs[::-1][:5]:
        print(f"  {a} - {b}: PMI {v:+.2f}  ({2 ** v:.1f}x chance)")
    print("Strongest avoidances:")
    for v, a, b in pairs[:5]:
        print(f"  {a} - {b}: PMI {v:+.2f}  ({2 ** v:.2f}x chance)")

    print(f"\nMean |PMI| off-diagonal: {strength:.3f} bits")
    if strength < 0.15:
        print(">>> WEAK. Adjacency is close to random. The co-occurrence prior")
        print("    may not carry enough signal - reconsider before building on it.")
    elif strength < 0.5:
        print(">>> MODERATE. Real but not dominant. Expect the prior to help at")
        print("    the margins; keep the neighbour-voting term as well.")
    else:
        print(">>> STRONG. Land cover has clear adjacency structure here.")
        print("    Your co-occurrence prior has real information to exploit.")
    return strength


def plot(res, out_png, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed - skipping heatmap; pip install matplotlib)")
        return

    names, M_row, PMI = res["names"], res["M_row"], res["PMI"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax, data, ttl, cmap, kw in [
        (axes[0], M_row, "P(neighbour | class)", "viridis", {}),
        (axes[1], PMI, "PMI vs independence", "RdBu_r",
         {"vmin": -float(np.abs(PMI).max()), "vmax": float(np.abs(PMI).max())}),
    ]:
        im = ax.imshow(data, cmap=cmap, **kw)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticklabels(names)
        ax.set_title(ttl)
        for i in range(len(names)):
            for j in range(len(names)):
                if i != j:
                    ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                            fontsize=7,
                            color="white" if cmap == "viridis" and data[i, j] > data.max() / 2
                            else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"\nheatmap written to {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="e.g. ~/data/loveda/Val")
    ap.add_argument("--limit", type=int, default=0, help="cap number of masks")
    ap.add_argument("--split-domains", action="store_true",
                    help="analyse Urban and Rural separately (transfer experiment)")
    ap.add_argument("--drop", nargs="+", default=[], metavar="CLASS",
                    help="exclude these classes, e.g. --drop background")
    ap.add_argument("--outdir", default="outputs/cooccurrence")
    args = ap.parse_args()

    if args.drop:
        known = {c.lower() for c in LOVEDA_CLASSES}
        bad = [d for d in args.drop if d.lower() not in known]
        if bad:
            sys.exit(f"Unknown class(es) {bad}. Known: {LOVEDA_CLASSES[1:]}")
        print(f"Dropping class(es): {', '.join(args.drop)}")

    root = os.path.expanduser(args.root)
    os.makedirs(args.outdir, exist_ok=True)
    n = len(LOVEDA_CLASSES)

    groups = {"all": find_masks(root)}
    if args.split_domains:
        groups = {}
        for dom in ("Urban", "Rural"):
            m = find_masks(os.path.join(root, dom))
            if m:
                groups[dom] = m
        if not groups:
            groups = {"all": find_masks(root)}

    if not any(groups.values()):
        sys.exit(f"No masks found under {root}. Expected a masks_png/ directory.")

    results = {}
    for name, paths in groups.items():
        if args.limit:
            paths = paths[: args.limit]
        print(f"\n[{name}] {len(paths)} masks")
        M, pix = accumulate(paths, n)
        res = analyse(M, pix, LOVEDA_CLASSES, drop=args.drop)
        suffix = ("_no-" + "-".join(d.lower() for d in args.drop)) if args.drop else ""
        title = f"{name}  ({len(paths)} masks)"
        if args.drop:
            title += f"  [without {', '.join(args.drop)}]"
        report(res, title)
        plot(res, os.path.join(args.outdir, f"cooccurrence_{name}{suffix}.png"),
             f"LoveDA {name} - land-cover adjacency{suffix.replace('_', ' ')}")
        np.save(os.path.join(args.outdir, f"M_{name}{suffix}.npy"), res["M_raw"])
        results[name] = res

    if len(results) == 2:
        (a, ra), (b, rb) = results.items()
        d = np.abs(ra["PMI"] - rb["PMI"])
        print("\n" + "=" * 64)
        print(f"DOMAIN TRANSFER: {a} vs {b}")
        print("=" * 64)
        print(f"mean |PMI difference| : {d.mean():.3f} bits")
        print(f"max  |PMI difference| : {d.max():.3f} bits")
        if d.mean() < 0.3:
            print(">>> Priors are SIMILAR - M likely transfers across domains.")
        else:
            print(">>> Priors DIFFER - M is domain-specific. That is a finding,")
            print("    and an argument for the per-image term in your blend.")

    print(f"\nArtefacts in {args.outdir}/")


if __name__ == "__main__":
    main()
