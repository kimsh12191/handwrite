"""How many degrees of freedom does one syllable class actually use?

The bootstrap's PCA runs on eleven global summary statistics (aspect,
compactness, moments...). Those are whole-syllable averages, so any variation
that is local to one jamo is smeared across all of them. This script repeats the
measurement in a spatially resolved basis — the bbox-normalised ink map — and
reports, per class, how many components are needed to reach 80% and 90% of the
within-class variance.

The number is what decides the size of the renderer's per-syllable parameter
budget in Phase 3, so it is reported per class rather than as a single average.

Caveat, stated because it changes how the output should be read: a large
component count in a pixel basis is evidence about the basis as much as about
handwriting. Unaligned ink maps make even a pure translation look like a
high-rank deformation. Read these counts as an upper bound on the true shape
dimensionality, and compare them against the summary-statistic run rather than
treating either as the truth.
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

from analyze_images import mask, IMAGE_SUFFIXES


def ink_map(path, grid):
    """Bbox-cropped ink density resampled onto a grid x grid square, unit L1 norm."""
    m = mask(path)
    ys, xs = np.nonzero(m)
    if len(xs) < 3:
        return None
    crop = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    im = Image.fromarray((crop * 255).astype(np.uint8)).resize((grid, grid), Image.BILINEAR)
    a = np.asarray(im, np.float32) / 255.0
    s = a.sum()
    return (a / s).ravel() if s > 0 else None


def pcs_to_reach(ratio, threshold):
    c = np.cumsum(ratio)
    idx = np.searchsorted(c, threshold) + 1
    return int(min(idx, len(ratio)))


def save_modes(mean, comps, grid, out_dir, label, k=4):
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles = [mean.reshape(grid, grid)]
    for i in range(min(k, len(comps))):
        v = comps[i].reshape(grid, grid)
        v = v / (np.abs(v).max() + 1e-12)
        tiles.append(v)
    # mean rendered as ink, modes rendered as signed red/blue deviation
    h = grid
    canvas = np.ones((h, h * len(tiles), 3), np.float32)
    m0 = tiles[0] / (tiles[0].max() + 1e-12)
    canvas[:, :h, :] = (1 - m0)[..., None]
    for i, v in enumerate(tiles[1:], start=1):
        pos, neg = np.clip(v, 0, 1), np.clip(-v, 0, 1)
        canvas[:, i * h:(i + 1) * h, 0] = 1 - neg
        canvas[:, i * h:(i + 1) * h, 1] = 1 - pos - neg
        canvas[:, i * h:(i + 1) * h, 2] = 1 - pos
    im = Image.fromarray((canvas * 255).astype(np.uint8)).resize(
        (h * len(tiles) * 4, h * 4), Image.NEAREST)
    im.save(out_dir / f"{label}_modes.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="extracted per-class image directories")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument("--max-per-class", type=int, default=100)
    ap.add_argument("--classes", type=int, default=0, help="limit number of classes (0 = all)")
    ap.add_argument("--modes-dir", default="", help="also dump PC1..PC4 mode images here")
    a = ap.parse_args()

    by_class = defaultdict(list)
    for p in sorted(Path(a.root).rglob("*")):
        if p.suffix.lower() in IMAGE_SUFFIXES:
            by_class[p.parent.name].append(p)

    labels = sorted(by_class)
    if a.classes:
        labels = labels[:a.classes]

    report = {}
    for label in labels:
        paths = by_class[label][:a.max_per_class]
        if len(paths) < a.min_n:
            continue
        X = [v for v in (ink_map(p, a.grid) for p in paths) if v is not None]
        if len(X) < a.min_n:
            continue
        X = np.asarray(X, np.float64)
        mean = X.mean(0)
        Z = X - mean
        _, S, Vt = np.linalg.svd(Z, full_matrices=False)
        var = S ** 2
        ratio = var / var.sum()
        report[label] = {
            "n": len(X),
            "grid": a.grid,
            "pcs_for_80pct": pcs_to_reach(ratio, 0.80),
            "pcs_for_90pct": pcs_to_reach(ratio, 0.90),
            "top5_explained": [float(x) for x in ratio[:5]],
        }
        if a.modes_dir:
            save_modes(mean, Vt, a.grid, Path(a.modes_dir), label)

    if not report:
        raise SystemExit("no class met --min-n")

    p80 = sorted(r["pcs_for_80pct"] for r in report.values())
    p90 = sorted(r["pcs_for_90pct"] for r in report.values())
    pc1 = [r["top5_explained"][0] for r in report.values()]
    summary = {
        "root": str(a.root),
        "grid": a.grid,
        "classes": len(report),
        "pcs_for_80pct": {"min": p80[0], "median": p80[len(p80) // 2], "max": p80[-1]},
        "pcs_for_90pct": {"min": p90[0], "median": p90[len(p90) // 2], "max": p90[-1]},
        "pc1_explained_mean": float(np.mean(pc1)),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_class": report}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
