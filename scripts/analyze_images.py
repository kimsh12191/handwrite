"""Per-image shape features for the Phase 2 variation analysis.

Feature set follows HANDOFF.md Phase 2: aspect, centroid, principal angle,
moment ratio, left/right and top/bottom asymmetry, compactness, connected
components, hole count.

Connected components and holes were originally a per-pixel Python flood fill,
which is ~10^9 interpreter steps over the 75k-image Hangul corpus. They are now
labelled with scipy.ndimage under the same 4-connectivity, so the values are
unchanged; `scripts/verify_features.py` checks the two implementations agree.
"""
import argparse, csv, math
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
# 4-connectivity, matching the original flood fill's neighbour set.
CONN4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)


def mask(path):
    a = np.asarray(Image.open(path).convert("L"), np.float32)
    t = (np.percentile(a, 10) + np.percentile(a, 90)) / 2
    m = a < t
    if m.mean() > .5:
        m = ~m
    return m


def count_components(m):
    """Number of 4-connected ink components."""
    return int(ndimage.label(m, structure=CONN4)[1])


def count_holes(ink):
    """Background components that do not touch the image border."""
    lab, n = ndimage.label(~ink, structure=CONN4)
    if n == 0:
        return 0
    border = np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])
    return int(n - len(set(border.tolist()) - {0}))


def feat(path):
    m = mask(path)
    ys, xs = np.nonzero(m)
    if len(xs) < 3:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    crop = m[y0:y1 + 1, x0:x1 + 1]
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    xn = (xs - x0) / max(1, bw - 1)
    yn = (ys - y0) / max(1, bh - 1)
    cx, cy = xn.mean(), yn.mean()
    X = np.stack([xn - cx, yn - cy], 1)
    cov = X.T @ X / max(1, len(X))
    vals, vecs = np.linalg.eigh(cov)
    o = np.argsort(vals)[::-1]
    vals = vals[o]
    v = vecs[:, o[0]]
    # 4-neighbour boundary edge count
    p = np.pad(crop.astype(np.uint8), 1)
    c = p[1:-1, 1:-1]
    per = int(((c != p[:-2, 1:-1]) | (c != p[2:, 1:-1]) | (c != p[1:-1, :-2]) | (c != p[1:-1, 2:])).sum())
    area = int(crop.sum())
    return {
        "path": str(path), "label": path.parent.name, "width": bw, "height": bh,
        "aspect": bw / max(1, bh),
        "area_ratio": area / (bw * bh),
        "centroid_x": float(cx), "centroid_y": float(cy),
        "principal_angle": float(math.atan2(v[1], v[0])),
        "moment_eigen_ratio": float(vals[0] / max(vals[1], 1e-9)),
        "lr_asym": float(abs((xn < .5).mean() - (xn >= .5).mean())),
        "tb_asym": float(abs((yn < .5).mean() - (yn >= .5).mean())),
        "compactness": float(per * per / (4 * math.pi * area)) if area else float("nan"),
        "components": count_components(crop),
        "holes": count_holes(crop),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    rows = []
    for p in sorted(Path(a.root).rglob("*")):
        if p.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            r = feat(p)
            if r:
                rows.append(r)
        except Exception as e:
            print("SKIP", p, e)
        if a.limit and len(rows) >= a.limit:
            break
    if not rows:
        raise SystemExit("no images")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "rows ->", out)


if __name__ == "__main__":
    main()
