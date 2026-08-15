"""Validate the dimensionality measurement before any taxonomy is read off it.

shape_space.py reports that a SERI95 syllable class needs ~28 principal
components to cover 80% of its within-class variance. That number is only
meaningful if the measurement can recognise a *low*-dimensional family as low
dimensional. This script checks exactly that.

It takes one real sample and generates a synthetic family from it using a known,
small number of degrees of freedom — rotation, shear, anisotropic scale, stroke
thickness — then runs the identical ink-map PCA over that family. A family built
from k parameters should need on the order of k components.

If a 4-parameter family also measures as ~28 components, then the component
count is reporting a property of the pixel basis and not of handwriting, and no
renderer parameter may be justified by it. HANDOFF.md §12 asks whether a piece
of work helps decide the taxonomy; this is the check that decides whether the
preceding measurement is admissible evidence at all.
"""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image

from analyze_images import mask, IMAGE_SUFFIXES
from shape_space import pcs_to_reach


def base_crop(path):
    m = mask(path)
    ys, xs = np.nonzero(m)
    if len(xs) < 3:
        return None
    return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def deform(crop, rot_deg, shear, sx, sy, thicken, grid):
    """Apply a similarity+shear deformation and a thickness change, then bbox-normalise.

    bbox-normalisation is applied exactly as in shape_space.ink_map, so the
    control passes through the same nuisance removal as the real data.
    """
    im = Image.fromarray((crop * 255).astype(np.uint8))
    # pad so rotation/shear does not clip
    w, h = im.size
    pad = int(max(w, h) * 0.6)
    canvas = Image.new("L", (w + 2 * pad, h + 2 * pad), 0)
    canvas.paste(im, (pad, pad))

    if thicken:
        a = np.asarray(canvas, np.float32) / 255.0
        k = np.ones((3, 3), np.float32)
        for _ in range(abs(thicken)):
            p = np.pad(a, 1)
            nb = sum(p[i:i + a.shape[0], j:j + a.shape[1]] for i in range(3) for j in range(3)) / k.sum()
            a = np.maximum(a, nb * 1.5) if thicken > 0 else np.minimum(a, nb * 0.6)
        canvas = Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8))

    cw, ch = canvas.size
    canvas = canvas.resize((max(1, int(cw * sx)), max(1, int(ch * sy))), Image.BILINEAR)
    cw, ch = canvas.size
    # shear about the centre
    canvas = canvas.transform((cw, ch), Image.AFFINE, (1, shear, -shear * ch / 2, 0, 1, 0),
                              resample=Image.BILINEAR)
    canvas = canvas.rotate(rot_deg, resample=Image.BILINEAR, expand=True)

    a = np.asarray(canvas, np.float32) / 255.0
    ink = a > 0.4
    ys, xs = np.nonzero(ink)
    if len(xs) < 3:
        return None
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    out = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((grid, grid), Image.BILINEAR),
                     np.float32) / 255.0
    s = out.sum()
    return (out / s).ravel() if s > 0 else None


def build_family(crop, n, dof, grid, rng):
    rows = []
    for _ in range(n):
        rot = rng.uniform(-12, 12) if dof >= 1 else 0.0
        shear = rng.uniform(-0.25, 0.25) if dof >= 2 else 0.0
        sx = rng.uniform(0.8, 1.25) if dof >= 3 else 1.0
        sy = rng.uniform(0.8, 1.25) if dof >= 4 else 1.0
        th = int(rng.integers(-1, 2)) if dof >= 5 else 0
        v = deform(crop, rot, shear, sx, sy, th, grid)
        if v is not None:
            rows.append(v)
    return np.asarray(rows, np.float64)


def measure(X):
    Z = X - X.mean(0)
    _, S, _ = np.linalg.svd(Z, full_matrices=False)
    var = S ** 2
    ratio = var / var.sum()
    return {
        "n": int(len(X)),
        "pcs_for_80pct": pcs_to_reach(ratio, 0.80),
        "pcs_for_90pct": pcs_to_reach(ratio, 0.90),
        "pc1_explained": float(ratio[0]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("class_dir", help="one extracted class directory, e.g. data/extracted/seri95/UAC00_가")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--n", type=int, default=100, help="family size, matched to the real class sample count")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    paths = [p for p in sorted(Path(a.class_dir).rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES]
    if not paths:
        raise SystemExit(f"no images in {a.class_dir}")
    crop = base_crop(paths[0])
    if crop is None:
        raise SystemExit("first sample has no ink")

    rng = np.random.default_rng(a.seed)
    controls = {}
    for dof, name in [(1, "1dof_rotation"), (2, "2dof_rotation_shear"),
                      (4, "4dof_rotation_shear_scale"), (5, "5dof_plus_thickness")]:
        X = build_family(crop, a.n, dof, a.grid, rng)
        controls[name] = {"declared_dof": dof, **measure(X)}
        print(f"{name:28s} declared_dof={dof}  ->  {controls[name]['pcs_for_80pct']} PCs for 80%, "
              f"{controls[name]['pcs_for_90pct']} for 90%, PC1={controls[name]['pc1_explained']*100:.1f}%")

    report = {
        "base_sample": str(paths[0]),
        "grid": a.grid,
        "family_size": a.n,
        "controls": controls,
        "reading": "Compare pcs_for_80pct against the same statistic on the real class. If a "
                   "declared-4-dof family needs a comparable number of components, the ink-map "
                   "basis cannot be used to size the renderer parameter budget.",
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n->", out)


if __name__ == "__main__":
    main()
