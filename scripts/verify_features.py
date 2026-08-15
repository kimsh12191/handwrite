"""Check the vectorised component/hole counters against the original flood fill.

analyze_images.py used to count connected components and holes with a
per-pixel Python flood fill. That is correct but far too slow for the ~75k
image Hangul corpus, so it now uses scipy.ndimage.label. This script keeps the
original implementation verbatim and asserts both agree on real samples, so the
speedup is not silently a behaviour change.
"""
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_images import mask, count_components, count_holes, IMAGE_SUFFIXES  # noqa: E402


def ref_count_components(m, target=True):
    """Original implementation, unchanged."""
    h, w = m.shape
    seen = np.zeros_like(m, bool)
    n = 0
    for y in range(h):
        for x in range(w):
            if bool(m[y, x]) != target or seen[y, x]:
                continue
            n += 1
            st = [(y, x)]
            seen[y, x] = 1
            while st:
                yy, xx = st.pop()
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = yy + dy, xx + dx
                    if 0 <= ny < h and 0 <= nx < w and bool(m[ny, nx]) == target and not seen[ny, nx]:
                        seen[ny, nx] = 1
                        st.append((ny, nx))
    return n


def ref_holes(ink):
    """Original implementation, unchanged."""
    bg = ~ink
    h, w = bg.shape
    seen = np.zeros_like(bg, bool)
    n = 0
    for y in range(h):
        for x in range(w):
            if not bg[y, x] or seen[y, x]:
                continue
            st = [(y, x)]
            seen[y, x] = 1
            touch = False
            while st:
                yy, xx = st.pop()
                if yy in (0, h - 1) or xx in (0, w - 1):
                    touch = True
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = yy + dy, xx + dx
                    if 0 <= ny < h and 0 <= nx < w and bg[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = 1
                        st.append((ny, nx))
            if not touch:
                n += 1
    return n


def crop_of(path):
    m = mask(path)
    ys, xs = np.nonzero(m)
    if len(xs) < 3:
        return None
    return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--n", type=int, default=300, help="how many images to check")
    a = ap.parse_args()

    paths = [p for p in sorted(Path(a.root).rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES]
    if not paths:
        raise SystemExit(f"no images under {a.root}")
    # Spread the sample across classes rather than taking one directory.
    step = max(1, len(paths) // a.n)
    sample = paths[::step][:a.n]

    checked = mismatch = 0
    for p in sample:
        crop = crop_of(p)
        if crop is None:
            continue
        got = (count_components(crop), count_holes(crop))
        want = (ref_count_components(crop, True), ref_holes(crop))
        checked += 1
        if got != want:
            mismatch += 1
            print(f"MISMATCH {p}: fast={got} reference={want}")
    print(f"checked {checked} images from {len(paths)} available, {mismatch} mismatches")
    if mismatch:
        raise SystemExit(1)
    print("OK: vectorised counters match the original flood fill")


if __name__ == "__main__":
    main()
