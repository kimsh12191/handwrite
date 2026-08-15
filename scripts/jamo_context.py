"""Does adding a 종성 just squash the 초성+중성, or re-shape it?

HANDOFF.md §5: the same jamo in a different role may not share a shape
distribution, and the renderer must not assume it does. That claim decides a
concrete design question — whether the syllable layer needs one packing
parameter or a separate shape per role — so it is measured here rather than
assumed either way.

Design. Group classes by (초성, 중성). Within a group, the open syllable (no
종성) and the closed syllables differ by exactly one factor: the 종성. Take the
class-mean ink map of each. If a 종성 only compresses what is above it, then
squashing the open syllable's map into the top f of the square should reproduce
the closed syllable's top-f region. Sweep f, keep the best cosine similarity.

Two baselines make the number readable:

  matched    open 가 vs closed 각 — same 초성 and 중성
  mismatched open 가 vs closed 밥 — different 초성, i.e. chance level
  ceiling    the closed class split in half, its two half-means compared over
             the same top-f region at the same f — the same syllable measured
             twice, so it carries only the corpus's own within-class noise and
             passes through the identical crop and resample path as the matched
             score. This is what "as good as the data allows" means here.

A matched score near the self ceiling means one compression parameter suffices.
A matched score down near mismatched means the jamo genuinely re-shapes.
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

from analyze_images import mask, IMAGE_SUFFIXES
from hangul import decompose, syllable_from_label, vowel_class

F_GRID = np.arange(0.45, 0.91, 0.025)


def ink_map(path, grid):
    m = mask(path)
    ys, xs = np.nonzero(m)
    if len(xs) < 3:
        return None
    crop = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    a = np.asarray(Image.fromarray((crop * 255).astype(np.uint8)).resize((grid, grid), Image.BILINEAR),
                   np.float32) / 255.0
    s = a.sum()
    return a / s if s > 0 else None


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float((a * b).sum() / (na * nb))


def squash_to_top(m, f, grid):
    """Resample the whole map into the top f fraction of a grid x grid square."""
    rows = max(2, int(round(f * grid)))
    im = Image.fromarray((m / (m.max() + 1e-12) * 255).astype(np.uint8)).resize((grid, rows), Image.BILINEAR)
    return np.asarray(im, np.float32) / 255.0


def best_fit(open_map, closed_map, grid):
    """Best cosine over vertical compression factors, comparing top-f regions.

    Also returns how peaked that optimum is: the spread of f values scoring
    within 0.005 cosine of the best. A wide plateau means the recovered
    compression factor is weakly identified and must not be read as a precise
    measurement of how far the 종성 pushes the syllable up.
    """
    scores = []
    for f in F_GRID:
        rows = max(2, int(round(f * grid)))
        pred = squash_to_top(open_map, f, grid)
        s = cosine(pred.ravel(), closed_map[:rows, :].ravel())
        scores.append((s, float(f)))
    best_s, best_f = max(scores)
    near = [f for s, f in scores if s >= best_s - 0.005]
    return best_s, best_f, float(max(near) - min(near))


def region_ceiling(half_maps, f, grid):
    """Noise floor: two half-means of the same class over the same top-f region."""
    rows = max(2, int(round(f * grid)))
    h1, h2 = half_maps
    return cosine(h1[:rows, :].ravel(), h2[:rows, :].ravel())


def class_mean_maps(root, grid, max_per_class, min_n):
    by_class = defaultdict(list)
    for p in sorted(Path(root).rglob("*")):
        if p.suffix.lower() in IMAGE_SUFFIXES:
            by_class[p.parent.name].append(p)
    means, halves = {}, {}
    for label, paths in by_class.items():
        syl = syllable_from_label(label)
        if syl is None:
            continue
        maps = [m for m in (ink_map(p, grid) for p in paths[:max_per_class]) if m is not None]
        if len(maps) < min_n:
            continue
        A = np.asarray(maps)
        means[syl] = A.mean(0)
        h = len(A) // 2
        halves[syl] = (A[:h].mean(0), A[h:].mean(0))
    return means, halves


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--max-per-class", type=int, default=100)
    ap.add_argument("--min-n", type=int, default=30)
    a = ap.parse_args()

    means, halves = class_mean_maps(a.root, a.grid, a.max_per_class, a.min_n)
    print(f"class means built for {len(means)} syllables")

    groups = defaultdict(lambda: {"open": None, "closed": []})
    for syl in means:
        d = decompose(syl)
        if d is None:
            continue
        cho, jung, jong = d
        key = (cho, jung)
        if jong:
            groups[key]["closed"].append(syl)
        else:
            groups[key]["open"] = syl

    rng = np.random.default_rng(0)
    usable = {k: v for k, v in groups.items() if v["open"] and v["closed"]}
    print(f"{len(usable)} (초성,중성) groups have both an open and a closed class")

    pairs = []
    open_syls = [v["open"] for v in usable.values()]
    for (cho, jung), v in sorted(usable.items()):
        om = means[v["open"]]
        for cs in v["closed"]:
            s, f, plateau = best_fit(om, means[cs], a.grid)
            # mismatched control: same closed syllable, an open syllable from a different 초성
            others = [o for o in open_syls if decompose(o)[0] != cho]
            mm = -1.0
            if others:
                pick = others[int(rng.integers(len(others)))]
                mm = best_fit(means[pick], means[cs], a.grid)[0]
            pairs.append({
                "open": v["open"], "closed": cs, "cho": cho, "jung": jung,
                "jong": decompose(cs)[2], "vowel_class": vowel_class(jung),
                "matched_cosine": s, "best_compression": f, "compression_plateau": plateau,
                "mismatched_cosine": mm,
                # noise floor for this exact comparison region
                "ceiling_cosine": region_ceiling(halves[cs], f, a.grid),
            })

    def stats(xs):
        xs = np.asarray([x for x in xs if x is not None and x >= 0], float)
        if not len(xs):
            return None
        return {"n": int(len(xs)), "mean": float(xs.mean()), "median": float(np.median(xs)),
                "p10": float(np.percentile(xs, 10)), "p90": float(np.percentile(xs, 90))}

    by_vowel = defaultdict(list)
    for p in pairs:
        by_vowel[p["vowel_class"]].append(p["matched_cosine"])
    by_jong = defaultdict(list)
    for p in pairs:
        by_jong[p["jong"]].append(p["matched_cosine"])

    summary = {
        "root": str(a.root),
        "grid": a.grid,
        "syllable_classes": len(means),
        "groups_with_open_and_closed": len(usable),
        "pairs": len(pairs),
        "ceiling_same_class_halves": stats([p["ceiling_cosine"] for p in pairs]),
        "matched": stats([p["matched_cosine"] for p in pairs]),
        "mismatched_control": stats([p["mismatched_cosine"] for p in pairs]),
        "best_compression": stats([p["best_compression"] for p in pairs]),
        "compression_plateau_width": stats([p["compression_plateau"] for p in pairs]),
        "gap_recovered": None,  # filled in below
        "matched_by_vowel_class": {k: stats(v) for k, v in sorted(by_vowel.items())},
        "compression_by_vowel_class": {
            k: stats([p["best_compression"] for p in pairs if p["vowel_class"] == k])
            for k in sorted(by_vowel)
        },
        "matched_by_jong_top10": {
            k: stats(v) for k, v in sorted(by_jong.items(), key=lambda kv: -len(kv[1]))[:10]
        },
    }
    # How much of the distance from chance to the noise floor does a single
    # compression parameter actually close? 1.0 would mean the 종성 effect is
    # entirely a vertical squash.
    ceil, match, mis = (summary["ceiling_same_class_halves"]["mean"],
                        summary["matched"]["mean"], summary["mismatched_control"]["mean"])
    summary["gap_recovered"] = float((match - mis) / (ceil - mis)) if ceil > mis else None
    summary["residual_to_ceiling"] = float(ceil - match)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "pairs": pairs}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
