"""Extract PE92 / SERI95 .hgu1 archives into per-class PNG directories.

HGU1 layout, verified against the dataset's own DisplayHGU1.cpp:

    <file header>  "HGU1    "                                    8 bytes
    <image>        code(2) width(1) height(1) type(1) reserved(1) + width*height bytes

The 2-byte code is KS X 1001 wansung (EUC-KR). This is measured, not assumed:
across all 2350 PE92 and 520 SERI95 class files every code decodes to a Hangul
syllable, and each PE92 filename stem is the code's own hex. See
reports/dataset_verification.json for the check.

Class directories are named `U{codepoint:04X}_{syllable}` so the label carried
into the feature CSV is the real syllable, which Phase 2 needs in order to
decompose into 초성/중성/종성.
"""
import argparse, json, zipfile
from pathlib import Path
import numpy as np
from PIL import Image

FILE_HEADER = b"HGU1"
IMAGE_HEADER_LEN = 6


def iter_hgu1(blob, source=""):
    """Yield (index, code_bytes, ndarray) for every image in one .hgu1 blob."""
    if not blob.startswith(FILE_HEADER):
        raise ValueError(f"{source}: not an HGU1 file (header {blob[:8]!r})")
    off, idx = 8, 0
    while off + IMAGE_HEADER_LEN <= len(blob):
        code = blob[off:off + 2]
        width, height = blob[off + 2], blob[off + 3]
        img_type = blob[off + 4]
        off += IMAGE_HEADER_LEN
        size = width * height
        if img_type != 0:
            raise ValueError(f"{source}: image {idx} has type {img_type}, only 0 (uint8) is supported")
        if off + size > len(blob):
            break  # truncated trailing record
        arr = np.frombuffer(blob, dtype=np.uint8, count=size, offset=off).reshape(height, width)
        off += size
        yield idx, code, arr
        idx += 1


def label_for(code):
    """Decode the 2-byte class code to `U{codepoint:04X}_{syllable}`."""
    try:
        ch = code.decode("euc-kr")
    except UnicodeDecodeError:
        return f"RAW_{code.hex()}", None
    if len(ch) != 1 or not (0xAC00 <= ord(ch) <= 0xD7A3):
        return f"RAW_{code.hex()}", None
    return f"U{ord(ch):04X}_{ch}", ch


def iter_sources(path):
    """Yield (name, blob) for .hgu1 members of a zip, or files under a directory."""
    path = Path(path)
    if path.is_dir():
        for p in sorted(path.rglob("*.hgu1")):
            yield p.name, p.read_bytes()
    elif path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            for name in sorted(n for n in z.namelist() if n.lower().endswith(".hgu1")):
                yield name.rsplit("/", 1)[-1], z.read(name)
    else:
        yield path.name, path.read_bytes()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help=".zip archive, directory of .hgu1 files, or a single .hgu1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-class", type=int, default=0, help="cap samples kept per class (0 = all)")
    ap.add_argument("--drop-tail", type=int, default=0,
                    help="drop the last N samples of every class file. The HangulDB README reports that "
                         "PE92's mislabeled samples cluster at the end of files.")
    ap.add_argument("--min-side", type=int, default=8,
                    help="skip images whose width or height is below this (PE92 carries a few 1x1 records)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N images total (0 = no limit)")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    saved = skipped_small = skipped_tail = 0
    per_class = {}
    undecoded = set()

    for name, blob in iter_sources(a.source):
        try:
            images = list(iter_hgu1(blob, name))
        except ValueError as e:
            print("SKIP", e)
            continue
        if a.drop_tail:
            keep = len(images) - a.drop_tail
            skipped_tail += len(images) - max(0, keep)
            images = images[:max(0, keep)]
        for idx, code, arr in images:
            label, ch = label_for(code)
            if ch is None:
                undecoded.add(code.hex())
            if arr.shape[0] < a.min_side or arr.shape[1] < a.min_side:
                skipped_small += 1
                continue
            if a.per_class and per_class.get(label, 0) >= a.per_class:
                continue
            d = out / label
            d.mkdir(exist_ok=True)
            Image.fromarray(arr, "L").save(d / f"{Path(name).stem}_{idx:05d}.png")
            per_class[label] = per_class.get(label, 0) + 1
            saved += 1
            if a.limit and saved >= a.limit:
                print(f"limit reached at {saved}")
                break
        if a.limit and saved >= a.limit:
            break

    counts = sorted(per_class.values())
    summary = {
        "source": str(a.source),
        "out": str(out),
        "images_saved": saved,
        "classes": len(per_class),
        "per_class_min": counts[0] if counts else 0,
        "per_class_median": counts[len(counts) // 2] if counts else 0,
        "per_class_max": counts[-1] if counts else 0,
        "skipped_below_min_side": skipped_small,
        "dropped_tail_samples": skipped_tail,
        "undecodable_codes": sorted(undecoded),
    }
    (out / "_extract_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
