"""Check the HGU1 format and label-encoding assumptions the pipeline relies on.

extract_hgu1.py decodes the 2-byte class code as EUC-KR. The bootstrap
deliberately refused to ("Do not guess legacy Korean encoding") and kept raw
hex instead, which is safe but leaves Phase 2 unable to decompose a sample into
초성/중성/종성 — the thing HANDOFF.md §5 requires. So the encoding is checked
here instead of guessed, over every class file in the archive:

  - the file header is "HGU1"
  - records tile the file exactly, with no trailing bytes
  - every class file holds exactly one code
  - every code decodes to a Hangul syllable under EUC-KR
  - for PE92, the code equals the filename stem, an independent confirmation
    that the byte pair really is the class identifier

Writes reports/dataset_verification.json.
"""
import argparse, json, zipfile
from collections import Counter
from pathlib import Path


def check_archive(path):
    z = zipfile.ZipFile(path)
    names = sorted(n for n in z.namelist() if n.lower().endswith(".hgu1"))
    res = {
        "archive": str(path), "class_files": len(names), "images": 0,
        "bad_header": [], "trailing_bytes": [], "multi_code_files": [],
        "undecodable_codes": [], "non_syllable_codes": [],
        "code_matches_filename_stem": 0, "code_vs_stem_checked": 0,
        "per_class_counts": [], "type_bytes": Counter(), "min_side": None, "max_side": None,
    }
    for nm in names:
        blob = z.read(nm)
        stem = nm.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
        if not blob.startswith(b"HGU1"):
            res["bad_header"].append(nm)
            continue
        off, codes, cnt = 8, set(), 0
        while off + 6 <= len(blob):
            code = blob[off:off + 2]
            w, h, t = blob[off + 2], blob[off + 3], blob[off + 4]
            off += 6 + w * h
            if off > len(blob):
                break
            codes.add(code)
            cnt += 1
            res["type_bytes"][t] += 1
            res["min_side"] = min(x for x in (res["min_side"], w, h) if x is not None)
            res["max_side"] = max(x for x in (res["max_side"] or 0, w, h))
        if off != len(blob):
            res["trailing_bytes"].append({"file": nm, "consumed": off, "size": len(blob)})
        res["images"] += cnt
        res["per_class_counts"].append(cnt)
        if len(codes) != 1:
            res["multi_code_files"].append(nm)
            continue
        code = next(iter(codes))
        # PE92 names each file after its code; SERI95 prefixes the name, so only
        # compare when the stem is exactly 4 hex digits.
        if len(stem) == 4:
            res["code_vs_stem_checked"] += 1
            if code.hex() == stem:
                res["code_matches_filename_stem"] += 1
        try:
            ch = code.decode("euc-kr")
        except UnicodeDecodeError:
            res["undecodable_codes"].append(code.hex())
            continue
        if not (len(ch) == 1 and 0xAC00 <= ord(ch) <= 0xD7A3):
            res["non_syllable_codes"].append({"hex": code.hex(), "decoded": ch})

    counts = sorted(res.pop("per_class_counts"))
    res["per_class_min"] = counts[0] if counts else 0
    res["per_class_median"] = counts[len(counts) // 2] if counts else 0
    res["per_class_max"] = counts[-1] if counts else 0
    res["type_bytes"] = {str(k): v for k, v in res["type_bytes"].items()}
    res["all_checks_pass"] = not (res["bad_header"] or res["trailing_bytes"] or res["multi_code_files"]
                                  or res["undecodable_codes"] or res["non_syllable_codes"])
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archives", nargs="+")
    ap.add_argument("--out", default="reports/dataset_verification.json")
    a = ap.parse_args()

    results = []
    for p in a.archives:
        print("checking", p)
        r = check_archive(p)
        results.append(r)
        print(f"  {r['class_files']} class files, {r['images']} images, "
              f"per-class {r['per_class_min']}/{r['per_class_median']}/{r['per_class_max']}, "
              f"code==stem {r['code_matches_filename_stem']}/{r['code_vs_stem_checked']}, "
              f"all_checks_pass={r['all_checks_pass']}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "claim": "The HGU1 2-byte class code is KS X 1001 wansung (EUC-KR).",
        "verdict": "confirmed" if all(r["all_checks_pass"] for r in results) else "FAILED",
        "archives": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("->", out)
    if not all(r["all_checks_pass"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
