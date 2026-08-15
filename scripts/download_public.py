"""Fetch the public corpora listed in public_manifest.json.

Two acquisition methods:

  git   shallow-clone a repository and link its archives into data/raw. HangulDB
        ships its splits as zips inside the repo, and the clone is the only way
        to get the train splits — the per-file raw.githubusercontent URLs the
        manifest used before reach only the small test splits, which are too
        thin for per-class analysis (PE92 test holds ~10 samples per class).
  http  stream a single file.

A host that is unreachable is reported and skipped, not retried around. Whoever
runs this next needs to know which parts of Phase 1 they actually have.
"""
import argparse, json, subprocess, time
from pathlib import Path

import requests

ARCHIVE_SUFFIXES = {".zip", ".z01", ".z02", ".z03", ".z04", ".z05", ".z06", ".gz", ".tar"}


def download(url, out, retries=3):
    out = Path(out)
    part = out.with_suffix(out.suffix + ".part")
    headers = {"User-Agent": "Mozilla/5.0 handwriting-renderer-space-study/1.0"}
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=(20, 180), headers=headers, allow_redirects=True) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0) or 0)
                got = 0
                with open(part, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            got += len(chunk)
                            if total:
                                print(f"\r{out.name}: {got/1024/1024:.1f}/{total/1024/1024:.1f} MB", end="")
                print()
            part.replace(out)
            return True, ""
        except Exception as e:
            msg = str(e)
            print(f"attempt {attempt}/{retries} failed: {msg}")
            if part.exists():
                part.unlink()
            if attempt == retries:
                return False, msg
            time.sleep(2 * attempt)
    return False, "exhausted retries"


def clone(url, dest):
    """Shallow clone, then expose the repo's archives under data/raw via symlink."""
    dest = Path(dest)
    if (dest / ".git").exists():
        print("EXISTS", dest)
        return True, ""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1", url, str(dest)]
    print("+", " ".join(cmd))
    p = subprocess.run(cmd, env={"GIT_LFS_SKIP_SMUDGE": "1", "PATH": "/usr/bin:/bin:/usr/local/bin"},
                       capture_output=True, text=True)
    if p.returncode != 0:
        return False, (p.stderr or p.stdout).strip().splitlines()[-1] if (p.stderr or p.stdout) else "clone failed"
    return True, ""


def link_archives(src_dir, raw):
    linked = []
    for p in sorted(Path(src_dir).iterdir()):
        if p.is_file() and p.suffix.lower() in ARCHIVE_SUFFIXES:
            dst = raw / p.name
            if not dst.exists():
                dst.symlink_to(p.resolve())
            linked.append(p.name)
    return linked


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="public_manifest.json")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--src", default="data/src", help="where git sources are cloned")
    ap.add_argument("--only", nargs="*", default=None, help="source ids")
    a = ap.parse_args()

    items = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    raw = Path(a.raw)
    raw.mkdir(parents=True, exist_ok=True)
    got, blocked, skipped = [], [], []

    for item in items:
        if not item.get("auto"):
            skipped.append((item["id"], f"manual: {item['url']}"))
            continue
        if a.only and item["id"] not in a.only:
            continue
        method = item.get("method", "http")

        if method == "git":
            dest = Path(a.src) / item["filename"]
            print(f"\nCLONE {item['id']} {item['url']}")
            ok, err = clone(item["url"], dest)
            if ok:
                names = link_archives(dest, raw)
                got.append((item["id"], f"{len(names)} archives -> {raw}"))
            else:
                blocked.append((item["id"], err))
            continue

        out = raw / item["filename"]
        if out.exists() and out.stat().st_size > 0:
            print("EXISTS", out)
            got.append((item["id"], "already present"))
            continue
        print(f"\nDOWNLOAD {item['id']} {item['url']}")
        ok, err = download(item["url"], out)
        (got if ok else blocked).append((item["id"], "downloaded" if ok else err))

    print("\n=== acquisition summary ===")
    for label, rows in (("available", got), ("BLOCKED", blocked), ("manual", skipped)):
        for i, msg in rows:
            print(f"  {label:10s} {i:22s} {msg}")
    if blocked:
        print("\nBlocked sources are an egress/policy result, not a bug here. Record which ones\n"
              "are missing before drawing conclusions from whatever did download.")


if __name__ == "__main__":
    main()
