import argparse, json, time
from pathlib import Path
import requests

def download(url, out, retries=3):
    out = Path(out)
    part = out.with_suffix(out.suffix + ".part")
    headers = {"User-Agent":"Mozilla/5.0 handwriting-renderer-space-study/1.0"}
    for attempt in range(1,retries+1):
        try:
            with requests.get(url, stream=True, timeout=(20,180), headers=headers, allow_redirects=True) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length",0) or 0)
                got = 0
                with open(part,"wb") as f:
                    for chunk in r.iter_content(1024*1024):
                        if chunk:
                            f.write(chunk); got += len(chunk)
                            if total:
                                print(f"\r{out.name}: {got/1024/1024:.1f}/{total/1024/1024:.1f} MB",end="")
                print()
            part.replace(out)
            return
        except Exception as e:
            print(f"attempt {attempt}/{retries} failed: {e}")
            if part.exists(): part.unlink()
            if attempt == retries: raise
            time.sleep(2*attempt)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",default="public_manifest.json")
    ap.add_argument("--raw",default="data/raw")
    ap.add_argument("--only",nargs="*",default=None,help="source ids")
    a=ap.parse_args()

    items=json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    raw=Path(a.raw); raw.mkdir(parents=True,exist_ok=True)
    for item in items:
        if not item["auto"]: continue
        if a.only and item["id"] not in a.only: continue
        out=raw/item["filename"]
        if out.exists() and out.stat().st_size>0:
            print("EXISTS",out)
            continue
        print("\nDOWNLOAD",item["id"],item["url"])
        download(item["url"],out)

if __name__=="__main__":
    main()
