import argparse, zipfile
from pathlib import Path
import numpy as np
from PIL import Image

def read_hgu1(path):
    with open(path,"rb") as f:
        header=f.read(8)
        if not header.startswith(b"HGU1"):
            raise ValueError(f"{path}: invalid header {header!r}")
        n=0
        while True:
            code=f.read(2)
            if len(code)<2: break
            h=f.read(4)
            if len(h)<4: break
            width,height,img_type,reserved=h
            size=width*height
            buf=f.read(size)
            if len(buf)!=size: break
            arr=np.frombuffer(buf,dtype=np.uint8).reshape(height,width)
            yield n,code,arr
            n+=1

def extract_zip(zip_path,out_dir,limit=0):
    zip_path=Path(zip_path); out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    temp=out_dir/"_archive"
    temp.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(temp)
    count=0
    for hp in temp.rglob("*"):
        if not hp.is_file(): continue
        try:
            for idx,code,arr in read_hgu1(hp):
                # Do not guess legacy Korean encoding. Retain raw 2-byte label hex.
                label=code.hex()
                d=out_dir/label; d.mkdir(exist_ok=True)
                Image.fromarray(arr,"L").save(d/f"{hp.stem}_{idx:07d}.png")
                count+=1
                if limit and count>=limit:
                    print("limit reached",count); return
        except ValueError:
            continue
    print("extracted",count,"images")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("--out",required=True)
    ap.add_argument("--limit",type=int,default=0)
    a=ap.parse_args()
    extract_zip(a.zip,a.out,a.limit)

if __name__=="__main__":main()
