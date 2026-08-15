import argparse, gzip, struct, zipfile, shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

def read_idx_images_gz(path):
    with gzip.open(path,"rb") as f:
        magic,n,rows,cols=struct.unpack(">IIII",f.read(16))
        if magic!=2051: raise ValueError("bad image IDX magic")
        a=np.frombuffer(f.read(),dtype=np.uint8).reshape(n,rows,cols)
    return a

def read_idx_labels_gz(path):
    with gzip.open(path,"rb") as f:
        magic,n=struct.unpack(">II",f.read(8))
        if magic!=2049: raise ValueError("bad label IDX magic")
        a=np.frombuffer(f.read(),dtype=np.uint8)
    return a

def parse_mapping(path):
    mp={}
    for line in Path(path).read_text().splitlines():
        p=line.split()
        if len(p)>=2: mp[int(p[0])]=chr(int(p[1]))
    return mp

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("--split",default="byclass",choices=["byclass","balanced","letters","digits","mnist"])
    ap.add_argument("--per-class",type=int,default=200)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    work=Path(a.out)/"_emnist_zip"
    work.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.zip) as z:z.extractall(work)

    prefix=f"emnist-{a.split}"
    imgs=next(work.rglob(prefix+"-train-images-idx3-ubyte.gz"))
    labels=next(work.rglob(prefix+"-train-labels-idx1-ubyte.gz"))
    mapping=next(work.rglob(prefix+"-mapping.txt"))

    X=read_idx_images_gz(imgs); y=read_idx_labels_gz(labels); mp=parse_mapping(mapping)
    counts=defaultdict(int); out=Path(a.out); saved=0
    for im,lab in zip(X,y):
        ch=mp.get(int(lab),f"label{int(lab)}")
        if counts[ch]>=a.per_class: continue
        d=out/f"U{ord(ch):04X}_{ch if ch.isalnum() else 'sym'}"; d.mkdir(parents=True,exist_ok=True)
        # EMNIST IDX orientation is transposed relative to normal display.
        im=im.T
        Image.fromarray(im,"L").save(d/f"{counts[ch]:04d}.png")
        counts[ch]+=1; saved+=1
    print("saved",saved,"images across",len(counts),"classes")

if __name__=="__main__":main()
