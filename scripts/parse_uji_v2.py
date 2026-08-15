import argparse, csv, zipfile, math
from pathlib import Path
import numpy as np

def get_txt(path):
    p=Path(path)
    if p.suffix.lower()==".zip":
        with zipfile.ZipFile(p) as z:
            name=next(n for n in z.namelist() if n.endswith("ujipenchars2.txt"))
            return z.read(name).decode("utf-8")
    return p.read_text(encoding="utf-8")

def parse(text):
    lines=text.splitlines(); i=0
    while i<len(lines):
        s=lines[i].strip()
        if not s.startswith("WORD "):
            i+=1; continue
        parts=s.split(maxsplit=2); char=parts[1]; session=parts[2]; i+=1
        while i<len(lines) and not lines[i].strip().startswith("NUMSTROKES"):
            i+=1
        ns=int(lines[i].split()[1]); i+=1
        strokes=[]
        for _ in range(ns):
            parts=lines[i].split(); i+=1
            n=int(parts[1]); hi=parts.index("#")
            vals=list(map(int,parts[hi+1:]))
            while len(vals)<2*n:
                vals += list(map(int,lines[i].split())); i+=1
            strokes.append(np.asarray(vals[:2*n],float).reshape(n,2))
        yield char,session,strokes

def summarize(char,session,strokes):
    site="UPV" if "_UPV_" in session else "UJI"
    scale=152.0 if site=="UPV" else 100.0
    pp=[]; path=0
    stroke_lengths=[]
    for s in strokes:
        s=s/scale
        pp.append(s)
        if len(s)>1:
            d=np.diff(s,axis=0); length=float(np.sqrt((d*d).sum(1)).sum())
        else:length=0
        stroke_lengths.append(length); path+=length
    pts=np.concatenate(pp)
    ex=float(np.ptp(pts[:,0])); ey=float(np.ptp(pts[:,1]))
    writer=session.rsplit("-",1)[0]
    repetition=session.rsplit("-",1)[-1]
    return {
        "char":char,"session":session,"writer":writer,"repetition":repetition,"site":site,
        "stroke_count":len(strokes),"point_count":sum(len(x) for x in strokes),
        "path_length_mm":path,"extent_x_mm":ex,"extent_y_mm":ey,
        "aspect":ex/max(ey,1e-9),
        "mean_stroke_length_mm":float(np.mean(stroke_lengths)),
        "max_stroke_length_mm":float(np.max(stroke_lengths)),
    }

def main():
    ap=argparse.ArgumentParser();ap.add_argument("input");ap.add_argument("--out",required=True);a=ap.parse_args()
    rows=[summarize(*x) for x in parse(get_txt(a.input))]
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    with open(out,"w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print(len(rows),"samples ->",out)

if __name__=="__main__":main()
