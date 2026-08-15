import argparse,csv,math
from pathlib import Path
import numpy as np
from PIL import Image

def mask(path):
    a=np.asarray(Image.open(path).convert("L"),np.float32)
    t=(np.percentile(a,10)+np.percentile(a,90))/2
    m=a<t
    if m.mean()>.5:m=~m
    return m

def count_components(m, target=True):
    h,w=m.shape;seen=np.zeros_like(m,bool);n=0
    for y in range(h):
        for x in range(w):
            if bool(m[y,x])!=target or seen[y,x]:continue
            n+=1;st=[(y,x)];seen[y,x]=1
            while st:
                yy,xx=st.pop()
                for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
                    ny,nx=yy+dy,xx+dx
                    if 0<=ny<h and 0<=nx<w and bool(m[ny,nx])==target and not seen[ny,nx]:
                        seen[ny,nx]=1;st.append((ny,nx))
    return n

def holes(ink):
    bg=~ink;h,w=bg.shape;seen=np.zeros_like(bg,bool);n=0
    for y in range(h):
        for x in range(w):
            if not bg[y,x] or seen[y,x]:continue
            st=[(y,x)];seen[y,x]=1;touch=False
            while st:
                yy,xx=st.pop()
                if yy in (0,h-1) or xx in (0,w-1):touch=True
                for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
                    ny,nx=yy+dy,xx+dx
                    if 0<=ny<h and 0<=nx<w and bg[ny,nx] and not seen[ny,nx]:
                        seen[ny,nx]=1;st.append((ny,nx))
            if not touch:n+=1
    return n

def feat(path):
    m=mask(path);ys,xs=np.nonzero(m)
    if len(xs)<3:return None
    x0,x1,y0,y1=xs.min(),xs.max(),ys.min(),ys.max()
    crop=m[y0:y1+1,x0:x1+1]; bw=x1-x0+1;bh=y1-y0+1
    xn=(xs-x0)/max(1,bw-1);yn=(ys-y0)/max(1,bh-1)
    cx,cy=xn.mean(),yn.mean();X=np.stack([xn-cx,yn-cy],1)
    cov=X.T@X/max(1,len(X));vals,vecs=np.linalg.eigh(cov);o=np.argsort(vals)[::-1]
    vals=vals[o];v=vecs[:,o[0]]
    # Simple 4-neighbour boundary edge count
    p=np.pad(crop.astype(np.uint8),1);c=p[1:-1,1:-1]
    per=int(((c!=p[:-2,1:-1])|(c!=p[2:,1:-1])|(c!=p[1:-1,:-2])|(c!=p[1:-1,2:])).sum())
    area=int(crop.sum())
    # label = nearest useful parent directory; works for Uxxxx dirs and HGU hex dirs
    label=path.parent.name
    return {
        "path":str(path),"label":label,"width":bw,"height":bh,"aspect":bw/max(1,bh),
        "area_ratio":area/(bw*bh),"centroid_x":float(cx),"centroid_y":float(cy),
        "principal_angle":float(math.atan2(v[1],v[0])),
        "moment_eigen_ratio":float(vals[0]/max(vals[1],1e-9)),
        "lr_asym":float(abs((xn<.5).mean()-(xn>=.5).mean())),
        "tb_asym":float(abs((yn<.5).mean()-(yn>=.5).mean())),
        "compactness":float(per*per/(4*math.pi*area)) if area else float("nan"),
        "components":count_components(crop,True),"holes":holes(crop)
    }

def main():
    ap=argparse.ArgumentParser();ap.add_argument("root");ap.add_argument("--out",required=True);ap.add_argument("--limit",type=int,default=0);a=ap.parse_args()
    rows=[]
    for p in Path(a.root).rglob("*"):
        if p.suffix.lower() not in {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}:continue
        try:
            r=feat(p)
            if r:rows.append(r)
        except Exception as e: print("SKIP",p,e)
        if a.limit and len(rows)>=a.limit:break
    if not rows: raise SystemExit("no images")
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    with open(out,"w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print(len(rows),"rows ->",out)

if __name__=="__main__":main()
