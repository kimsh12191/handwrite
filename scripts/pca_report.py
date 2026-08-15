import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np

FEATS=["aspect","area_ratio","centroid_x","centroid_y","principal_angle","moment_eigen_ratio","lr_asym","tb_asym","compactness","components","holes"]

def main():
    ap=argparse.ArgumentParser();ap.add_argument("csv");ap.add_argument("--out",required=True);ap.add_argument("--min-n",type=int,default=15);a=ap.parse_args()
    with open(a.csv,encoding="utf-8-sig") as f:rows=list(csv.DictReader(f))
    g=defaultdict(list)
    for r in rows:g[r["label"]].append(r)
    rep={}
    for lab,rs in g.items():
        X=[]
        for r in rs:
            try:X.append([float(r[k]) for k in FEATS])
            except:pass
        X=np.asarray(X,float) if X else np.empty((0,len(FEATS)))
        X=X[np.isfinite(X).all(1)] if len(X) else X
        if len(X)<a.min_n:continue
        sd=X.std(0);sd[sd<1e-9]=1;Z=(X-X.mean(0))/sd
        _,S,Vt=np.linalg.svd(Z,full_matrices=False);ratio=(S*S);ratio/=ratio.sum()
        pcs=[]
        for i in range(min(5,len(ratio))):
            tops=sorted(zip(FEATS,Vt[i]),key=lambda z:abs(z[1]),reverse=True)[:5]
            pcs.append({"pc":i+1,"explained":float(ratio[i]),"top_loadings":[[k,float(v)] for k,v in tops]})
        rep[lab]={"n":len(X),"pcs":pcs}
    Path(a.out).write_text(json.dumps(rep,ensure_ascii=False,indent=2),encoding="utf-8")
    print(len(rep),"groups ->",a.out)

if __name__=="__main__":main()
