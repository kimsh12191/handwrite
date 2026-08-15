import argparse, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

def render(font_path,ch,size=160):
    font=ImageFont.truetype(str(font_path),size)
    c=Image.new("L",(256,256),255);d=ImageDraw.Draw(c)
    box=d.textbbox((0,0),ch,font=font)
    if box is None:return None
    w,h=box[2]-box[0],box[3]-box[1]
    d.text(((256-w)//2-box[0],(256-h)//2-box[1]),ch,font=font,fill=0)
    bb=ImageOps.invert(c).getbbox()
    return c.crop(bb) if bb else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("fonts")
    ap.add_argument("--probe",default="metadata/probe_set.json")
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    chars=json.loads(Path(a.probe).read_text(encoding="utf-8"))["all_probe_chars"]
    fonts=[p for p in Path(a.fonts).rglob("*") if p.suffix.lower() in {".ttf",".otf",".ttc"}]
    n=0
    for fi,fp in enumerate(fonts):
        fd=Path(a.out)/f"{fi:03d}_{fp.stem[:60]}"
        for ch in chars:
            try:
                im=render(fp,ch)
                if not im: continue
                # Detect tofu-ish blank glyph only coarsely; later analysis can filter duplicates.
                cd=fd/f"U{ord(ch):04X}"
                cd.mkdir(parents=True,exist_ok=True)
                im.save(cd/"glyph.png"); n+=1
            except Exception:
                pass
    print(n,"glyph renders from",len(fonts),"fonts")

if __name__=="__main__":main()
