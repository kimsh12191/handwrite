import subprocess,sys,zipfile,shutil
from pathlib import Path

PY=sys.executable
def run(*args):
    print("+"," ".join(map(str,args)))
    subprocess.run([PY,*map(str,args)],check=True)

def unzip(src,dst):
    dst=Path(dst);dst.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(src) as z:z.extractall(dst)

def main():
    Path("data/extracted").mkdir(parents=True,exist_ok=True)
    Path("data/features").mkdir(parents=True,exist_ok=True)

    run("scripts/make_probe_set.py")

    pe=Path("data/raw/PE92_test.zip")
    if pe.exists():
        run("scripts/extract_hgu1.py",pe,"--out","data/extracted/pe92")
        run("scripts/analyze_images.py","data/extracted/pe92","--out","data/features/pe92_shapes.csv")
        run("scripts/pca_report.py","data/features/pe92_shapes.csv","--out","reports/pe92_pca.json")

    se=Path("data/raw/SERI_Test.zip")
    if se.exists():
        run("scripts/extract_hgu1.py",se,"--out","data/extracted/seri95")
        run("scripts/analyze_images.py","data/extracted/seri95","--out","data/features/seri95_shapes.csv")
        run("scripts/pca_report.py","data/features/seri95_shapes.csv","--out","reports/seri95_pca.json")

    uj=Path("data/raw/uji_pen_characters_v2.zip")
    if uj.exists():
        run("scripts/parse_uji_v2.py",uj,"--out","data/features/uji_v2_motor.csv")

    em=Path("data/raw/emnist_gzip.zip")
    if em.exists():
        run("scripts/extract_emnist.py",em,"--split","byclass","--per-class","200","--out","data/extracted/emnist_byclass")
        run("scripts/analyze_images.py","data/extracted/emnist_byclass","--out","data/features/emnist_shapes.csv")
        run("scripts/pca_report.py","data/features/emnist_shapes.csv","--out","reports/emnist_pca.json")

    nf=Path("data/raw/NanumFont_TTF_ALL.zip")
    if nf.exists():
        fontdir=Path("data/extracted/nanum_fonts"); unzip(nf,fontdir)
        run("scripts/render_fonts.py",fontdir,"--out","data/extracted/nanum_font_probe")
        run("scripts/analyze_images.py","data/extracted/nanum_font_probe","--out","data/features/nanum_font_shapes.csv")

    # Manually supplied Clova 109 TTFs:
    manual=Path("data/fonts_manual")
    if any(p.suffix.lower() in {".ttf",".otf",".ttc"} for p in manual.rglob("*")):
        run("scripts/render_fonts.py",manual,"--out","data/extracted/clova109_probe")
        run("scripts/analyze_images.py","data/extracted/clova109_probe","--out","data/features/clova109_shapes.csv")

if __name__=="__main__":main()
