"""Run the whole Phase 1 -> Phase 2 chain over whatever data actually arrived.

Each stage is skipped if its input is missing, and what ran is printed at the
end. A stage that is skipped because a host was blocked must not look the same
as a stage that passed.

Split choice matters here. PE92's test split holds ~10 samples per class, which
does not clear the per-class PCA threshold for even one class, so the Hangul
analysis runs on PE92 train (~81 per class over all 2350) and SERI95 test (~100
over 520). `--drop-tail 5` follows the HangulDB README's warning that PE92's
mislabeled samples cluster at the end of each class file.
"""
import subprocess, sys, zipfile
from pathlib import Path

PY = sys.executable
RAW = Path("data/raw")
ran, skipped = [], []


def run(*args):
    print("+", " ".join(map(str, args)))
    subprocess.run([PY, *map(str, args)], check=True)


def stage(name, inputs, fn):
    missing = [str(p) for p in inputs if not Path(p).exists()]
    if missing:
        skipped.append((name, f"missing {', '.join(missing)}"))
        return
    fn()
    ran.append(name)


def unzip(src, dst):
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as z:
        z.extractall(dst)


def hangul(tag, archive, extra=()):
    def go():
        run("scripts/extract_hgu1.py", archive, "--out", f"data/extracted/{tag}", *extra)
        run("scripts/analyze_images.py", f"data/extracted/{tag}", "--out", f"data/features/{tag}_shapes.csv")
        run("scripts/pca_report.py", f"data/features/{tag}_shapes.csv", "--out", f"reports/{tag}_pca.json")
        # Phase 2 proper: spatial basis, its own validity control, and the §5 context question.
        run("scripts/shape_space.py", f"data/extracted/{tag}", "--out", f"reports/{tag}_shape_space.json",
            "--classes", "120", "--modes-dir", f"reports/modes/{tag}")
        run("scripts/jamo_context.py", f"data/extracted/{tag}", "--out", f"reports/{tag}_jamo_context.json")
    return go


def main():
    for d in ("data/extracted", "data/features", "reports"):
        Path(d).mkdir(parents=True, exist_ok=True)

    run("scripts/make_probe_set.py")

    hangul_archives = [p for p in (RAW / "PE92_train.zip", RAW / "SERI_Test.zip") if p.exists()]
    if hangul_archives:
        stage("verify_dataset", hangul_archives,
              lambda: run("scripts/verify_dataset.py", *hangul_archives))

    stage("seri95", [RAW / "SERI_Test.zip"], hangul("seri95", RAW / "SERI_Test.zip"))
    stage("pe92", [RAW / "PE92_train.zip"], hangul("pe92", RAW / "PE92_train.zip", ("--drop-tail", "5")))

    # The measurement is only admissible if it reads a known-low-dimensional
    # family as low dimensional, so the control runs alongside, not optionally.
    for tag, cls in (("seri95", "UAC00_가"), ("pe92", "UAC00_가")):
        stage(f"{tag}_basis_control", [f"data/extracted/{tag}/{cls}"],
              lambda t=tag, c=cls: run("scripts/basis_control.py", f"data/extracted/{t}/{c}",
                                       "--out", f"reports/{t}_basis_control.json"))

    stage("uji_v2", [RAW / "uji_pen_characters_v2.zip"],
          lambda: run("scripts/parse_uji_v2.py", RAW / "uji_pen_characters_v2.zip",
                      "--out", "data/features/uji_v2_motor.csv"))

    def emnist():
        run("scripts/extract_emnist.py", RAW / "emnist_gzip.zip", "--split", "byclass",
            "--per-class", "200", "--out", "data/extracted/emnist_byclass")
        run("scripts/analyze_images.py", "data/extracted/emnist_byclass",
            "--out", "data/features/emnist_shapes.csv")
        run("scripts/pca_report.py", "data/features/emnist_shapes.csv", "--out", "reports/emnist_pca.json")
    stage("emnist", [RAW / "emnist_gzip.zip"], emnist)

    def nanum():
        fontdir = Path("data/extracted/nanum_fonts")
        unzip(RAW / "NanumFont_TTF_ALL.zip", fontdir)
        run("scripts/render_fonts.py", fontdir, "--out", "data/extracted/nanum_font_probe")
        run("scripts/analyze_images.py", "data/extracted/nanum_font_probe",
            "--out", "data/features/nanum_font_shapes.csv")
    stage("nanum_fonts", [RAW / "NanumFont_TTF_ALL.zip"], nanum)

    manual = Path("data/fonts_manual")
    if manual.exists() and any(p.suffix.lower() in {".ttf", ".otf", ".ttc"} for p in manual.rglob("*")):
        run("scripts/render_fonts.py", manual, "--out", "data/extracted/clova109_probe")
        run("scripts/analyze_images.py", "data/extracted/clova109_probe",
            "--out", "data/features/clova109_shapes.csv")
        ran.append("clova109")
    else:
        skipped.append(("clova109", "no TTFs under data/fonts_manual"))

    print("\n=== bootstrap summary ===")
    for name in ran:
        print(f"  ran      {name}")
    for name, why in skipped:
        print(f"  SKIPPED  {name:20s} {why}")
    if skipped:
        print("\nSkipped stages are missing data, not passing tests. Check which Phase 1 sources\n"
              "are actually in hand before treating the reports as covering that script.")


if __name__ == "__main__":
    main()
