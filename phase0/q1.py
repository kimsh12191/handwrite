"""
Q1 — 한 장의 필기 사진에서 자모가 분리되는가?

plan.md v4 의 밑돌. 세 가지를 한 번에 답한다.
  1. 자모 분리가 되는가          -> 성공률 + 실패 사유별 집계
  2. 자모 빈도가 실제로 충분한가  -> 실측 표 (계획의 추정치는 눈대중이었다)
  3. 변동이 어떻게 생겼는가      -> 표준편차 맵 (골격 층 섭동의 사양)

VLM·ODE·최적화 전부 쓰지 않는다.

사용법
------
  # 1) 블록 검출 -> 오버레이 보고 전사 파일 채우기
  python -m phase0.q1 detect 사진.jpg -o out/

  # 2) out/transcript.txt 를 열어 블록마다 전사 입력 ('-' 는 건너뜀)

  # 3) 실행
  python -m phase0.q1 run 사진.jpg out/transcript.txt -o out/

블록이 두 줄을 합치거나 한 줄을 쪼개면 --gap-y / --gap-x 를 조정한다.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

from . import layout, segment, variation, hangul


# ---------------------------------------------------------------------------
def cmd_detect(args):
    os.makedirs(args.out, exist_ok=True)
    r = layout.analyze(args.photo, bg_sigma=args.bg_sigma, thresh=args.thresh,
                       gap_x=args.gap_x, gap_y=args.gap_y)
    blks = r['blocks']
    ov = layout.overlay(r['ink'], blks, os.path.join(args.out, 'overlay.png'),
                        dropped=r['dropped'])

    tpath = os.path.join(args.out, 'transcript.txt')
    if os.path.exists(tpath) and not args.force:
        print(f'! {tpath} 이미 있음 — 덮어쓰려면 --force')
    else:
        with open(tpath, 'w') as f:
            f.write('# 블록별 전사. overlay.png 의 빨간 번호와 대응.\n'
                    '#   - 공백은 어절 구분으로 쓴다.\n'
                    '#   - 한글이 아닌 문자(괄호·영문)도 보이는 대로 적어라.\n'
                    '#     위치 맞추기에 쓰이고, 한글만 골라 분석한다.\n'
                    "#   - 건너뛸 블록은 '-' 로 둔다.\n\n")
            for i, b in enumerate(blks):
                x0, y0, x1, y1 = b['box']
                f.write(f"{i}: -    # {x1-x0}x{y1-y0}px  성분{len(b['comps'])}"
                        f"  at({x0},{y0})\n")
        print(f'전사 서식  {tpath}')

    print(f'오버레이   {ov}')
    print(f'블록 {len(blks)}개 / 텍스트 성분 {len(r["comps"])}개 / '
          f'도형 제거 {len(r["dropped"])}개')
    hs = [b['box'][3] - b['box'][1] for b in blks]
    if hs:
        print(f'블록 높이 중앙 {int(np.median(hs))}px  최대 {max(hs)}px'
              f'  (중앙의 2배를 넘는 블록은 두 줄이 합쳐진 것)')
        for i, b in enumerate(blks):
            h = b['box'][3] - b['box'][1]
            if h > 2.0 * np.median(hs):
                print(f'  ! 블록 {i} 높이 {h}px — --gap-y 를 줄여 보라')


# ---------------------------------------------------------------------------
def read_transcript(path):
    out = {}
    for ln in open(path):
        ln = ln.split('#')[0].strip()
        if not ln or ':' not in ln:
            continue
        k, v = ln.split(':', 1)
        v = v.strip()
        if not k.strip().isdigit() or v in ('', '-'):
            continue
        out[int(k)] = v
    return out


def cmd_run(args):
    os.makedirs(args.out, exist_ok=True)
    tr = read_transcript(args.transcript)
    if not tr:
        sys.exit('전사가 비었다. transcript.txt 를 채워라.')

    r = layout.analyze(args.photo, bg_sigma=args.bg_sigma, thresh=args.thresh,
                       gap_x=args.gap_x, gap_y=args.gap_y)
    blks, mask = r['blocks'], r['mask']
    print(f'블록 {len(blks)}개 중 전사된 것 {len(tr)}개')

    inst = defaultdict(list)          # (자모, 역할) -> [dict(img,w,h,src)]
    warns = Counter()
    n_glyph = n_ok = n_skip = 0
    per_block = []

    for bi, text in sorted(tr.items()):
        if bi >= len(blks):
            warns['블록 번호 범위 밖'] += 1
            continue
        pieces = segment.split_block(blks[bi], mask, text)
        b_ok = b_bad = 0
        for p in pieces:
            if p['ch'] is None:
                n_skip += 1
                continue
            n_glyph += 1
            x0, y0, x1, y1 = p['box']
            if x1 - x0 < args.min_glyph or y1 - y0 < args.min_glyph:
                warns['글자가 너무 작음'] += 1
                b_bad += 1
                continue
            gm = mask[y0:y1, x0:x1]
            jam, lay, w = segment.glyph_jamo(gm, p['ch'])
            for t in w:
                warns[t] += 1
            need = {role for _, role in hangul.parts(p['ch'])[0]}
            if {role for _, role, _ in jam} != need:
                warns['역할 누락'] += 1
                b_bad += 1
                continue
            for j, role, box in jam:
                img = variation.normalize(gm, box, args.size)
                if img is None:
                    continue
                jy0, jy1, jx0, jx1 = box
                # 자모 단위는 '자모+역할' — 초성 ㄱ 과 종성 ㄱ 은 위치도 크기도 다르다
                inst[(j, role)].append(
                    dict(img=img, w=jx1 - jx0, h=jy1 - jy0,
                         src=f'b{bi}:{p["ch"]}'))
            n_ok += 1
            b_ok += 1
        per_block.append((bi, text, b_ok, b_bad))

    # -- 리포트 ------------------------------------------------------------
    print(f'\n한글 글자 {n_glyph}개 (비한글 {n_skip}개 건너뜀)')
    print(f'자모 분리 성공 {n_ok}  실패 {n_glyph - n_ok}  '
          f'성공률 {100*n_ok/max(n_glyph,1):.1f}%')
    if warns:
        print('\n경고 집계:')
        for k, v in warns.most_common():
            print(f'  {v:4d}  {k}')

    rows = sorted(inst.items(), key=lambda kv: -len(kv[1]))
    n_inst = sum(len(v) for v in inst.values())
    by_jamo = Counter()
    for (j, _), v in inst.items():
        by_jamo[j] += len(v)
    print(f'\n자모+역할 {len(rows)}종 / 인스턴스 {n_inst}개 '
          f'/ 자모만으로는 {len(by_jamo)}종')
    print('  자모 역할   n')
    for (j, role), v in rows[:20]:
        print(f'   {j}  {role:4s}  {len(v):3d}')

    stats_rows, report = [], {}
    sheet_dir = os.path.join(args.out, 'jamo')
    os.makedirs(sheet_dir, exist_ok=True)
    for (j, role), v in rows:
        if len(v) < args.min_n:
            continue
        name = f'{j}:{role}'
        st = variation.stats(v)
        variation.sheet(name, st,
                        os.path.join(sheet_dir, f'{ord(j):04X}_{role}.png'))
        stats_rows.append((name, st))
        report[name] = {k: st[k] for k in
                        ('n', 'frame_var', 'shape_var', 'size_cv', 'aspect_cv',
                         'chamfer', 'chamfer_sd', 'chamfer_frame',
                         'pair_mean', 'pair_sd')}

    if stats_rows:
        idx = variation.index_sheet(stats_rows, os.path.join(args.out, 'index.png'))
        print(f'\nn>={args.min_n} 인 자모+역할 {len(stats_rows)}종 시트 생성')
        print(f'  {sheet_dir}/  및  {idx}')
        print('\n  자모:역할   n   chamfer(형태)  chamfer(프레임)  형태변동  '
              '크기CV 비율CV   [L2]')
        for name, st in stats_rows:
            print(f'  {name:9s} {st["n"]:3d}    '
                  f'{st["chamfer"]*100:5.2f}%±{st["chamfer_sd"]*100:4.2f}    '
                  f'{st["chamfer_frame"]*100:5.2f}%        '
                  f'{st["shape_var"]:.3f}   {st["size_cv"]:.3f}  '
                  f'{st["aspect_cv"]:.3f}  [{st["pair_mean"]:.2f}]')
    else:
        print(f'\n! n>={args.min_n} 인 것이 없다. --min-n 을 낮추거나 전사를 더 채워라.')

    out = dict(photo=args.photo, n_glyph=n_glyph, n_ok=n_ok,
               success_rate=n_ok / max(n_glyph, 1),
               warns=dict(warns),
               counts_by_jamo=dict(by_jamo),
               counts={f'{j}:{r}': len(v) for (j, r), v in rows},
               stats=report,
               per_block=[dict(block=b, text=t, ok=o, bad=x)
                          for b, t, o, x in per_block])
    jp = os.path.join(args.out, 'q1.json')
    json.dump(out, open(jp, 'w'), ensure_ascii=False, indent=1)
    print(f'\n{jp}')

    print('\n-- Q1 판정 근거 --')
    print(f'  자모 분리 성공률      {100*n_ok/max(n_glyph,1):.1f}%  '
          f'({n_ok}/{n_glyph})')
    if by_jamo:
        j, c = by_jamo.most_common(1)[0]
        print(f'  최다 자모(역할 무시)  {j}  n={c}   [계획 추정: ㅇ~40-50, ㅁ~15]')
    print(f'  n>=10 인 자모+역할    {sum(1 for _, v in rows if len(v) >= 10)}종')
    print(f'  n>=5  인 자모+역할    {sum(1 for _, v in rows if len(v) >= 5)}종')
    if stats_rows:
        ch = [st['chamfer'] for _, st in stats_rows]
        cf = [st['chamfer_frame'] for _, st in stats_rows]
        pm = [st['pair_mean'] for _, st in stats_rows]
        print(f'  chamfer 중앙(형태)    {np.median(ch)*100:.2f}%  '
              f'= 같은 자모끼리 평균 이만큼 어긋난다')
        print(f'  chamfer 중앙(프레임)  {np.median(cf)*100:.2f}%  '
              f'= 위치까지 포함하면 이만큼')
        print(f'  [참고] L2 중앙        {np.median(pm):.3f}  '
              f'(0=동일, 1.41=직교) — 얇은 획에서 무의미')


# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(prog='phase0.q1', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    def common(q):
        q.add_argument('-o', '--out', default='q1_out')
        q.add_argument('--bg-sigma', type=float, default=80,
                       help='조명 평탄화 반경 (기본 80)')
        q.add_argument('--thresh', type=float, default=0.30,
                       help='이진화 임계 (기본 0.30)')
        q.add_argument('--gap-x', type=int, default=88,
                       help='같은 줄로 볼 최대 가로 간격 px (기본 88)')
        q.add_argument('--gap-y', type=int, default=12,
                       help='세로 병합 여유 px. 줄이 합쳐지면 낮춰라 (기본 12)')

    d = sub.add_parser('detect', help='블록 검출 + 오버레이 + 전사 서식')
    d.add_argument('photo')
    d.add_argument('--force', action='store_true', help='전사 파일 덮어쓰기')
    common(d)
    d.set_defaults(fn=cmd_detect)

    q = sub.add_parser('run', help='자모 분리 + 변동 측정')
    q.add_argument('photo')
    q.add_argument('transcript')
    q.add_argument('--min-n', type=int, default=5,
                   help='시트를 만들 최소 인스턴스 수 (기본 5)')
    q.add_argument('--size', type=int, default=variation.SIZE,
                   help='자모 정규화 해상도 (기본 128)')
    q.add_argument('--min-glyph', type=int, default=24,
                   help='이보다 작은 글자는 버림 px (기본 24)')
    common(q)
    q.set_defaults(fn=cmd_run)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == '__main__':
    main()
