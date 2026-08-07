"""
자모 인스턴스 -> 변동 구조.

Q1 의 산출물.

  프레임 변동 — 자모를 **글자 프레임** 안에 놓고 잰 변동. 위치·크기·형태가
                모두 들어간다. 조립(자모->음절) 어긋남의 크기가 여기 있다.
  형태 변동   — 무게중심 정렬 후. 순수 형태. v4 의 '골격 층 섭동' 사양이
                되는 값이고, 표준편차 맵이 **어디가 흔들리는지**를 그림으로 준다.

실측으로 확인한 두 가지 (둘 다 처음에 틀리게 짰다가 고쳤다):

1. **자모 bbox 를 정사각으로 늘리면 안 된다.** ㅣ(5x100)·ㅡ(100x5) 처럼 사실상
   1차원인 자모가 얼룩으로 뭉개져 평균이 의미를 잃는다. 글자 프레임을 쓴다.
2. **정규화 L2 는 얇은 획의 유사도를 재지 못한다.** 같은 사람 같은 자모끼리도
   1.1 근처(0=동일, 1.41=직교)가 나온다. 유사도는 chamfer 로 잰다.
"""
import glob
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import (center_of_mass, shift as ndshift, gaussian_filter,
                           distance_transform_edt)

SIZE = 128          # 자모 정규화 해상도. 글자는 256, 자모는 그 절반이면 충분

# 한글이 찍히는 폰트를 찾는다. 내부망 장비에 없을 수 있으므로 없으면 로마자로
# 라벨을 바꾼다 (label() 참조). 시트가 두부(□)로 깨지면 읽을 수가 없다.
_FONT_HINTS = [
    '/usr/share/fonts/**/*Nanum*', '/usr/share/fonts/**/*NotoSansCJK*',
    '/usr/share/fonts/**/*NotoSerifCJK*', '/usr/share/fonts/**/*Malgun*',
    '/usr/share/fonts/**/*UnDotum*', '/usr/share/fonts/**/*Baekmuk*',
    '/usr/share/fonts/**/*gothic*', '/usr/share/fonts/**/*Gothic*',
    'C:/Windows/Fonts/malgun.ttf', '/System/Library/Fonts/AppleSDGothicNeo.ttc',
]
_font_cache = {}


def _find_font():
    if 'p' in _font_cache:
        return _font_cache['p']
    env = os.environ.get('HANDWRITE_FONT')
    cands = ([env] if env else []) + [f for pat in _FONT_HINTS
                                      for f in sorted(glob.glob(pat, recursive=True))]
    for f in cands:
        if not f or not os.path.isfile(f):
            continue
        try:
            ft = ImageFont.truetype(f, 14)
            if ft.getmask('가').getbbox() and ft.getmask('ㅁ').getbbox():
                _font_cache['p'] = f
                return f
        except Exception:
            continue
    _font_cache['p'] = None
    return None


def font(size=14):
    key = ('f', size)
    if key not in _font_cache:
        p = _find_font()
        try:
            _font_cache[key] = ImageFont.truetype(p, size) if p else ImageFont.load_default()
        except Exception:
            _font_cache[key] = ImageFont.load_default()
    return _font_cache[key]


def label(name):
    """한글 폰트가 없으면 '\u3141:cho' -> 'm:cho' 로 바꿔 준다."""
    if _find_font():
        return name
    from . import hangul
    j, _, role = name.partition(':')
    return f'{hangul.roman(j)}:{role}' if role else hangul.roman(j)


def normalize(glyph_mask, box, size=SIZE):
    """
    자모를 **글자 프레임 안에** 놓고 size x size 로 정규화 (0..1 float).

    자모 bbox 를 정사각으로 늘리면 안 된다 — ㅣ(5x100) 나 ㅡ(100x5) 같은
    사실상 1차원인 자모가 얼룩으로 뭉개진다(실측 확인). 글자 프레임을 쓰면
    자모의 **위치·크기·종횡비·형태가 전부 보존**되고, 제거되는 것은 글자의
    절대 크기뿐이다. 그 넷이 모두 개인차이므로 이쪽이 맞다.

    box: (y0,y1,x0,x1) — glyph_mask 좌표계의 자모 bbox
    """
    y0, y1, x0, x1 = box
    canvas = np.zeros_like(glyph_mask, dtype=np.uint8)
    canvas[y0:y1, x0:x1] = glyph_mask[y0:y1, x0:x1]
    if canvas.sum() == 0:
        return None
    im = Image.fromarray(canvas * 255).resize((size, size), Image.BILINEAR)
    return np.asarray(im, float) / 255.0


def normalize_tight(mask, size=SIZE):
    """bbox 를 정사각으로 늘림. 2차원 자모에만 의미가 있다 (ㅣ·ㅡ 에는 쓰지 말 것)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    sub = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8) * 255
    return np.asarray(Image.fromarray(sub).resize((size, size), Image.BILINEAR),
                      float) / 255.0


def align(imgs, iters=3, max_shift=0.12):
    """
    평균에 맞춰 평행이동만 정렬. 크기는 이미 정규화됐으므로 이동만 남는다.
    max_shift: 이미지 크기 대비 최대 이동량
    """
    if not imgs:
        return []
    out = [a.copy() for a in imgs]
    n = out[0].shape[0]
    lim = max(1, int(n * max_shift))
    for _ in range(iters):
        ref = np.mean(out, 0)
        rc = np.array(center_of_mass(ref))
        for i, a in enumerate(out):
            c = np.array(center_of_mass(a))
            if not np.all(np.isfinite(c)):
                continue
            d = np.clip(rc - c, -lim, lim)
            out[i] = ndshift(a, d, order=1, mode='constant', cval=0.0)
    return out


def dist(a, b, sigma=2.0):
    """
    정규화 L2. **얇은 획에는 쓸 수 없다** — 실측으로 확인했다.

    같은 사람의 같은 자모끼리 재도 1.1 근처가 나온다(0=동일, 1.41=직교).
    획이 몇 px 굵기라 조금만 어긋나도 겹침이 사라지기 때문이다.
    v3 가 목적함수에서 겪은 것과 같은 종류의 고장이라 남겨두되,
    유사도 판단에는 chamfer 를 쓴다.
    """
    fa, fb = gaussian_filter(a, sigma), gaussian_filter(b, sigma)
    fa = fa / (np.linalg.norm(fa) + 1e-9)
    fb = fb / (np.linalg.norm(fb) + 1e-9)
    return float(np.linalg.norm(fa - fb))


def chamfer(a, b, thr=0.35):
    """
    대칭 챔퍼 거리 — 이미지 변 길이 대비 비율.

    "A 의 잉크에서 가장 가까운 B 의 잉크까지 평균 몇 px 인가" 의 대칭판.
    얇은 구조에 맞는 척도이고, 값이 곧 '평균 몇 % 어긋났는가' 라 해석된다.
    L2 와 달리 획이 겹치지 않아도 가까우면 작은 값이 나온다.
    """
    A, B = a > thr, b > thr
    if not A.any() or not B.any():
        return float('nan')
    da = distance_transform_edt(~A)
    db = distance_transform_edt(~B)
    n = max(a.shape)
    return float(0.5 * (db[A].mean() + da[B].mean()) / n)


def stats(instances):
    """
    instances: [dict(img=글자프레임 정규화 2D, w=, h=), ...]

    두 가지 변동을 **따로** 낸다. 섞으면 무엇을 섭동해야 할지 알 수 없다.
      frame_var — 글자 프레임 그대로. 위치·크기·형태가 모두 들어간다.
                  조립(자모->음절) 어긋남의 크기가 여기 있다.
      shape_var — 무게중심 정렬 후. 순수 형태 변동.
                  골격 층 섭동의 크기가 여기 있다.
    """
    raw = [x['img'] for x in instances]
    n = len(raw)
    ali = align(raw)
    W = np.array([x['w'] for x in instances], float)
    H = np.array([x['h'] for x in instances], float)
    asp = W / np.maximum(H, 1)
    size = np.sqrt(W * H)

    ij = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pairs = [dist(ali[i], ali[j]) for i, j in ij]
    ch_a = [chamfer(ali[i], ali[j]) for i, j in ij]
    ch_f = [chamfer(raw[i], raw[j]) for i, j in ij]
    _m = lambda v: float(np.nanmean(v)) if v else 0.0
    _s = lambda v: float(np.nanstd(v)) if v else 0.0
    return dict(
        n=n, imgs=raw, aligned=ali,
        mean=np.mean(raw, 0), std=np.std(raw, 0),
        mean_a=np.mean(ali, 0), std_a=np.std(ali, 0),
        frame_var=float(np.std(raw, 0).mean()),
        shape_var=float(np.std(ali, 0).mean()),
        size_cv=float(size.std() / max(size.mean(), 1e-9)),
        aspect_cv=float(asp.std() / max(asp.mean(), 1e-9)),
        pair_mean=_m(pairs), pair_sd=_s(pairs),
        chamfer=_m(ch_a), chamfer_sd=_s(ch_a),      # 형태만 (정렬 후)
        chamfer_frame=_m(ch_f),                     # 위치 포함 (프레임)
        pairs=pairs, chamfers=ch_a,
    )


# ---------------------------------------------------------------------------
def _tile(a, invert=True):
    a = np.clip(a, 0, 1)
    return Image.fromarray(np.uint8((1 - a if invert else a) * 255)).convert('RGB')


def _heat(a):
    """표준편차 맵 — 흔들리는 곳이 붉게."""
    a = a / max(a.max(), 1e-9)
    r = np.uint8(255 * (1 - a * 0.15))
    g = np.uint8(255 * (1 - a))
    b = np.uint8(255 * (1 - a))
    return Image.fromarray(np.stack([r, g, b], -1))


def sheet(name, st, path, max_show=14, cell=SIZE, gap=6):
    """
    대조 시트. 윗줄 = 인스턴스 원본(글자 프레임), 아랫줄 = 요약.
    Q1 에서 눈으로 봐야 하는 그림이다.
    """
    show = st['imgs'][:max_show]
    cols = max(len(show), 5)
    W = gap + cols * (cell + gap)
    H = gap + 2 * (cell + gap) + 40
    im = Image.new('RGB', (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    dr.text((gap, 3), f"{label(name)}   n={st['n']}", fill=(0, 0, 0), font=font(15))
    dr.text((gap, 17), f"chamfer(shape) {st['chamfer']*100:.1f}%±{st['chamfer_sd']*100:.1f}"
                       f"   chamfer(frame) {st['chamfer_frame']*100:.1f}%"
                       f"   형태변동 {st['shape_var']:.3f}"
                       f"   크기CV {st['size_cv']:.3f}   비율CV {st['aspect_cv']:.3f}",
            fill=(60, 60, 60), font=font(12))
    y = 36 + gap
    for i, a in enumerate(show):
        im.paste(_tile(a), (gap + i * (cell + gap), y))
    y2 = y + cell + gap
    tiles = [('mean(frame)', _tile(st['mean'])),
             ('std(frame)', _heat(st['std'])),
             ('mean(aligned)', _tile(st['mean_a'])),
             ('std(aligned)', _heat(st['std_a'])),
             ('stack', _tile(np.clip(np.max(st['imgs'], 0), 0, 1)))]
    for i, (cap, tile) in enumerate(tiles):
        x = gap + i * (cell + gap)
        im.paste(tile, (x, y2))
        dr.text((x + 2, y2 + cell + 1), cap, fill=(90, 90, 90), font=font(11))
    im.save(path)
    return path


def index_sheet(rows, path, cell=96, gap=5):
    """자모별 평균/표준편차 한 장 요약. 왼쪽=평균(정렬 후), 오른쪽=표준편차."""
    if not rows:
        return None
    cols = min(7, len(rows))
    r = (len(rows) + cols - 1) // cols
    W = gap + cols * (cell * 2 + gap * 2)
    H = gap + r * (cell + 26)
    im = Image.new('RGB', (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    for k, (name, st) in enumerate(rows):
        cx, cy = k % cols, k // cols
        x = gap + cx * (cell * 2 + gap * 2)
        y = gap + cy * (cell + 26)
        im.paste(_tile(st['mean_a']).resize((cell, cell)), (x, y))
        im.paste(_heat(st['std_a']).resize((cell, cell)), (x + cell + gap, y))
        dr.text((x + 2, y + cell + 3),
                f"{label(name)} n={st['n']} ch={st['chamfer']*100:.1f}%",
                fill=(0, 0, 0), font=font(12))
    im.save(path)
    return path
