"""
필기 사진 -> 텍스트 블록(줄).

메모 사진은 도형·화살표·상자가 섞여 있어 세로 투영으로 줄을 나눌 수 없다
(도형이 페이지를 관통해 밴드가 하나로 붙는다). 그래서 순서가 다르다.

  1. 조명 평탄화 후 이진화
  2. 연결성분 중 '글자일 수 없는 것' 을 크기·채움비로 걷어냄 (도형 제거)
  3. 남은 것만 RLSA(가로 팽창) 로 묶어 줄 블록을 만듦

참조 사진 실측: 성분 339개 중 도형 11개 제거, 블록 27개(높이 중앙 144px).
"""
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import (gaussian_filter, label, binary_closing,
                           binary_dilation, find_objects)

# 도형 판정 기준 — 글자 성분의 상한. 참조 사진 실측(중앙 50x55, 90% 112x103)에
# 여유를 둔 값. 글자가 훨씬 크거나 작은 사진이면 조정해야 한다.
MAX_SIDE = 260      # px. 이보다 크면 도형
MIN_PIX = 150       # px. 이보다 작으면 잡티
MIN_FILL = 0.06     # bbox 대비 잉크 비율. 이보다 작으면 가늘고 긴 선


def ink_map(path, bg_sigma=80, thresh=0.30):
    """사진 -> (잉크강도 0..255, 이진 마스크). 잉크가 밝음."""
    a = np.array(Image.open(path).convert('L'), float)
    bg = gaussian_filter(a, bg_sigma)
    ink = np.clip(bg - a, 0, None)
    if ink.max() > 0:
        ink = ink / ink.max() * 255
    mask = binary_closing(ink > thresh * 255, np.ones((3, 3)))
    return ink, mask


def text_components(mask, max_side=MAX_SIDE, min_pix=MIN_PIX, min_fill=MIN_FILL):
    """
    연결성분을 글자 후보와 도형으로 가른다.
    returns (comps, dropped, text_mask)
      comps/dropped: [dict(x0,y0,x1,y1,w,h,npx), ...]
    """
    lab, _ = label(mask)
    comps, dropped = [], []
    tmask = np.zeros_like(mask)
    for i, s in enumerate(find_objects(lab), 1):
        if s is None:
            continue
        sub = (lab[s] == i)
        npx = int(sub.sum())
        if npx < min_pix:
            continue
        y0, y1, x0, x1 = s[0].start, s[0].stop, s[1].start, s[1].stop
        w, h = x1 - x0, y1 - y0
        c = dict(x0=int(x0), y0=int(y0), x1=int(x1), y1=int(y1),
                 w=int(w), h=int(h), npx=npx)
        if w > max_side or h > max_side or npx / (w * h) < min_fill:
            dropped.append(c)
        else:
            comps.append(c)
            tmask[s] |= sub
    return comps, dropped, tmask


def _rlsa(mask, gx, gy):
    """분리형 팽창. 2D 커널을 그대로 쓰면 고해상도에서 느리다."""
    d = binary_dilation(mask, np.ones((1, 2 * gx + 1)))
    return binary_dilation(d, np.ones((2 * gy + 1, 1)))


def _box_of(cs):
    return (min(c['x0'] for c in cs), min(c['y0'] for c in cs),
            max(c['x1'] for c in cs), max(c['y1'] for c in cs))


def _split_tall(blks, ratio=1.6, rounds=3):
    """
    RLSA 가 두 줄을 붙여버린 블록을 y 방향으로 다시 가른다.

    비스듬히 쓴 줄이나 줄 간격이 좁은 곳에서 흔하다. 성분의 y 중심 히스토그램에서
    가장 깊은 골짜기를 찾아 자르고, 성분은 중심으로 배정한다(획이 갈리지 않는다).
    """
    for _ in range(rounds):
        hs = [b['box'][3] - b['box'][1] for b in blks]
        if not hs:
            return blks
        med = float(np.median(hs))
        out, changed = [], False
        for b in blks:
            h = b['box'][3] - b['box'][1]
            if h <= ratio * med or len(b['comps']) < 4:
                out.append(b)
                continue
            y0, y1 = b['box'][1], b['box'][3]
            cen = np.array([(c['y0'] + c['y1']) / 2 for c in b['comps']])
            hist, edges = np.histogram(cen, bins=max(6, int(h / 12)),
                                       range=(y0, y1))
            hist = gaussian_filter(hist.astype(float), 1.2)
            lo, hi = int(len(hist) * 0.25), int(len(hist) * 0.75)
            if hi <= lo:
                out.append(b)
                continue
            k = lo + int(np.argmin(hist[lo:hi]))
            cut = (edges[k] + edges[k + 1]) / 2
            a = [c for c in b['comps'] if (c['y0'] + c['y1']) / 2 < cut]
            z = [c for c in b['comps'] if (c['y0'] + c['y1']) / 2 >= cut]
            if len(a) < 2 or len(z) < 2:
                out.append(b)
                continue
            out.append(dict(box=_box_of(a), comps=a))
            out.append(dict(box=_box_of(z), comps=z))
            changed = True
        blks = out
        if not changed:
            break
    return blks


def blocks(comps, tmask, gap_x=88, gap_y=12, scale=4, min_comps=2,
           split_tall=True):
    """
    글자 성분을 줄 단위로 묶는다. gap_x 는 같은 줄로 볼 최대 가로 간격(px).
    레이아웃은 1/scale 해상도로 계산한다 (원해상도는 느리고 불필요).
    returns [dict(box=(x0,y0,x1,y1), comps=[...]), ...] 위->아래, 좌->우
    """
    gx, gy = max(1, gap_x // scale), max(1, gap_y // scale)
    small = tmask[::scale, ::scale]
    lab, _ = label(_rlsa(small, gx, gy))
    out = []
    for s in find_objects(lab):
        if s is None:
            continue
        y0, y1 = (s[0].start + gy) * scale, (s[0].stop - gy) * scale
        x0, x1 = (s[1].start + gx) * scale, (s[1].stop - gx) * scale
        inside = [c for c in comps
                  if c['x0'] >= x0 - scale * 2 and c['x1'] <= x1 + scale * 2
                  and c['y0'] >= y0 - scale * 2 and c['y1'] <= y1 + scale * 2]
        if len(inside) < min_comps:
            continue
        out.append(dict(box=_box_of(inside), comps=inside))
    if split_tall:
        out = _split_tall(out)
    for b in out:
        b['comps'].sort(key=lambda c: c['x0'])
    out.sort(key=lambda b: (b['box'][1], b['box'][0]))
    return out


def overlay(ink, blks, path, dropped=None, scale=3):
    """블록 번호 + 성분 상자를 그린 확인용 이미지. 사용자가 이걸 보고 전사한다."""
    im = Image.fromarray(np.uint8(255 - np.clip(ink, 0, 255))).convert('RGB')
    dr = ImageDraw.Draw(im)
    for c in (dropped or []):
        dr.rectangle([c['x0'], c['y0'], c['x1'], c['y1']],
                     outline=(160, 160, 160), width=3)
    for i, b in enumerate(blks):
        x0, y0, x1, y1 = b['box']
        dr.rectangle([x0 - 6, y0 - 6, x1 + 6, y1 + 6], outline=(220, 0, 0), width=5)
        dr.text((x0 - 4, max(0, y0 - 46)), str(i), fill=(220, 0, 0))
        for c in b['comps']:
            dr.rectangle([c['x0'], c['y0'], c['x1'], c['y1']],
                         outline=(0, 120, 220), width=2)
    im.resize((im.width // scale, im.height // scale), Image.LANCZOS).save(path)
    return path


def analyze(path, bg_sigma=80, thresh=0.30, gap_x=88, gap_y=12, **kw):
    """
    사진 -> dict(ink, mask, blocks, comps, dropped). 한 번에 도는 편의 함수.
    mask 는 도형을 제거한 **텍스트만의** 마스크다. 글자 크롭은 이것을 써야
    옆을 지나는 화살표가 딸려 들어오지 않는다.
    """
    ink, raw = ink_map(path, bg_sigma, thresh)
    comps, dropped, tmask = text_components(raw, **kw)
    return dict(ink=ink, mask=tmask, comps=comps, dropped=dropped,
                blocks=blocks(comps, tmask, gap_x, gap_y))
