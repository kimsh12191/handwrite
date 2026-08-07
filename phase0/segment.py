"""
블록 -> 낱글자 -> 자모.

두 가지 원칙으로 짰다.

1. **글자 수를 라벨에서 안다.** 전사를 받으므로 "이 블록은 7글자" 라는 제약이
   있다. 무제약 분할보다 훨씬 안정적이다. 균등 분할에서 시작해 투영 골짜기로
   붙인다.

2. **절단선이 연결성분을 가르지 않는다.** 자모는 획의 집합이므로, 획 하나가
   두 자모로 쪼개지는 일은 없어야 한다. 그래서 절단 위치는 투영으로 정하되
   **성분 배정은 무게중심으로** 한다. 절단선이 획을 지나가도 그 획은 통째로
   한쪽에 간다.
"""
import numpy as np
from scipy.ndimage import label, find_objects, gaussian_filter

from . import hangul


# ---------------------------------------------------------------------------
# 블록 -> 낱글자
# ---------------------------------------------------------------------------
def _profile(mask, axis, sigma=None, scale=0.02):
    """
    투영 프로파일. **평활 폭이 중요하다.**

    처음에 sigma 를 프로파일 길이에 비례시켰더니(폭 1400px 이면 sigma 28)
    글자 사이 골짜기가 통째로 뭉개져 절단 위치가 무의미해졌다. 평활은
    **찾으려는 것의 크기**(글자 폭)에 비례해야지 전체 길이에 비례하면 안 된다.
    """
    p = mask.sum(axis).astype(float)
    if sigma is None:
        sigma = max(1.0, len(p) * scale)
    return gaussian_filter(p, max(0.6, sigma))


def _snap_cuts(prof, n_parts, window=0.35):
    """
    균등 분할점을 투영 골짜기로 끌어당긴다.
    window: 탐색 반경 (조각 폭 대비 비율)
    returns (cuts, costs)  costs = 절단 지점의 투영값 (낮을수록 깨끗한 절단)
    """
    L = len(prof)
    step = L / n_parts
    w = max(2, int(step * window))
    cuts, costs, prev = [], [], 0
    for k in range(1, n_parts):
        c = int(round(k * step))
        lo, hi = max(prev + 1, c - w), min(L - 1, c + w)
        if hi <= lo:
            pos = c
        else:
            pos = lo + int(np.argmin(prof[lo:hi]))
        cuts.append(pos)
        costs.append(float(prof[pos]))
        prev = pos
    return cuts, costs


def _assign(comps, cuts, key):
    """성분을 무게중심 기준으로 조각에 배정. 절단선이 획을 가르지 않게 한다."""
    groups = [[] for _ in range(len(cuts) + 1)]
    for c in comps:
        mid = (c[key + '0'] + c[key + '1']) / 2
        g = int(np.searchsorted(cuts, mid))
        groups[g].append(c)
    return groups


def _split_words(comps, n_words):
    """
    성분을 가로 간격이 큰 곳에서 끊어 어절 n_words 개로. 전사의 띄어쓰기를
    제약으로 쓴다 — 어절 경계는 글자 경계보다 훨씬 뚜렷하므로 먼저 잡는 편이
    안정적이다. 개수를 못 맞추면 None (호출부가 균등 분할로 폴백).
    """
    if n_words <= 1 or len(comps) < n_words:
        return None
    cs = sorted(comps, key=lambda c: c['x0'])
    gaps = [(cs[i + 1]['x0'] - cs[i]['x1'], i) for i in range(len(cs) - 1)]
    gaps.sort(reverse=True)
    cutpts = sorted(i for _, i in gaps[:n_words - 1])
    groups, prev = [], 0
    for i in cutpts:
        groups.append(cs[prev:i + 1])
        prev = i + 1
    groups.append(cs[prev:])
    return groups if all(groups) else None


def split_block(block, mask, text):
    """
    블록을 전사 text 의 글자 수만큼 자른다.

    2단계로 나눈다.
      1) 전사의 **띄어쓰기**로 어절을 먼저 가른다 (간격이 크므로 안정적)
      2) 어절 안에서 글자 수만큼 투영 절단

    1단계를 건너뛰면 균등 분할이 띄어쓰기를 글자 폭으로 착각해 뒤가 전부
    밀린다. 실측에서 인스턴스 오염의 주원인이었다.

    returns [dict(ch, box, cut_cost), ...]  — ch 가 한글 음절이 아니면 None
    """
    words = [w for w in text.split() if w]
    chars = [ch for w in words for ch in w]
    if not chars:
        return []
    x0, y0, x1, y1 = block['box']
    sub = mask[y0:y1, x0:x1]
    if sub.size == 0:
        return []

    rel = [dict(x0=c['x0'] - x0, x1=c['x1'] - x0,
                y0=c['y0'] - y0, y1=c['y1'] - y0) for c in block['comps']]
    wgroups = _split_words(rel, len(words))
    if wgroups is None:
        wgroups, words = [rel], [''.join(words)]

    out = []
    for wcomps, w in zip(wgroups, words):
        n = len(w)
        wx0 = min(c['x0'] for c in wcomps)
        wx1 = max(c['x1'] for c in wcomps)
        span = max(1, wx1 - wx0)
        if n <= 1:
            groups, costs = [wcomps], [0.0]
        else:
            # 평활 폭은 **글자 폭**에 비례해야 한다 (블록 폭이 아니라)
            prof = _profile(sub[:, wx0:wx1], 0, sigma=max(0.8, span / n * 0.10))
            cuts, costs = _snap_cuts(prof, n)
            cuts = [c + wx0 for c in cuts]
            groups = _assign(wcomps, cuts, 'x')

        for i, ch in enumerate(w):
            g = groups[i] if i < len(groups) else []
            if not g:
                out.append(dict(ch=None, box=None, cut_cost=None,
                                reason='빈 조각'))
                continue
            bx = (x0 + min(c['x0'] for c in g), y0 + min(c['y0'] for c in g),
                  x0 + max(c['x1'] for c in g), y0 + max(c['y1'] for c in g))
            cost = max([costs[i - 1] if 0 < i <= len(costs) else 0.0,
                        costs[i] if i < len(costs) else 0.0])
            out.append(dict(ch=ch if hangul.is_syllable(ch) else None,
                            box=bx, cut_cost=float(cost),
                            reason=None if hangul.is_syllable(ch) else '한글 아님'))
    return out


# ---------------------------------------------------------------------------
# 낱글자 -> 자모
# ---------------------------------------------------------------------------
# 배치별 기대 비율. (초성|중성) 세로 절단, (윗부분|종성) 가로 절단 위치.
CHO_JUNG_X = 0.55       # 가로모임에서 초성이 차지하는 폭 비율
CHO_JUNG_Y = 0.50       # 세로모임에서 초성이 차지하는 높이 비율 (종성 없을 때)
CHO_JUNG_Y_J = 0.38     # 세로모임, 종성 있을 때
JONG_Y = 0.62           # 종성이 시작하는 높이 비율


def _cut_at(prof, frac, window=0.18):
    L = len(prof)
    c = int(round(L * frac))
    w = max(2, int(L * window))
    lo, hi = max(1, c - w), min(L - 1, c + w)
    if hi <= lo:
        return c
    return lo + int(np.argmin(prof[lo:hi]))


def _components(mask, min_pix=8):
    lab, _ = label(mask)
    out = []
    for i, s in enumerate(find_objects(lab), 1):
        if s is None:
            continue
        npx = int((lab[s] == i).sum())
        if npx < min_pix:
            continue
        out.append(dict(y0=s[0].start, y1=s[0].stop,
                        x0=s[1].start, x1=s[1].stop, npx=npx))
    return out


def _ink_bbox(mask, y0, y1, x0, x1):
    """영역 안의 잉크에 딱 맞는 bbox. 잉크가 없으면 None."""
    sub = mask[y0:y1, x0:x1]
    if sub.size == 0 or not sub.any():
        return None
    ys, xs = np.nonzero(sub)
    return (y0 + int(ys.min()), y0 + int(ys.max()) + 1,
            x0 + int(xs.min()), x0 + int(xs.max()) + 1)


def _by_comps(comps, axis, cut, key_lo, key_hi):
    """성분을 절단선 기준 두 무리로. 획이 갈리지 않는다."""
    a = [c for c in comps if (c[key_lo] + c[key_hi]) / 2 < cut]
    b = [c for c in comps if (c[key_lo] + c[key_hi]) / 2 >= cut]
    return a, b


def _bbox_of(cs):
    return (min(c['y0'] for c in cs), max(c['y1'] for c in cs),
            min(c['x0'] for c in cs), max(c['x1'] for c in cs))


def split_glyph(glyph_mask, ch):
    """
    낱글자 이진 마스크 -> {역할: (y0,y1,x0,x1)}, 배치유형, 경고 태그 목록.

    두 경로가 있다.
      * 성분 배정 (선호) — 절단선이 획을 가르지 않는다. 자모가 서로 떨어져
        쓰였을 때 정확하다.
      * 영역 절단 (대비) — 자모가 붙어 쓰여 성분 수가 모자랄 때. 획이 갈릴 수
        있으므로 'cut' 태그를 붙인다. 참조 사진에서 이 경우가 흔했다.
    """
    parts, lay = hangul.parts(ch)
    H, W = glyph_mask.shape
    if H < 4 or W < 4 or not glyph_mask.any():
        return {}, lay, ['잉크 없음']

    comps = _components(glyph_mask)
    has_jong = any(r == 'jong' for _, r in parts)
    warn = []
    if len(comps) < len(parts):
        warn.append('붙여씀')

    out = {}
    # -- 1단계: 종성 분리 (가로 절단) ---------------------------------------
    top_y0, top_y1 = 0, H
    if has_jong:
        yc = _cut_at(_profile(glyph_mask, 1),
                     0.68 if lay == hangul.LAYOUT_H else JONG_Y)
        a, b = _by_comps(comps, 0, yc, 'y0', 'y1')
        if a and b:
            out['jong'] = _bbox_of(b)
            top_y0, top_y1 = 0, max(c['y1'] for c in a)
            comps = a
        else:
            bb = _ink_bbox(glyph_mask, int(yc), H, 0, W)
            if bb is None:
                warn.append('종성실패')
                has_jong = False
            else:
                out['jong'] = bb
                top_y0, top_y1 = 0, int(yc)
                warn.append('cut')
                comps = [c for c in comps if (c['y0'] + c['y1']) / 2 < yc]

    top = glyph_mask[top_y0:top_y1]
    if top.size == 0 or not top.any():
        warn.append('초중성없음')
        return out, lay, warn
    rel = [dict(y0=c['y0'] - top_y0, y1=c['y1'] - top_y0,
                x0=c['x0'], x1=c['x1'], npx=c['npx'])
           for c in comps if c['y0'] >= top_y0]

    # -- 2단계: 초성 | 중성 --------------------------------------------------
    if lay == hangul.LAYOUT_V:
        cut = _cut_at(_profile(top, 0), CHO_JUNG_X)
        a, b = _by_comps(rel, 1, cut, 'x0', 'x1')
        ra = (0, top.shape[0], 0, int(cut))
        rb = (0, top.shape[0], int(cut), W)
    elif lay == hangul.LAYOUT_H:
        cut = _cut_at(_profile(top, 1), CHO_JUNG_Y_J if has_jong else CHO_JUNG_Y)
        a, b = _by_comps(rel, 0, cut, 'y0', 'y1')
        ra = (0, int(cut), 0, W)
        rb = (int(cut), top.shape[0], 0, W)
    else:                                     # 섞임 — 근사
        warn.append('섞임근사')
        cut = int(W * 0.45)
        a = [c for c in rel if (c['x0'] + c['x1']) / 2 < cut
             and (c['y0'] + c['y1']) / 2 < top.shape[0] * 0.55]
        b = [c for c in rel if c not in a]
        ra = (0, int(top.shape[0] * 0.55), 0, cut)
        rb = (0, top.shape[0], cut, W)

    if a and b:
        ba, bb = _bbox_of(a), _bbox_of(b)
    else:
        ba, bb = _ink_bbox(top, *ra), _ink_bbox(top, *rb)
        if ba is None or bb is None:
            warn.append('초중성실패')
            return out, lay, warn
        warn.append('cut')

    out['cho'] = (ba[0] + top_y0, ba[1] + top_y0, ba[2], ba[3])
    out['jung'] = (bb[0] + top_y0, bb[1] + top_y0, bb[2], bb[3])
    return out, lay, warn


def glyph_jamo(glyph_mask, ch):
    """
    낱글자 -> ([(자모, 역할, bbox), ...], 배치유형, 경고태그).
    겹받침은 낱자모로 펴지 않는다 (이미지 분할이 별도 문제).
    """
    boxes, lay, warn = split_glyph(glyph_mask, ch)
    parts, _ = hangul.parts(ch)
    return ([(j, role, boxes[role]) for j, role in parts if role in boxes],
            lay, warn)
