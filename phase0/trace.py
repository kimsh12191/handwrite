"""
Q2 — 자모 이미지 -> 획 -> 원호열.  **계획자와 실행자를 여기서 가른다.**

    골격 추출 -> 획 분리 -> 획마다 원호 N개 근사
                              │
                              ├─ 호가 잡은 것  = 계획자 (목표점·곡률)
                              └─ 남은 잔차     = 실행자 (떨림·속도 흔들림)

호의 개수 N 이 노브다. 많이 쓰면 떨림까지 계획으로 흡수하므로, N 은
잔차와 함께 벌점(lam)으로 고른다. "몇 개의 호로 설명되는가" 가 곧 계획의
복잡도이고 측정 가능한 양이다.

속도는 다루지 않는다 — 정지 이미지에 속도가 없다 (`summary.md` §1.6).
여기서 얻는 것은 ΣΛ 의 action plan 에 해당하는 **기하**뿐이다.
"""
import numpy as np
from scipy.ndimage import convolve, label
from skimage.morphology import skeletonize

# 8-이웃 수를 세는 커널
_K = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)


# ---------------------------------------------------------------------------
# 골격 -> 획(폴리라인)
# ---------------------------------------------------------------------------
def skeleton(mask):
    return skeletonize(np.asarray(mask) > 0)


def _neighbors(sk):
    return convolve(sk.astype(np.uint8), _K, mode='constant')


def branches(sk, min_len=4):
    """
    골격을 분기점에서 끊어 가지(폴리라인) 목록으로.
    returns [ndarray(n,2) in (y,x)], 그리고 분기점 좌표 집합
    """
    nb = _neighbors(sk)
    junc = sk & (nb >= 3)
    seg = sk & ~junc
    lab, n = label(seg, structure=np.ones((3, 3)))
    out = []
    for i in range(1, n + 1):
        pts = np.argwhere(lab == i)
        if len(pts) < min_len:
            continue
        out.append(_order_path(pts))
    return out, np.argwhere(junc)


def _order_path(pts):
    """가지 픽셀들을 한쪽 끝에서 다른 끝까지 걸어 순서를 준다."""
    if len(pts) <= 2:
        return pts.astype(float)
    idx = {tuple(p): k for k, p in enumerate(map(tuple, pts))}
    adj = [[] for _ in pts]
    for k, (y, x) in enumerate(pts):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                j = idx.get((y + dy, x + dx))
                if j is not None:
                    adj[k].append(j)
    ends = [k for k in range(len(pts)) if len(adj[k]) <= 1]
    start = ends[0] if ends else 0
    order, seen, cur, prev = [start], {start}, start, -1
    while True:
        nxt = [j for j in adj[cur] if j not in seen]
        if not nxt:
            break
        # 직진을 우선 (분기 잔재가 있을 때 되돌아가지 않게)
        if prev >= 0 and len(nxt) > 1:
            v = pts[cur] - pts[prev]
            nxt.sort(key=lambda j: -float(np.dot(pts[j] - pts[cur], v)))
        prev, cur = cur, nxt[0]
        seen.add(cur)
        order.append(cur)
    return pts[order].astype(float)


def merge_smooth(paths, juncs, max_gap=3.0, max_turn_deg=45.0):
    """
    분기점에서 끊긴 가지들 중 **방향이 이어지는** 것을 다시 붙인다.
    사람이 한 획으로 그은 것이 교차 때문에 끊긴 경우를 되살린다.
    """
    if len(paths) < 2:
        return [p.copy() for p in paths]
    segs = [p.copy() for p in paths]
    cos_lim = np.cos(np.radians(max_turn_deg))

    def tang(p, at_end):
        k = min(5, len(p) - 1)
        v = (p[-1] - p[-1 - k]) if at_end else (p[0] - p[k])
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else np.zeros(2)

    merged = True
    while merged:
        merged = False
        for i in range(len(segs)):
            for j in range(len(segs)):
                if i == j or segs[i] is None or segs[j] is None:
                    continue
                a, b = segs[i], segs[j]
                if len(a) < 3 or len(b) < 3:
                    continue
                if np.linalg.norm(a[-1] - b[0]) > max_gap:
                    continue
                if float(np.dot(tang(a, True), tang(b, False))) < cos_lim:
                    continue
                segs[i] = np.vstack([a, b])
                segs[j] = None
                merged = True
                break
            if merged:
                break
        segs = [s for s in segs if s is not None]
    return segs


def prune_twigs(sk, min_len=5):
    """
    골격의 잔가지(털)를 쳐낸다. **없으면 획수를 셀 수 없다.**

    실측: 떨림을 0->8px 넣으면 가지치기 없이 획수가 1.0 -> 9.0 으로 튄다
    (정답은 1획). 떨림이 잔차가 아니라 '골격이 조각남' 으로 새기 때문이다.
    획수는 계획의 이산 구조 — Q4 가 제안하고 채점할 대상 — 이므로 이것이
    흔들리면 루프 전체가 성립하지 않는다. 가지치기 후 1.0~1.8 로 안정된다.

    분기점에 닿아 있으면서 min_len 보다 짧은 조각만 지운다. 고립된 짧은
    성분(점·마침표)은 진짜일 수 있으므로 건드리지 않는다.
    """
    sk = sk.astype(bool).copy()
    for _ in range(6):
        nb = _neighbors(sk)
        junc = sk & (nb >= 3)
        seg = sk & ~junc
        lab, n = label(seg, structure=np.ones((3, 3)))
        if n == 0:
            break
        drop = np.zeros_like(sk)
        changed = False
        for i in range(1, n + 1):
            comp = (lab == i)
            npx = int(comp.sum())
            if npx >= min_len:
                continue
            # 분기점에 닿아 있는 짧은 조각만 잔가지로 본다
            grown = convolve(comp.astype(np.uint8), _K, mode='constant') > 0
            if (grown & junc).any():
                drop |= comp
                changed = True
        if not changed:
            break
        sk = sk & ~drop
        sk = sk & (_neighbors(sk) > 0)      # 고립 픽셀 정리
    return sk


def strokes(mask, min_len=4, merge=True, prune_len=None):
    """
    자모 마스크 -> 획 폴리라인 목록.
    prune_len: 이보다 짧은 끝가지는 골격에서 제거. None 이면 펜 굵기에서 정한다.
    """
    mask = np.asarray(mask) > 0
    sk = skeleton(mask)
    if prune_len is None:
        w = mask.sum() / max(sk.sum(), 1.0)          # 평균 펜 굵기 (px)
        prune_len = max(3, int(round(1.6 * w)))
    sk = prune_twigs(sk, prune_len)
    br, juncs = branches(sk, min_len)
    return (merge_smooth(br, juncs) if merge else br), sk


# ---------------------------------------------------------------------------
# 폴리라인 -> 원호열
# ---------------------------------------------------------------------------
def arclen(path):
    if len(path) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def resample(path, n=40):
    """
    호길이 등간격으로 **고정 개수** 재표본. 이후 계산이 전부 이 위에서 돈다.
    개수를 고정해야 아래 DP 가 O(n^2) 로 묶인다 (px 단위면 획 길이에 따라 폭발).
    """
    P = np.asarray(path, float)
    if len(P) < 2:
        return P
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if d[-1] < 1e-9:
        return P[:1]
    t = np.linspace(0, d[-1], max(3, int(n)))
    return np.stack([np.interp(t, d, P[:, 0]), np.interp(t, d, P[:, 1])], 1)


def _cross2(a, b):
    """numpy 2.x 에서 2D np.cross 가 없어졌다."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def fit_arc(pts):
    """
    점열에 원호 하나. 대수적 원 적합(Kasa) 후 잔차.
    거의 직선이면 직선으로 떨어뜨린다 (반지름 발산 방지).
    returns dict(kind, ...), rms
    """
    P = np.asarray(pts, float)
    if len(P) < 2:
        return dict(kind='point', p=P[0] if len(P) else np.zeros(2)), 0.0
    p0, p1 = P[0], P[-1]
    chord = np.linalg.norm(p1 - p0)

    A = np.c_[2 * P, np.ones(len(P))]
    b = (P ** 2).sum(1)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        sol = None
    if sol is not None:
        c = sol[:2]
        r2 = sol[2] + c @ c
        r = np.sqrt(max(r2, 0.0))
    else:
        c, r = np.zeros(2), 0.0

    # 직선 잔차
    if chord > 1e-9:
        u = (p1 - p0) / chord
        res_line = np.abs(_cross2(np.broadcast_to(u, P.shape), P - p0))
        rms_line = float(np.sqrt((res_line ** 2).mean()))
    else:
        rms_line = float(np.sqrt((((P - p0) ** 2).sum(1)).mean()))

    if r < 1e-6 or r > 50 * max(chord, 1.0):
        return dict(kind='line', p0=p0, p1=p1), rms_line

    rms_arc = float(np.sqrt(((np.linalg.norm(P - c, axis=1) - r) ** 2).mean()))
    if rms_arc >= rms_line:
        return dict(kind='line', p0=p0, p1=p1), rms_line

    # 좌표는 (y, x). dy = r sin(th), dx = r cos(th) 로 두면 th = arctan2(dy, dx).
    th0 = float(np.arctan2(*(p0 - c)))
    th1 = float(np.arctan2(*(p1 - c)))
    thm = float(np.arctan2(*(P[len(P) // 2] - c)))
    # 중간점이 호 위에 오도록 진행 방향 결정
    def _un(a, b):
        d = b - a
        while d > np.pi:
            d -= 2 * np.pi
        while d < -np.pi:
            d += 2 * np.pi
        return d
    d1, dm = _un(th0, th1), _un(th0, thm)
    if d1 * dm < 0 or abs(dm) > abs(d1):
        d1 = d1 - np.sign(d1) * 2 * np.pi
    return dict(kind='arc', c=c, r=float(r), th0=float(th0),
                dth=float(d1)), rms_arc


def arc_points(a, n=32):
    """호/직선 -> 점열 (y,x). 렌더·비교용."""
    if a['kind'] == 'line':
        t = np.linspace(0, 1, n)[:, None]
        return a['p0'] * (1 - t) + a['p1'] * t
    if a['kind'] == 'point':
        return np.asarray(a['p'], float)[None, :]
    th = a['th0'] + np.linspace(0, a['dth'], n)
    return np.asarray(a['c'], float) + a['r'] * np.stack([np.sin(th),
                                                          np.cos(th)], 1)


def fit_arcs(path, lam=0.9, max_arcs=6, n_pts=36):
    """
    폴리라인 -> 원호열.  DP 로 분절점을 고른다.
    비용 = Σ (구간 rms^2 * 길이) + lam^2 * 총길이 * (호 개수)

    lam 이 곧 '떨림을 계획으로 흡수하지 않는 선' 이다. 크면 호가 적어지고
    잔차(=실행자)가 커진다.

    returns dict(arcs, rms, n_arcs, length, resid)
    """
    raw = np.asarray(path, float)
    L = arclen(raw)
    P = resample(raw, n_pts)
    m = len(P)
    if m < 3 or L < 1e-6:
        a, r = fit_arc(P)
        return dict(arcs=[a], rms=r, n_arcs=1, length=L,
                    resid=np.zeros(max(m, 1)), pts=P)

    pen = (lam ** 2) * m
    INF = float('inf')
    # cost[i][j] = 구간 [i, j] 를 호 하나로 덮는 비용
    cost = np.full((m, m), INF)
    fits = {}
    for i in range(m - 1):
        for j in range(i + 2, m):
            a, r = fit_arc(P[i:j + 1])
            cost[i, j] = (r ** 2) * (j - i)
            fits[(i, j)] = a

    best = np.full((m, max_arcs + 1), INF)
    back = np.full((m, max_arcs + 1), -1, int)
    best[0, 0] = 0.0
    for k in range(1, max_arcs + 1):
        for j in range(2, m):
            for i in range(0, j - 1):
                if best[i, k - 1] == INF or cost[i, j] == INF:
                    continue
                v = best[i, k - 1] + cost[i, j] + pen
                if v < best[j, k]:
                    best[j, k] = v
                    back[j, k] = i
    k = int(np.argmin(best[m - 1, 1:])) + 1
    if best[m - 1, k] == INF:
        a, r = fit_arc(P)
        return dict(arcs=[a], rms=r, n_arcs=1, length=L,
                    resid=np.zeros(m), pts=P)

    arcs, j = [], m - 1
    while k > 0 and j > 0:
        i = back[j, k]
        arcs.append(fits[(i, j)])
        j, k = i, k - 1
    arcs.reverse()

    resid = _residual(P, arcs)
    return dict(arcs=arcs, rms=float(np.sqrt((resid ** 2).mean())),
                n_arcs=len(arcs), length=L, resid=resid, pts=P)


def _residual(P, arcs):
    """각 점에서 가장 가까운 호까지의 거리 = 실행자 몫."""
    Q = np.vstack([arc_points(a, 48) for a in arcs])
    d = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2)
    return d.min(1)


# ---------------------------------------------------------------------------
# 자모 하나를 통째로
# ---------------------------------------------------------------------------
def trace_jamo(mask, lam=0.9, max_arcs=6, min_len=6, prune_len=None):
    """
    자모 마스크 -> dict.

      plan     : 획마다 원호열 (계획자)
      exec_    : 잔차·굵기 통계 (실행자)
      n_stroke : 획 개수 (계획의 이산 구조)
    """
    mask = np.asarray(mask) > 0
    if mask.sum() < 8:
        return None
    paths, sk = strokes(mask, min_len=min_len, prune_len=prune_len)
    paths = [p for p in paths if len(p) >= min_len]
    if not paths:
        return None

    scale = float(max(mask.shape))
    fitted = [fit_arcs(p, lam=lam, max_arcs=max_arcs) for p in paths]
    resid = np.concatenate([f['resid'] for f in fitted])
    total_len = sum(f['length'] for f in fitted)
    sk_len = float(sk.sum())

    return dict(
        plan=[dict(arcs=f['arcs'], n_arcs=f['n_arcs'], length=f['length'])
              for f in fitted],
        n_stroke=len(fitted),
        n_arcs=int(sum(f['n_arcs'] for f in fitted)),
        # -- 실행자 --------------------------------------------------------
        exec_=dict(
            resid_rms=float(np.sqrt((resid ** 2).mean()) / scale),
            resid_p95=float(np.percentile(resid, 95) / scale),
            # 굵기 = 잉크면적 / 골격길이. 피팅이 아니라 측정이다.
            stroke_w=float(mask.sum() / max(sk_len, 1.0) / scale),
            ink_ratio=float(mask.mean()),
        ),
        length=total_len / scale,
        scale=scale,
        paths=paths,
        fitted=fitted,
    )
