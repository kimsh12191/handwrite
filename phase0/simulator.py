"""
Phase 0 시뮬레이터: 스켈레톤 x 스타일 -> 28x28 MNIST 형식 이미지.

plan.md 의 구조를 최소한으로 구현한 것. ΣΛ 속도 모델은 Phase 1 에서.
여기서는 '합성 데이터가 실제 손글씨로 전이되는가'만 측정하면 되므로
기하 + 렌더링에 집중한다.
"""
import numpy as np
from PIL import Image, ImageDraw

# ----------------------------------------------------------------------------
# Layer 0: 스켈레톤 (정규화 좌표, y 는 위쪽이 +, em-box 0..1)
#   각 글자 = 획 리스트. 획 = 제어점 리스트. 획 사이는 펜 들림.
#   획 순서/방향은 사람이 실제로 쓰는 순서를 따름.
# ----------------------------------------------------------------------------
SKELETONS = {
    '0': [[(.50,.98),(.17,.76),(.16,.26),(.50,.02),(.84,.26),(.83,.76),(.50,.98)]],
    '1': [[(.26,.74),(.52,.97),(.52,.02)]],
    '2': [[(.14,.76),(.30,.95),(.60,.97),(.79,.79),(.70,.56),(.14,.05),(.86,.06)]],
    '3': [[(.16,.82),(.42,.98),(.73,.89),(.67,.63),(.44,.55),(.73,.47),(.77,.18),
           (.45,.02),(.15,.12)]],
    '4': [[(.68,.97),(.11,.31),(.90,.31)],
          [(.68,.97),(.63,.02)]],
    '5': [[(.24,.96),(.20,.56),(.48,.64),(.75,.51),(.75,.21),(.44,.02),(.15,.11)],
          [(.24,.96),(.82,.95)]],
    '6': [[(.75,.94),(.43,.87),(.21,.56),(.17,.27),(.35,.03),(.63,.05),(.79,.27),
           (.67,.46),(.39,.49),(.20,.35)]],
    '7': [[(.11,.94),(.87,.95),(.36,.02)]],
    '8': [[(.62,.96),(.34,.90),(.30,.70),(.52,.56),(.75,.44),(.79,.20),(.50,.03),
           (.23,.14),(.26,.38),(.50,.56),(.66,.71),(.62,.96)]],
    '9': [[(.77,.71),(.61,.92),(.35,.88),(.25,.68),(.39,.51),(.65,.55),(.77,.71),
           (.72,.36),(.57,.08),(.33,.02)]],
}

# 자주 쓰이는 이체(異體) — 사람마다 다르게 쓰는 형태. 다양성의 큰 축.
VARIANTS = {
    '1': [[[(.52,.97),(.52,.02)]],                                  # 밋밋한 1
          [[(.26,.74),(.52,.97),(.52,.02)],[(.28,.02),(.76,.02)]]], # 밑변 있는 1
    '4': [[[(.68,.97),(.11,.31),(.90,.31)],[(.68,.97),(.63,.02)]],
          [[(.66,.97),(.14,.33),(.90,.33),(.62,.33),(.62,.02)]]],   # 이어 쓴 4
    '7': [[[(.11,.94),(.87,.95),(.36,.02)]],
          [[(.11,.94),(.87,.95),(.36,.02)],[(.24,.48),(.66,.52)]]], # 가로줄 7
    '9': [[[(.77,.71),(.61,.92),(.35,.88),(.25,.68),(.39,.51),(.65,.55),(.77,.71),
            (.72,.36),(.57,.08),(.33,.02)]],
          [[(.77,.71),(.61,.92),(.35,.88),(.25,.68),(.39,.51),(.65,.55),(.77,.71),
            (.75,.02)]]],                                            # 직선 꼬리 9
    '2': [[[(.14,.76),(.30,.95),(.60,.97),(.79,.79),(.70,.56),(.14,.05),(.86,.06)]],
          [[(.14,.76),(.32,.96),(.62,.95),(.78,.74),(.52,.44),(.16,.05),(.86,.10)]]],
}


# ----------------------------------------------------------------------------
# Layer 1: 스타일 파라미터
# ----------------------------------------------------------------------------
STYLE_RANGES = dict(
    slant_deg      = (-22.0, 28.0),   # 기울기
    aspect         = (0.62, 1.30),    # 가로/세로 비
    curvature      = (-0.075, 0.075), # 획 부풀림 (법선 방향)
    tremor_amp     = (0.0,   0.030),  # 손떨림 진폭
    tremor_freq    = (4.0,   16.0),
    overshoot      = (-0.03, 0.075),  # 획 끝 넘김/모자람
    corner_round   = (0.0,   1.0),    # 꺾임 둥글기
    stroke_w       = (0.090, 0.230),  # 획 굵기 (em 대비)
    width_var      = (0.0,   0.55),   # 굵기 변동 (필압 대용)
    ink_noise      = (0.0,   0.35),   # 잉크 끊김/농도 얼룩
    rotation_deg   = (-11.0, 11.0),   # 전체 회전
    ctrl_jitter    = (0.0,   0.035),  # 제어점 흔들림 = 글자 모양 개인차
    endpoint_gap   = (-0.04, 0.06),   # 획 시작/끝 잘림 (o 가 안 닫히는 등)
)


def sample_style(rng):
    """스타일 파라미터 한 벌을 샘플링. = 한 사람의 필체."""
    s = {k: rng.uniform(lo, hi) for k, (lo, hi) in STYLE_RANGES.items()}
    # 굵기와 크기는 상관이 있다 (크게 쓰면 굵게 쓰는 경향)
    s['stroke_w'] *= 0.75 + 0.5 * (s['aspect'] - 0.62) / 0.68
    return s


# ----------------------------------------------------------------------------
# 기하 처리
# ----------------------------------------------------------------------------
def _catmull_rom(P, per_seg=14):
    """제어점 -> 부드러운 곡선. corner_round=0 이면 각지게, 1 이면 둥글게."""
    P = np.asarray(P, float)
    if len(P) < 2:
        return P
    Q = np.vstack([P[0] + (P[0] - P[1]) * 0.5, P, P[-1] + (P[-1] - P[-2]) * 0.5])
    out = []
    for i in range(len(Q) - 3):
        p0, p1, p2, p3 = Q[i], Q[i+1], Q[i+2], Q[i+3]
        t = np.linspace(0, 1, per_seg, endpoint=False)[:, None]
        out.append(0.5 * ((2*p1) + (-p0+p2)*t + (2*p0-5*p1+4*p2-p3)*t**2
                          + (-p0+3*p1-3*p2+p3)*t**3))
    out.append(Q[-2][None, :])
    return np.vstack(out)


def _polyline(P, per_seg=14):
    """각진 보간 (corner_round=0 쪽)."""
    P = np.asarray(P, float)
    out = []
    for i in range(len(P) - 1):
        t = np.linspace(0, 1, per_seg, endpoint=False)[:, None]
        out.append(P[i] + (P[i+1] - P[i]) * t)
    out.append(P[-1][None, :])
    return np.vstack(out)


def build_trajectory(strokes, style, rng):
    """스켈레톤 획들 -> 스타일이 입혀진 (points, width) 리스트."""
    S = style
    out = []
    for pts in strokes:
        P = np.asarray(pts, float).copy()

        # 제어점 지터: 같은 '3' 도 사람마다 모양이 다름
        if S['ctrl_jitter'] > 0:
            P += rng.normal(0, S['ctrl_jitter'], P.shape)

        # 획 끝 잘림/넘침 (o 가 안 닫히거나 삐져나옴)
        g = S['endpoint_gap']
        if abs(g) > 1e-4 and len(P) > 2:
            d0 = P[1] - P[0]; d1 = P[-1] - P[-2]
            P[0]  = P[0]  + d0 / (np.linalg.norm(d0) + 1e-9) * g
            P[-1] = P[-1] - d1 / (np.linalg.norm(d1) + 1e-9) * g

        # 둥글기: 두 보간 방식을 섞는다
        A = _catmull_rom(P); B = _polyline(P)
        n = min(len(A), len(B))
        C = S['corner_round'] * A[:n] + (1 - S['corner_round']) * B[:n]
        x, y = C[:, 0].copy(), C[:, 1].copy()
        s = np.linspace(0, 1, len(x))

        # overshoot
        if abs(S['overshoot']) > 1e-4 and len(x) > 3:
            dx, dy = x[-1]-x[-2], y[-1]-y[-2]
            L = np.hypot(dx, dy) + 1e-9
            k = S['overshoot']
            ramp = np.clip((s - 0.9) / 0.1, 0, 1)
            x += dx/L * k * ramp; y += dy/L * k * ramp

        # curvature: 법선 방향 부풀림
        if abs(S['curvature']) > 1e-4:
            tx, ty = np.gradient(x), np.gradient(y)
            L = np.hypot(tx, ty) + 1e-9
            bulge = np.sin(np.pi * s) * S['curvature']
            x += -ty/L * bulge; y += tx/L * bulge

        # tremor
        if S['tremor_amp'] > 1e-4:
            ph = rng.uniform(0, 2*np.pi, 2)
            f = S['tremor_freq']
            x += S['tremor_amp'] * np.sin(2*np.pi*f*s + ph[0])
            y += S['tremor_amp'] * np.sin(2*np.pi*f*s*1.13 + ph[1])

        # 굵기 프로파일: 획 중간이 굵고 끝이 가늘어지는 경향 + 랜덤 변동
        base = S['stroke_w']
        prof = 1.0 + S['width_var'] * (np.sin(np.pi*s)**0.6 - 0.5)
        prof *= 1.0 + rng.normal(0, 0.06, len(s))
        w = np.clip(base * prof, base*0.35, base*1.9)

        out.append((np.stack([x, y], 1), w))

    # 전역 변환: aspect -> slant -> rotation
    allp = np.vstack([p for p, _ in out])
    cx, cy = allp[:, 0].mean(), allp[:, 1].mean()
    th = np.deg2rad(S['rotation_deg']); ct, st = np.cos(th), np.sin(th)
    tanv = np.tan(np.deg2rad(S['slant_deg']))
    res = []
    for P, w in out:
        X = (P[:, 0] - cx) * S['aspect']
        Y = (P[:, 1] - cy)
        X = X + Y * tanv                      # slant
        Xr = X*ct - Y*st; Yr = X*st + Y*ct    # rotation
        res.append((np.stack([Xr, Yr], 1), w))
    return res


# ----------------------------------------------------------------------------
# 렌더링
# ----------------------------------------------------------------------------
def rasterize(traj, style, rng, canvas=112, margin=0.10):
    """(points,width) 리스트 -> 고해상도 grayscale (잉크=밝음).

    획은 반드시 연속이어야 한다: 선분 + 조인트 원반을 함께 그린다.
    잉크 얼룩/끊김은 스탬프 단위가 아니라 호 길이 구간 단위로 후처리한다
    (스탬프 단위로 하면 획이 구슬처럼 끊어져 보인다).
    """
    allp = np.vstack([p for p, _ in traj])
    lo, hi = allp.min(0), allp.max(0)
    span = max((hi - lo).max(), 1e-6)
    sc = canvas * (1 - 2*margin) / span
    off = np.array([canvas/2, canvas/2]) - (lo + hi)/2 * sc

    img = Image.new('L', (canvas, canvas), 0)
    d = ImageDraw.Draw(img)

    for P, w in traj:
        Q = P * sc + off
        Q[:, 1] = canvas - Q[:, 1]              # y 축 뒤집기 (이미지 좌표)
        W = np.maximum(w * sc, 1.2)

        # 획을 따라 끊길 구간을 미리 정한다 (펜이 잠깐 뜬 효과)
        gaps = []
        if style['ink_noise'] > 0.12 and len(Q) > 30:   # 짧은 획은 통째로 사라짐
            nseg = rng.poisson(style['ink_noise'] * 1.2)
            for _ in range(int(nseg)):
                a = rng.integers(0, max(1, len(Q)-1))
                gaps.append((a, a + rng.integers(2, 5)))

        def skipped(i):
            return any(a <= i <= b for a, b in gaps)

        for i in range(len(Q) - 1):
            if skipped(i):
                continue
            r = W[i] / 2
            (x0, y0), (x1, y1) = Q[i], Q[i+1]
            d.line([x0, y0, x1, y1], fill=255, width=max(1, int(round(W[i]))))
            d.ellipse([x0-r, y0-r, x0+r, y0+r], fill=255)   # 조인트/캡
        r = W[-1] / 2
        x0, y0 = Q[-1]
        d.ellipse([x0-r, y0-r, x0+r, y0+r], fill=255)

    # 잉크 농도 얼룩: 저주파 곱셈 필드 (획 형태를 깨지 않는다)
    if style['ink_noise'] > 0.02:
        k = 7
        f = rng.normal(0, 1, (k, k))
        f = np.array(Image.fromarray(
            ((f - f.min()) / (np.ptp(f) + 1e-9) * 255).astype(np.uint8)
        ).resize((canvas, canvas), Image.BICUBIC), float) / 255.0
        amp = style['ink_noise'] * 0.55
        a = np.array(img, float) * (1.0 - amp * (1.0 - f))
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return img


def to_mnist_format(img_hi, out=28, box=20):
    """
    MNIST 전처리를 그대로 재현:
      1) 잉크 바운딩박스로 크롭
      2) 종횡비 유지하며 box(20) 안에 맞춤, 안티에일리어싱
      3) 무게중심을 28x28 의 중앙에 오도록 배치
    이 단계를 빠뜨리면 프레이밍 차이가 도메인 갭으로 오측정된다.
    """
    a = np.array(img_hi)
    ys, xs = np.nonzero(a > 8)
    if len(xs) == 0:
        return Image.new('L', (out, out), 0)
    crop = img_hi.crop((xs.min(), ys.min(), xs.max()+1, ys.max()+1))

    w, h = crop.size
    if w >= h:
        nw, nh = box, max(1, int(round(h * box / w)))
    else:
        nh, nw = box, max(1, int(round(w * box / h)))
    crop = crop.resize((nw, nh), Image.LANCZOS)

    canvas = Image.new('L', (out, out), 0)
    canvas.paste(crop, ((out-nw)//2, (out-nh)//2))

    # 무게중심 정렬
    a = np.array(canvas, float)
    tot = a.sum()
    if tot > 0:
        gy, gx = np.mgrid[0:out, 0:out]
        cy = (a*gy).sum()/tot; cx = (a*gx).sum()/tot
        sx = int(round(out/2 - 0.5 - cx)); sy = int(round(out/2 - 0.5 - cy))
        if sx or sy:
            canvas = Image.fromarray(
                np.roll(np.roll(np.array(canvas), sy, 0), sx, 1))
    return canvas


def render_digit(digit, style, rng, use_variants=True):
    """숫자 하나 -> 28x28 uint8 배열 (잉크=밝음, MNIST 와 동일 규약)."""
    if use_variants and digit in VARIANTS and rng.random() < 0.45:
        strokes = VARIANTS[digit][rng.integers(len(VARIANTS[digit]))]
    else:
        strokes = SKELETONS[digit]
    traj = build_trajectory(strokes, style, rng)
    hi = rasterize(traj, style, rng)
    return np.array(to_mnist_format(hi), dtype=np.uint8)
