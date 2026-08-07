"""
필기자 단위 피팅.

한 사람이 쓴 숫자 여러 개 -> 공통 스타일 파라미터 + 글자별 이체
-> 그 사람처럼, 매번 조금씩 다르게 생성.

Q2 에서 드러난 문제를 푼다: 한 글자만 피팅하면 전역 파라미터가 그 글자의
골격 우연을 흡수해버려, 다른 글자로 전이할 때 의미가 깨진다.
여러 글자를 동시에 맞추면 그런 조합은 다른 글자에서 벌점을 받아 살아남지 못한다.
"""
import numpy as np
from scipy.optimize import minimize

from .simulator import STYLE_RANGES, NEUTRAL_STYLE, ALLOGRAPHS, render_with
from .loop import blur_l2

KEYS = list(STYLE_RANGES)
FIT_SEED = 0          # 피팅 중 렌더 노이즈 고정 -> 목적함수 결정적


def decode(z):
    return {k: STYLE_RANGES[k][0] + np.clip(v, 0, 1) *
               (STYLE_RANGES[k][1] - STYLE_RANGES[k][0])
            for k, v in zip(KEYS, z)}


def encode(p):
    return np.array([(p[k] - STYLE_RANGES[k][0]) /
                     (STYLE_RANGES[k][1] - STYLE_RANGES[k][0]) for k in KEYS])


def jitter_style(params, rng, amount=0.06):
    """
    같은 사람도 매번 똑같이 쓰지는 않는다. 파라미터 범위의 amount 비율만큼 흔든다.
    duplicated-signature 문헌이 쓰는 방식(피팅된 파라미터 주변 샘플링)과 같다.
    """
    out = {}
    for k, v in params.items():
        lo, hi = STYLE_RANGES[k]
        out[k] = float(np.clip(v + rng.normal(0, (hi - lo) * amount), lo, hi))
    return out


# ---------------------------------------------------------------------------
def _best_allo(target, ch, params, seed=FIT_SEED):
    """주어진 스타일에서 그 글자에 가장 맞는 이체와 거리."""
    best = (None, 1e9)
    for a in range(len(ALLOGRAPHS[ch])):
        d = blur_l2(target, render_with(ch, a, params,
                                        np.random.default_rng(seed)))
        if d < best[1]:
            best = (a, d)
    return best


def fit_writer(samples, n_random=200, refine=True, maxiter=400, seed=0,
               verbose=True):
    """
    samples: [(img28, '4'), ...]  한 사람이 쓴 글자들
    returns dict(params, allos, dist, per_glyph)

    목적함수: 각 글자마다 '그 스타일에서 최적 이체'를 고른 뒤 거리 합.
    이체는 이산 변수라 최적화 대신 매 평가에서 argmin 으로 처리한다.
    """
    rng = np.random.default_rng(seed)

    def obj(z):
        p = decode(z)
        return float(np.mean([_best_allo(t, ch, p)[1] for t, ch in samples]))

    Z = rng.random((n_random, len(KEYS)))
    Z[0] = encode(NEUTRAL_STYLE)
    sc = np.array([obj(z) for z in Z])
    i = int(np.argmin(sc))
    best_z, best_d = Z[i], float(sc[i])
    if verbose:
        print(f'  랜덤 {n_random}회 최선 {best_d:.4f}')

    if refine:
        r = minimize(obj, best_z, method='Powell', bounds=[(0, 1)]*len(KEYS),
                     options=dict(maxiter=maxiter, xtol=2e-3, ftol=2e-3))
        if r.fun < best_d:
            best_z, best_d = r.x, float(r.fun)
        if verbose:
            print(f'  국소 정제 후   {best_d:.4f}')

    params = decode(best_z)
    per = [(ch,) + _best_allo(t, ch, params) for t, ch in samples]
    return dict(params=params, dist=best_d,
                allos={ch: a for ch, a, _ in per},
                per_glyph=[(ch, a, round(d, 4)) for ch, a, d in per])


def generate(params, digits, n_each=1, jitter=0.06, allos=None, seed=0):
    """
    피팅된 필체로 숫자 생성. 매번 파라미터를 조금씩 흔들어 자연스러운 변동을 준다.
    allos: {글자: 이체} 를 주면 그 사람이 쓰는 형태로 고정. 없으면 랜덤.
    returns [(글자, img), ...]
    """
    rng = np.random.default_rng(seed)
    out = []
    for d in digits:
        for _ in range(n_each):
            p = jitter_style(params, rng, jitter)
            if allos and d in allos:
                a = allos[d]
            else:
                w = np.array([e[1] for e in ALLOGRAPHS[d]], float)
                a = int(rng.choice(len(w), p=w/w.sum()))
            out.append((d, render_with(d, a, p,
                                       np.random.default_rng(int(rng.integers(1e9))))))
    return out
