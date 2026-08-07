"""
도달 가능성 측정.

Q1 에서 "LLM 이 조종을 못 하는 것"과 "파라미터 공간이 타겟을 표현 못 하는 것"은
전혀 다른 문제다. 후자면 어떤 LLM 을 붙여도 못 맞춘다.

수치 탐색으로 각 이체별 최소 도달 거리를 구해 하한을 잰다.
LLM 이 도달한 거리가 이 하한에 가까우면 조종은 잘 된 것이고,
하한 자체가 크면 파라미터 공간(또는 스켈레톤)이 부족한 것이다.
"""
import numpy as np
from scipy.optimize import minimize

from .simulator import STYLE_RANGES, NEUTRAL_STYLE, ALLOGRAPHS, render_with
from .loop import blur_l2

KEYS = list(STYLE_RANGES)


def _decode(z):
    p = {}
    for k, v in zip(KEYS, z):
        lo, hi = STYLE_RANGES[k]
        p[k] = lo + np.clip(v, 0, 1) * (hi - lo)
    return p


def _encode(params):
    z = []
    for k in KEYS:
        lo, hi = STYLE_RANGES[k]
        z.append((params.get(k, NEUTRAL_STYLE[k]) - lo) / (hi - lo))
    return np.array(z)


def reach(target, ch, allo, n_random=400, n_refine=2, seed=0, maxiter=900):
    """이체 하나에 대해 최소 도달 거리 탐색. returns (dist, params)"""
    rng = np.random.default_rng(seed)
    render_rng = lambda: np.random.default_rng(0)     # 렌더 노이즈 고정 -> 목적함수 결정적

    def obj(z):
        img = render_with(ch, allo, _decode(z), render_rng())
        return blur_l2(target, img)

    # 1) 랜덤 탐색으로 시작점 확보
    Z = rng.random((n_random, len(KEYS)))
    Z[0] = _encode(NEUTRAL_STYLE)
    scores = np.array([obj(z) for z in Z])
    order = np.argsort(scores)

    # 2) 상위 몇 개에서 국소 정제
    best_d, best_z = scores[order[0]], Z[order[0]]
    for i in order[:n_refine]:
        r = minimize(obj, Z[i], method='Powell',
                     bounds=[(0, 1)]*len(KEYS),
                     options=dict(maxiter=maxiter, xtol=1e-3, ftol=1e-3))
        if r.fun < best_d:
            best_d, best_z = float(r.fun), r.x
    return best_d, _decode(best_z)


def reach_all(target, ch, **kw):
    """모든 이체에 대해. returns [(allo, dist, params), ...] 거리 오름차순"""
    out = []
    for a in range(len(ALLOGRAPHS[ch])):
        d, p = reach(target, ch, a, **kw)
        out.append((a, d, p))
    return sorted(out, key=lambda t: t[1])
