"""
Q1 하네스 — LLM(나) 이 시뮬레이터를 조종해 타겟 필체를 맞출 수 있는가.

자동 최적화 없음. 내가 이미지를 보고 파라미터를 제안하면 렌더해서 나란히 보여준다.
거리값은 수렴 추적용으로만 출력한다 (파라미터 선택은 눈으로).
"""
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from .simulator import ALLOGRAPHS, NEUTRAL_STYLE, render_with
from .data import load_mnist

_M = None
def mnist():
    global _M
    if _M is None:
        _M = load_mnist('test')
    return _M


def dist(a, b, sigma=1.2):
    """흐린 뒤 정규화 L2. 얇은 획에서 픽셀 L2 는 너무 예민해서."""
    fa = gaussian_filter(a.astype(float)/255., sigma)
    fb = gaussian_filter(b.astype(float)/255., sigma)
    fa /= np.linalg.norm(fa)+1e-9; fb /= np.linalg.norm(fb)+1e-9
    return float(np.linalg.norm(fa-fb))


def render(digit, allo, params, seed=0):
    st = dict(NEUTRAL_STYLE); st.update(params or {})
    return render_with(str(digit), allo, st, np.random.default_rng(seed))


def panel(items, path, scale=7, gap=6):
    """items: [(caption, img28), ...] 를 가로로 붙여 크게 저장"""
    n = len(items); c = 28
    W = n*(c+gap)+gap; H = c+gap*2+6
    im = Image.new('L', (W, H), 20)
    for i, (cap, a) in enumerate(items):
        im.paste(Image.fromarray(a), (gap+i*(c+gap), gap+6))
    im = im.resize((W*scale, H*scale), Image.NEAREST)
    dr = ImageDraw.Draw(im)
    for i, (cap, a) in enumerate(items):
        dr.text(((gap+i*(c+gap))*scale+4, 4), cap, fill=255)
    im.save(path); return path


def targets(idxs, path='/tmp/q1_targets.png', scale=7):
    X, Y = mnist()
    return panel([(f'#{i} ({Y[i]})', X[i]) for i in idxs], path, scale), \
           [int(Y[i]) for i in idxs]


def compare(idx, allo, params, path='/tmp/q1_cmp.png', seed=0, scale=7):
    """타겟 vs 렌더 나란히. 거리 반환."""
    X, Y = mnist()
    t = X[idx]; d = int(Y[idx])
    r = render(d, allo, params, seed)
    p = panel([(f'TARGET #{idx}', t), (f'allo{allo}', r)], path, scale)
    return dist(t, r), p


def variants(idx, cands, path='/tmp/q1_var.png', seed=0, scale=6):
    """
    후보 여러 개를 타겟 옆에 한 줄로. cands: [(label, allo, params), ...]
    반환: [(label, 거리), ...]
    """
    X, Y = mnist()
    t = X[idx]; d = int(Y[idx])
    items = [(f'TARGET', t)]; out = []
    for lab, allo, pr in cands:
        r = render(d, allo, pr, seed)
        items.append((lab, r))
        out.append((lab, dist(t, r)))
    panel(items, path, scale)
    return out, path


def all_allographs(idx, params=None, path='/tmp/q1_allo.png', seed=0, scale=6):
    """타겟 옆에 그 숫자의 모든 이체를 같은 파라미터로."""
    X, Y = mnist(); d = int(Y[idx])
    return variants(idx, [(f'a{k}', k, params)
                          for k in range(len(ALLOGRAPHS[str(d)]))],
                    path, seed, scale)
