"""
매칭 루프 러너.

  [LLM] 파라미터 -> [PY] 시뮬 + 렌더 -> [VLM] 비교/방향 -> [LLM] 수정 -> 반복

모델 호출은 전부 ask(prompt, image_paths) -> str 하나로 추상화되어 있다.
지금은 내가(대화 중인 LLM) 그 자리를 대신한다. 내부망 397B 로 바꾸려면
ask 를 그 엔드포인트로 구현해 넘기면 된다. 나머지 코드는 그대로다.
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from .simulator import NEUTRAL_STYLE, render_with, ALLOGRAPHS
from .data import load_mnist
from . import contract

OUT = '/tmp/q1'


def blur_l2(a, b, sigma=1.2):
    """수렴 추적용 거리. 파라미터 선택을 이걸로 하지는 않는다."""
    fa = gaussian_filter(a.astype(float)/255., sigma)
    fb = gaussian_filter(b.astype(float)/255., sigma)
    fa /= np.linalg.norm(fa)+1e-9
    fb /= np.linalg.norm(fb)+1e-9
    return float(np.linalg.norm(fa-fb))


def _panel(items, path, scale=8, gap=6, cap_h=14):
    c = 28
    W = len(items)*(c+gap)+gap
    H = c+gap*2
    im = Image.new('L', (W, H), 20)
    for i, (_, a) in enumerate(items):
        im.paste(Image.fromarray(a), (gap+i*(c+gap), gap))
    im = im.resize((W*scale, H*scale), Image.NEAREST)
    big = Image.new('L', (im.width, im.height+cap_h*2), 20)
    big.paste(im, (0, cap_h*2))
    dr = ImageDraw.Draw(big)
    for i, (cap, _) in enumerate(items):
        dr.text(((gap+i*(c+gap))*scale+4, 6), cap, fill=255)
    big.save(path)
    return path


class MatchSession:
    """타겟 하나에 대한 매칭 세션."""

    def __init__(self, target, ch, name='t0', out=OUT, seed=0):
        self.target = np.asarray(target, np.uint8)
        self.ch = str(ch)
        self.name = name
        self.out = out
        self.seed = seed
        self.history = []          # [(allo, params, dist, cmp_path)]
        os.makedirs(out, exist_ok=True)
        self.target_path = _panel([('TARGET', self.target)],
                                  f'{out}/{name}_target.png')

    # -- 파이썬이 하는 일 ---------------------------------------------------
    def render(self, allo, params):
        st = dict(NEUTRAL_STYLE)
        st.update(params or {})
        return render_with(self.ch, allo, st, np.random.default_rng(self.seed))

    def step(self, allo, params):
        """파라미터 -> 렌더 -> 비교 이미지 저장. 거리 반환."""
        img = self.render(allo, params)
        d = blur_l2(self.target, img)
        k = len(self.history)
        p = _panel([('TARGET', self.target), (f'SIM r{k}', img)],
                   f'{self.out}/{self.name}_r{k}.png')
        self.history.append(dict(round=k, allo=allo, params=dict(params or {}),
                                 dist=d, path=p))
        return d, p

    def sweep_allographs(self, params=None):
        """이체 전체를 같은 파라미터로 나란히. 골격 선택용."""
        n = len(ALLOGRAPHS[self.ch])
        imgs = [(f'allo{k}', self.render(k, params)) for k in range(n)]
        p = _panel([('TARGET', self.target)] + imgs,
                   f'{self.out}/{self.name}_allo.png', scale=6)
        ds = [(k, blur_l2(self.target, im)) for k, (_, im) in enumerate(imgs)]
        return ds, p

    def best(self):
        return min(self.history, key=lambda h: h['dist']) if self.history else None

    def report(self):
        lines = [f'[{self.name}] 글자 {self.ch}  라운드 {len(self.history)}']
        for h in self.history:
            lines.append(f"  r{h['round']}  allo={h['allo']}  dist={h['dist']:.4f}"
                         f"  {json.dumps(h['params'], ensure_ascii=False)}")
        b = self.best()
        if b:
            lines.append(f"  best: r{b['round']} dist={b['dist']:.4f}")
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 자동 루프 — ask 를 397B 로 구현해 넘기면 그대로 돈다
# ---------------------------------------------------------------------------
def run_auto(session, ask, rounds=5, verbose=True):
    """
    ask(prompt: str, images: list[str]) -> str

    1) [VLM] 타겟 서술
    2) [LLM] 초기 파라미터
    3) 반복: [PY] 렌더 -> [VLM] 비교/방향 -> [LLM] 수정
    """
    ch = session.ch
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)

    desc = ask(contract.prompt_describe(ch), [session.target_path])
    log(f'[VLM 서술]\n{desc}\n')

    out = ask(contract.prompt_propose(ch, desc), [session.target_path])
    allo, params, warn = contract.parse(out, ch)
    if warn:
        log('[검증 경고]', '; '.join(warn))

    for r in range(rounds):
        d, path = session.step(allo, params)
        log(f'[r{r}] allo={allo} dist={d:.4f} params={params}')
        if r == rounds - 1:
            break
        crit = ask(contract.prompt_critique(ch), [path])
        log(f'[VLM 비평]\n{crit}\n')
        out = ask(contract.prompt_revise(ch, params, allo, crit), [path])
        allo, params, warn = contract.parse(out, ch)
        if warn:
            log('[검증 경고]', '; '.join(warn))

    return session


def mnist_session(idx, name=None, out=OUT):
    X, Y = load_mnist('test')
    return MatchSession(X[idx], int(Y[idx]), name or f'mnist{idx}', out)
