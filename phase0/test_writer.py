"""
필기자 피팅 검증 — 정답을 아는 상태에서.

가상의 사람 하나를 만들어(숨긴 파라미터) 숫자 몇 개를 쓰게 한 뒤,
그 샘플만 보고 파라미터를 되찾을 수 있는지 본다.
되찾은 파라미터로 '본 적 없는 숫자'를 생성해 원래 사람과 비교한다.
"""
import argparse
import numpy as np

from .simulator import STYLE_RANGES, ALLOGRAPHS, sample_style, render_with
from .writer import fit_writer, generate, jitter_style, KEYS
from .loop import _panel, blur_l2


def make_person(seed, shown='401', held='2795', jitter=0.05):
    """숨긴 파라미터로 사람 하나. shown=피팅에 줄 글자, held=검증용."""
    rng = np.random.default_rng(seed)
    true_p = sample_style(rng)
    true_allo = {d: int(rng.integers(len(ALLOGRAPHS[d]))) for d in '0123456789'}

    def write(ds):
        out = []
        for d in ds:
            p = jitter_style(true_p, rng, jitter)      # 같은 사람도 매번 조금 다름
            out.append((render_with(d, true_allo[d], p,
                                    np.random.default_rng(int(rng.integers(1e9)))), d))
        return out

    return true_p, true_allo, write(shown), write(held)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--person', type=int, default=11)
    ap.add_argument('--shown', default='4013')
    ap.add_argument('--held', default='2795')
    ap.add_argument('--random', type=int, default=200)
    a = ap.parse_args()

    true_p, true_allo, shown, held = make_person(a.person, a.shown, a.held)

    print(f'가상 필기자 #{a.person}')
    print(f'  피팅에 보여줄 글자: {a.shown}   검증용(안 보여줌): {a.held}\n')

    print('[피팅]')
    fit = fit_writer(shown, n_random=a.random)
    print(f'  글자별: {fit["per_glyph"]}\n')

    # 1) 파라미터 복원 정확도
    print(f'  {"파라미터":<14} {"정답":>9} {"복원":>9} {"오차(범위%)":>12}')
    print('  ' + '-'*48)
    errs = []
    for k in KEYS:
        lo, hi = STYLE_RANGES[k]
        e = abs(fit['params'][k] - true_p[k]) / (hi - lo) * 100
        errs.append(e)
        print(f'  {k:<14} {true_p[k]:>9.3f} {fit["params"][k]:>9.3f} {e:>11.1f}%')
    print(f'  평균 오차 {np.mean(errs):.1f}%  (랜덤 추측이면 ~33%)\n')

    # 2) 이체 복원
    hit = sum(1 for ch, aa, _ in fit['per_glyph'] if aa == true_allo[ch])
    print(f'  이체 복원: {hit}/{len(fit["per_glyph"])} '
          f'({[(ch, aa, true_allo[ch]) for ch, aa, _ in fit["per_glyph"]]})\n')

    # 3) 본 적 없는 글자 생성 -> 진짜 그 사람 것과 비교
    gen = generate(fit['params'], a.held, n_each=1, jitter=0.05,
                   allos=fit['allos'], seed=7)
    ds = [blur_l2(t, g) for (t, _), (_, g) in zip(held, gen)]
    print(f'  검증 글자 거리(복원 vs 정답): ' +
          '  '.join(f'{c}:{d:.3f}' for (_, c), d in zip(held, ds)) +
          f'   평균 {np.mean(ds):.3f}')

    # 4) 같은 파라미터로 여러 장 -> '조금씩 다르게' 나오는가
    var = generate(fit['params'], a.held[:1]*6, n_each=1, jitter=0.06,
                   allos=fit['allos'], seed=3)
    vd = [blur_l2(var[0][1], g) for _, g in var[1:]]
    print(f'  같은 글자 6장 사이 거리: 평균 {np.mean(vd):.3f} '
          f'(0 이면 전부 동일 = 변동 없음)\n')

    _panel([('진짜(피팅에 봄)', x) for x, _ in shown] +
           [('진짜(안 봄)', x) for x, _ in held] +
           [('복원 생성', g) for _, g in gen],
           '/tmp/q1/writer_fit.png', scale=6)
    _panel([('복원 생성 변동', g) for _, g in var], '/tmp/q1/writer_var.png', scale=6)
    print('  /tmp/q1/writer_fit.png  /tmp/q1/writer_var.png')


if __name__ == '__main__':
    main()
