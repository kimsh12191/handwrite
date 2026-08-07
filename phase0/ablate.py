"""
Phase 0 애블레이션: 이체(allograph) 다양성이 병목인가?

1차 실험 결과 SIM 81.40% < FONT 86.60%. 오류 분석 결과 시뮬레이터가
글자당 형태를 사실상 하나만 생성하고 있었다 (MNIST 4 의 42% 를 9 로 오인).
이체를 34개로 늘린 뒤 같은 조건에서 재측정한다.

  SIM        이체 확장 시뮬레이터 (n장)
  SIM+FONT   합성 n/2 + 폰트 n/2 (총량 동일) — 상보적인지 확인
"""
import argparse, json
import numpy as np

from .data import load_mnist, gen_synth, gen_font
from .run import train, evaluate

BASELINE = {'SIM(1차, 이체 5개)': 0.8140, 'FONT': 0.8660, 'REAL': 0.9947}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20000)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--out', default='phase0/ablation.json')
    a = ap.parse_args()

    Xte, Yte = load_mnist('test')
    res = {}

    print(f'[SIM] 이체 확장 합성 {a.n}장 생성...')
    Xs, Ys = gen_synth(a.n, seed=1, writers_per=32, progress=a.n//4)
    print('[SIM] 학습...')
    m = train(Xs, Ys, epochs=a.epochs)
    acc, pc, cm = evaluate(m, Xte, Yte)
    res['SIM'] = dict(acc=acc, per_class=pc.round(4).tolist())
    print(f'  -> {acc:.4f}\n')
    print('  클래스별: ' + '  '.join(f'{d}:{pc[d]*100:5.1f}' for d in range(10)) + '\n')

    print(f'[SIM+FONT] 합성 {a.n//2} + 폰트 {a.n//2} (총량 동일)...')
    Xf, Yf = gen_font(a.n//2, seed=2, progress=a.n//4)
    Xc = np.concatenate([Xs[:a.n//2], Xf]); Yc = np.concatenate([Ys[:a.n//2], Yf])
    print('[SIM+FONT] 학습...')
    m = train(Xc, Yc, epochs=a.epochs)
    acc2, pc2, _ = evaluate(m, Xte, Yte)
    res['SIM+FONT'] = dict(acc=acc2, per_class=pc2.round(4).tolist())
    print(f'  -> {acc2:.4f}\n')

    print('=' * 58)
    print('Phase 0 애블레이션 — MNIST 테스트셋 기준')
    print('=' * 58)
    for k, v in BASELINE.items():
        print(f'  {k:22s} {v*100:6.2f}%')
    print(f'  {"SIM(이체 34개)":22s} {res["SIM"]["acc"]*100:6.2f}%   '
          f'({(res["SIM"]["acc"]-0.8140)*100:+.2f}%p vs 1차)')
    print(f'  {"SIM+FONT":22s} {res["SIM+FONT"]["acc"]*100:6.2f}%')
    print(f'\n  SIM - FONT = {(res["SIM"]["acc"]-0.8660)*100:+.2f}%p')

    with open(a.out, 'w') as f:
        json.dump(res, f, indent=2)
    print(f'  저장: {a.out}')


if __name__ == '__main__':
    main()
