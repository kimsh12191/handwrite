"""
Phase 0 오류 분석.

정확도 숫자 하나로는 '무엇을 고쳐야 하는지' 를 알 수 없다.
시뮬레이터로 학습한 모델이 실제 MNIST 의 어떤 글자를 틀리는지 눈으로 본다.

  1) 혼동 행렬 상위 오류 쌍
  2) 가장 자신 있게 틀린 MNIST 샘플들 (= 시뮬레이터가 못 만드는 형태)
  3) 같은 클래스의 시뮬레이터 샘플과 나란히 비교
"""
import argparse
import numpy as np
import torch
from PIL import Image, ImageDraw

from .data import load_mnist, gen_synth
from .run import SmallCNN, train, _norm, DEV


@torch.no_grad()
def predict(model, X, bs=512):
    model.eval()
    out = []
    Xt = _norm(X)
    for i in range(0, len(Xt), bs):
        out.append(torch.softmax(model(Xt[i:i+bs].to(DEV)), 1).cpu())
    return torch.cat(out).numpy()


def sheet(rows, path, cell=28, pad=2, scale=3, labels=None):
    """rows: list of (title, [img,...])"""
    W = max(len(r[1]) for r in rows)*(cell+pad)+pad
    H = sum(cell+pad+16 for _ in rows)+pad
    im = Image.new('L', (W, H), 30); dr = ImageDraw.Draw(im)
    y = 0
    for title, imgs in rows:
        dr.text((3, y+3), title, fill=255); y += 16
        for c, a in enumerate(imgs):
            im.paste(Image.fromarray(a), (pad+c*(cell+pad), y+pad))
        y += cell+pad
    im.resize((W*scale, H*scale), Image.NEAREST).save(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20000)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--out', default='/tmp/phase0_errors.png')
    a = ap.parse_args()

    print('합성 데이터 생성...')
    Xs, Ys = gen_synth(a.n, seed=1, writers_per=32)
    print('학습...')
    model = train(Xs, Ys, epochs=a.epochs, log=lambda s: None)

    Xte, Yte = load_mnist('test')
    P = predict(model, Xte)
    pred = P.argmax(1); conf = P.max(1)
    wrong = pred != Yte
    print(f'\nMNIST 정확도 {1-wrong.mean():.4f}   오답 {wrong.sum()}장')

    # 1) 상위 혼동 쌍
    cm = np.zeros((10, 10), int)
    for t, q in zip(Yte, pred):
        cm[t, q] += 1
    pairs = [(cm[i, j], i, j) for i in range(10) for j in range(10) if i != j]
    pairs.sort(reverse=True)
    print('\n상위 혼동 쌍 (정답 -> 예측, 건수):')
    for n, i, j in pairs[:8]:
        print(f'  {i} -> {j} : {n}   (클래스 {i} 의 {n/max(cm[i].sum(),1)*100:.1f}%)')

    # 2) 클래스별 정확도
    percls = cm.diagonal()/np.maximum(cm.sum(1), 1)
    print('\n클래스별 정확도:')
    print('  ' + '  '.join(f'{d}:{percls[d]*100:5.1f}' for d in range(10)))

    # 3) 자신 있게 틀린 샘플 + 같은 클래스 시뮬 샘플 비교
    rows = []
    order = np.argsort(-percls)[::-1][:3]          # 최악 3개 클래스
    rng = np.random.default_rng(0)
    for d in order:
        idx = np.flatnonzero(wrong & (Yte == d))
        idx = idx[np.argsort(-conf[idx])][:12]     # 가장 확신하며 틀린 것
        rows.append((f'MNIST 실제={d} 오답 (예측: '
                     + ','.join(str(pred[i]) for i in idx[:12]) + ')',
                     [Xte[i] for i in idx]))
        sidx = rng.choice(np.flatnonzero(Ys == d), 12, replace=False)
        rows.append((f'SIM  {d} (학습에 쓰인 형태)', [Xs[i] for i in sidx]))
    print('\n저장:', sheet(rows, a.out))
    print('=> MNIST 오답 행과 SIM 행을 비교해 시뮬레이터가 못 만드는 형태를 찾는다.')


if __name__ == '__main__':
    main()
