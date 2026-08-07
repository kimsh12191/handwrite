"""
Phase 0 관문 실험.

질문: 합성 손글씨로 학습한 모델이 진짜 손글씨(MNIST)에서 작동하는가?
      그리고 시뮬레이터가 폰트 기반보다 나은가?

세 가지 학습 소스를 동일 조건(같은 수량/모델/에폭)에서 비교하고,
전부 MNIST 테스트셋(실제 사람 손글씨) 으로 평가한다.

  SIM   시뮬레이터 합성
  FONT  폰트 렌더 + 왜곡  (기준선)
  REAL  MNIST 학습셋      (상한선 참조용)
"""
import argparse, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import load_mnist, gen_synth, gen_font

torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


class SmallCNN(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
        )
        self.c = nn.Sequential(nn.Flatten(), nn.Linear(64*7*7, 128), nn.ReLU(),
                               nn.Dropout(0.4), nn.Linear(128, nc))

    def forward(self, x):
        return self.c(self.f(x))


def _norm(X):
    """uint8 (N,28,28) -> float tensor (N,1,28,28), MNIST 표준 정규화."""
    t = torch.from_numpy(X).float().div_(255.).unsqueeze(1)
    return t.sub_(0.1307).div_(0.3081)


def train(X, Y, epochs=12, bs=128, lr=2e-3, log=print):
    model = SmallCNN().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt, Yt = _norm(X), torch.from_numpy(Y)
    n = len(Xt)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * max(1, n // bs + 1))
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = cor = 0; loss_sum = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            xb, yb = Xt[idx].to(DEV), Yt[idx].to(DEV)
            opt.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, yb, label_smoothing=0.05)
            loss.backward(); opt.step(); sched.step()
            loss_sum += loss.item()*len(idx)
            cor += (out.argmax(1) == yb).sum().item(); tot += len(idx)
        log(f'    epoch {ep+1:2d}/{epochs}  loss {loss_sum/tot:.4f}  train_acc {cor/tot:.4f}')
    return model


@torch.no_grad()
def evaluate(model, X, Y, bs=512):
    model.eval()
    Xt, Yt = _norm(X), torch.from_numpy(Y)
    preds = []
    for i in range(0, len(Xt), bs):
        preds.append(model(Xt[i:i+bs].to(DEV)).argmax(1).cpu())
    p = torch.cat(preds)
    acc = (p == Yt).float().mean().item()
    cm = np.zeros((10, 10), int)
    for t, q in zip(Yt.numpy(), p.numpy()):
        cm[t, q] += 1
    per_class = cm.diagonal() / np.maximum(cm.sum(1), 1)
    return acc, per_class, cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20000, help='학습 데이터 수 (소스별 동일)')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--writers-per', type=int, default=32,
                    help='시뮬레이터: 한 스타일 벡터당 생성 장수')
    ap.add_argument('--out', default='phase0/results.json')
    a = ap.parse_args()

    print(f'device={DEV}  n={a.n}  epochs={a.epochs}\n')

    print('[데이터] MNIST 테스트셋 (평가용, 실제 사람 손글씨)')
    Xte, Yte = load_mnist('test')
    print(f'  {Xte.shape}\n')

    sources = {}

    t0 = time.time()
    print(f'[데이터] 시뮬레이터 합성 {a.n}장 생성...')
    sources['SIM'] = gen_synth(a.n, seed=1, writers_per=a.writers_per,
                               progress=max(1, a.n//4))
    print(f'  {time.time()-t0:.1f}s\n')

    t0 = time.time()
    print(f'[데이터] 폰트 기준선 {a.n}장 생성...')
    sources['FONT'] = gen_font(a.n, seed=1, progress=max(1, a.n//4))
    print(f'  {time.time()-t0:.1f}s\n')

    print(f'[데이터] MNIST 학습셋 {a.n}장 (상한선 참조)')
    Xtr, Ytr = load_mnist('train')
    sel = np.random.default_rng(0).choice(len(Xtr), a.n, replace=False)
    sources['REAL'] = (Xtr[sel], Ytr[sel])
    print()

    results = {}
    for name, (X, Y) in sources.items():
        print(f'=== {name} 학습 ===')
        t0 = time.time()
        model = train(X, Y, epochs=a.epochs)
        acc, per_class, cm = evaluate(model, Xte, Yte)
        results[name] = dict(mnist_acc=acc,
                             per_class=per_class.round(4).tolist(),
                             confusion=cm.tolist(),
                             train_sec=round(time.time()-t0, 1))
        print(f'  -> MNIST 정확도 {acc:.4f}   ({time.time()-t0:.0f}s)\n')

    print('=' * 62)
    print('Phase 0 결과 — 전부 실제 MNIST 테스트셋(10,000장) 기준')
    print('=' * 62)
    for k in ['SIM', 'FONT', 'REAL']:
        print(f'  {k:5s}  {results[k]["mnist_acc"]*100:6.2f}%')
    sim, font = results['SIM']['mnist_acc'], results['FONT']['mnist_acc']
    print(f'\n  시뮬레이터 - 폰트 = {(sim-font)*100:+.2f}%p')
    print(f'  실제 대비 시뮬레이터 달성률 = {sim/results["REAL"]["mnist_acc"]*100:.1f}%')

    print('\n  클래스별 정확도 (SIM):')
    pc = results['SIM']['per_class']
    print('   ' + '  '.join(f'{d}:{pc[d]*100:5.1f}' for d in range(10)))
    worst = int(np.argmin(pc))
    cm = np.array(results['SIM']['confusion'])
    conf = cm[worst].copy(); conf[worst] = 0
    print(f'   최악 클래스 {worst} ({pc[worst]*100:.1f}%) -> 주로 {int(np.argmax(conf))} 로 오인')

    with open(a.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  저장: {a.out}')


if __name__ == '__main__':
    main()
