"""
가독성 판정기 (오라클).

실제 MNIST 로 학습한 분류기를 '이게 사람이 쓴 숫자 d 로 읽히는가' 판정에 쓴다.
파라미터를 극단으로 밀었을 때 글자가 깨지는 지점을 정량적으로 찾기 위한 도구.

주의: 오라클은 MNIST 분포를 기준으로 판정한다. 따라서 'MNIST 스러운가'를 재는
것이지 절대적 가독성이 아니다. 사람 눈 확인을 대체하지 않고 보조한다.
"""
import os
import numpy as np
import torch

from .run import SmallCNN, train, _norm, DEV
from .data import load_mnist

CKPT = os.path.join(os.path.dirname(__file__), '_cache', 'oracle.pt')


def get_oracle(epochs=12, force=False):
    """MNIST 학습 분류기를 반환 (캐시)."""
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    model = SmallCNN().to(DEV)
    if os.path.exists(CKPT) and not force:
        model.load_state_dict(torch.load(CKPT, map_location=DEV))
        model.eval()
        return model

    print('[oracle] MNIST 로 가독성 판정기 학습 (최초 1회)...')
    Xtr, Ytr = load_mnist('train')
    model = train(Xtr, Ytr, epochs=epochs)
    Xte, Yte = load_mnist('test')
    with torch.no_grad():
        model.eval()
        Xt = _norm(Xte)
        pred = torch.cat([model(Xt[i:i+512].to(DEV)).argmax(1).cpu()
                          for i in range(0, len(Xt), 512)])
    acc = (pred.numpy() == Yte).mean()
    print(f'[oracle] MNIST 테스트 정확도 {acc:.4f}')
    torch.save(model.state_dict(), CKPT)
    model.eval()
    return model


@torch.no_grad()
def legibility(model, imgs, labels, bs=512):
    """
    imgs: (N,28,28) uint8, labels: (N,) int
    returns: (N,) 정답 클래스 확률. 1 에 가까울수록 그 숫자로 잘 읽힘.
    """
    model.eval()
    imgs = np.asarray(imgs)
    if imgs.ndim == 2:
        imgs = imgs[None]
    labels = np.atleast_1d(np.asarray(labels))
    Xt = _norm(imgs.astype(np.uint8))
    out = []
    for i in range(0, len(Xt), bs):
        out.append(torch.softmax(model(Xt[i:i+bs].to(DEV)), 1).cpu())
    P = torch.cat(out).numpy()
    return P[np.arange(len(labels)), labels]
