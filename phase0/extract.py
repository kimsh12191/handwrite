"""
실제 필기 사진에서 낱글자를 뽑아 MNIST 형식으로 변환.

사진이라 조명이 고르지 않고 JPEG 잡음이 있으므로,
크게 흐린 배경을 빼서 조명을 평탄화한 뒤 이진화한다.
"""
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, label, binary_dilation, binary_closing

from .simulator import to_mnist_format


def ink_map(path, bg_sigma=40, thresh=0.30):
    """사진 -> 잉크 강도 맵 (잉크=밝음, 0..255)."""
    a = np.array(Image.open(path).convert('L'), float)
    bg = gaussian_filter(a, bg_sigma)
    ink = np.clip(bg - a, 0, None)          # 배경보다 어두운 만큼이 잉크
    if ink.max() > 0:
        ink = ink / ink.max() * 255
    mask = ink > thresh * 255
    mask = binary_closing(mask, np.ones((3, 3)))
    return ink, mask


def components(mask, min_pix=60, merge_gap=6):
    """
    연결 성분 -> (x0,y0,x1,y1) 목록, 좌->우 정렬.
    merge_gap: 가로로 이만큼 이내면 같은 글자로 합침 (획이 끊긴 경우 대비).
    """
    lab, n = label(binary_dilation(mask, np.ones((3, merge_gap*2+1))))
    boxes = []
    for i in range(1, n+1):
        ys, xs = np.nonzero(lab == i)
        if len(xs) < min_pix:
            continue
        boxes.append((xs.min(), ys.min(), xs.max()+1, ys.max()+1))
    return sorted(boxes, key=lambda b: b[0])


def crop_to_mnist(ink, box, pad=4):
    x0, y0, x1, y1 = box
    h, w = ink.shape
    sub = ink[max(0, y0-pad):min(h, y1+pad), max(0, x0-pad):min(w, x1+pad)]
    return np.array(to_mnist_format(
        Image.fromarray(sub.astype(np.uint8))), dtype=np.uint8)


def baseline_angle(boxes):
    """
    성분 중심들을 지나는 직선의 각도(도). 기울어진 줄에 쓴 경우를 잡는다.
    글자꼴이 아니라 레이아웃 성질이므로 피팅 전에 제거하는 편이 낫다.
    """
    if len(boxes) < 2:
        return 0.0
    c = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in boxes], float)
    c -= c.mean(0)
    # 주성분 방향 = 글자들이 늘어선 방향
    u, s, vt = np.linalg.svd(c, full_matrices=False)
    dx, dy = vt[0]
    return float(np.degrees(np.arctan2(dy, dx)))


def extract(path, deskew=True, **kw):
    """returns (ink, boxes, [img28,...], angle)"""
    ink, mask = ink_map(path, **{k: v for k, v in kw.items()
                                 if k in ('bg_sigma', 'thresh')})
    comp_kw = {k: v for k, v in kw.items() if k in ('min_pix', 'merge_gap')}
    boxes = components(mask, **comp_kw)

    angle = baseline_angle(boxes) if deskew else 0.0
    if deskew and abs(angle) > 2:
        im = Image.fromarray(ink.astype(np.uint8)).rotate(
            angle, resample=Image.BICUBIC, expand=True, fillcolor=0)
        ink = np.array(im, float)
        mask = ink > (kw.get('thresh', 0.30) * 255)
        mask = binary_closing(mask, np.ones((3, 3)))
        boxes = components(mask, **comp_kw)

    return ink, boxes, [crop_to_mnist(ink, b) for b in boxes], angle
