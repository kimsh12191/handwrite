"""MNIST 로더 + 합성/폰트 데이터셋 생성."""
import gzip, os, urllib.request
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CACHE = os.path.join(os.path.dirname(__file__), '_cache')
MIRRORS = [
    'https://ossci-datasets.s3.amazonaws.com/mnist/',
    'https://raw.githubusercontent.com/fgnt/mnist/master/',
]
FILES = dict(
    train_x='train-images-idx3-ubyte.gz', train_y='train-labels-idx1-ubyte.gz',
    test_x='t10k-images-idx3-ubyte.gz',   test_y='t10k-labels-idx1-ubyte.gz')


def _fetch(fname):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, fname)
    if os.path.exists(p):
        return p
    last = None
    for m in MIRRORS:
        try:
            urllib.request.urlretrieve(m + fname, p)
            return p
        except Exception as e:
            last = e
    raise RuntimeError(f'{fname} 다운로드 실패: {last}')


def load_mnist(split='test'):
    """returns (N,28,28) uint8, (N,) int64"""
    xp = _fetch(FILES[f'{split}_x']); yp = _fetch(FILES[f'{split}_y'])
    with gzip.open(xp, 'rb') as f:
        x = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    with gzip.open(yp, 'rb') as f:
        y = np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)
    return x.copy(), y


# ---------------------------------------------------------------- 합성 (시뮬레이터)
def gen_synth(n, seed=0, writers_per=64, progress=None):
    """
    시뮬레이터로 n 장 생성.
    writers_per: 한 '필기자'(스타일 벡터)당 생성할 장수.
                 실제 데이터도 소수의 사람이 많이 쓰므로 이 구조를 흉내낸다.
    """
    from .simulator import sample_style, render_digit
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 28, 28), np.uint8); Y = np.zeros(n, np.int64)
    style = None
    for i in range(n):
        if i % writers_per == 0:
            style = sample_style(rng)
        d = int(rng.integers(10))
        X[i] = render_digit(str(d), style, rng); Y[i] = d
        if progress and i % progress == 0 and i:
            print(f'    {i}/{n}', flush=True)
    return X, Y


# ---------------------------------------------------------------- 폰트 기준선
def _find_fonts(limit=120):
    roots = ['/usr/share/fonts', '/mnt/skills', '/opt']
    import matplotlib
    roots.append(os.path.join(os.path.dirname(matplotlib.__file__),
                              'mpl-data', 'fonts', 'ttf'))
    out = []
    for r in roots:
        for dp, _, fns in os.walk(r):
            for fn in fns:
                if fn.lower().endswith(('.ttf', '.otf')):
                    out.append(os.path.join(dp, fn))
    return sorted(set(out))[:limit]


def gen_font(n, seed=0, progress=None):
    """
    폰트 기반 기준선. plan.md 의 '시뮬레이터가 폰트보다 나은가' 를 재기 위한 것.
    합성 쪽과 동일한 MNIST 전처리·왜곡 예산을 준다 (공정 비교).
    """
    from .simulator import to_mnist_format
    rng = np.random.default_rng(seed)
    fonts = _find_fonts()
    if not fonts:
        raise RuntimeError('폰트를 찾지 못함')
    cache = {}
    X = np.zeros((n, 28, 28), np.uint8); Y = np.zeros(n, np.int64)
    made = 0
    while made < n:
        fp = fonts[rng.integers(len(fonts))]
        size = int(rng.integers(64, 96))
        key = (fp, size)
        if key not in cache:
            try:
                cache[key] = ImageFont.truetype(fp, size)
            except Exception:
                cache[key] = None
        font = cache[key]
        if font is None:
            continue
        d = int(rng.integers(10))
        img = Image.new('L', (160, 160), 0)
        dr = ImageDraw.Draw(img)
        try:
            dr.text((80, 80), str(d), fill=255, font=font, anchor='mm')
        except Exception:
            continue
        if np.array(img).max() < 20:
            continue
        # 합성 쪽과 대등한 왜곡: 회전 + 전단 + 굵기 변화
        ang = rng.uniform(-11, 11)
        shear = np.tan(np.deg2rad(rng.uniform(-22, 28)))
        img = img.rotate(ang, resample=Image.BILINEAR)
        img = img.transform((160, 160), Image.AFFINE,
                            (1, shear, -shear*80, 0, 1, 0),
                            resample=Image.BILINEAR)
        if rng.random() < 0.5:
            from PIL import ImageFilter
            img = img.filter(ImageFilter.MaxFilter(3) if rng.random() < 0.5
                             else ImageFilter.MinFilter(3))
        X[made] = np.array(to_mnist_format(img), np.uint8); Y[made] = d
        made += 1
        if progress and made % progress == 0:
            print(f'    {made}/{n}', flush=True)
    return X, Y


# ---------------------------------------------------------------- 시각화
def contact_sheet(blocks, path, cell=28, pad=2, scale=3):
    """blocks: [(label, X(n,28,28)), ...] -> 비교용 이미지 저장"""
    from PIL import ImageFont as IF
    rows = sum(len(X) // 10 for _, X in blocks)
    H = rows*(cell+pad)+pad + 16*len(blocks)
    W = 10*(cell+pad)+pad
    sheet = Image.new('L', (W, H), 30)
    y = 0
    dr = ImageDraw.Draw(sheet)
    for label, X in blocks:
        dr.text((3, y+3), label, fill=255)
        y += 16
        for r in range(len(X)//10):
            for c in range(10):
                sheet.paste(Image.fromarray(X[r*10+c]),
                            (pad+c*(cell+pad), y+pad+r*(cell+pad)))
            y_end = y+pad+(r+1)*(cell+pad)
        y = y_end
    sheet = sheet.resize((W*scale, H*scale), Image.NEAREST)
    sheet.save(path)
    return path
