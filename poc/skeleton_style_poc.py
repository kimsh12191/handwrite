"""
새 접근의 핵심 가정 검증:
  글자 = [스켈레톤: 무엇을 쓰는가] x [스타일 파라미터: 어떻게 쓰는가]

Phase 1 은 '무엇을 쓰는가'가 없어서 직선만 나왔다.
여기서는 스켈레톤을 넣고, 스타일 파라미터만 바꿔서
같은 글자가 다른 필체로 나오는지 확인한다.
"""
import numpy as np

# ---------- Layer 0: 스켈레톤 (정규화 em-box 0..1, 획별 제어점) ----------
# 각 획 = 제어점 리스트. 획 사이는 pen-up.
SKEL = {
    '0': [[(.5,.95),(.15,.72),(.15,.28),(.5,.05),(.85,.28),(.85,.72),(.5,.95)]],
    '1': [[(.30,.78),(.52,.95),(.52,.05)]],
    '2': [[(.15,.78),(.35,.95),(.70,.92),(.75,.70),(.15,.10),(.85,.08)]],
    '3': [[(.18,.85),(.55,.97),(.78,.80),(.50,.55),(.80,.35),(.62,.06),(.18,.15)]],
    '7': [[(.12,.92),(.88,.92),(.38,.05)]],
    'ㄱ': [[(.10,.85),(.85,.85),(.55,.10)]],
    'ㅣ': [[(.50,.95),(.50,.05)]],
    'ㅁ': [[(.15,.85),(.15,.15)],[(.15,.85),(.85,.85),(.85,.15),(.15,.15)]],
}

# ---------- Layer 1: 스타일 파라미터 ----------
DEFAULT_STYLE = dict(
    x_height=1.0,       # 세로 크기
    width_ratio=1.0,    # 가로/세로 비
    slant_deg=0.0,      # 기울기
    curvature=0.0,      # 획 부풀림 (양수=바깥으로 볼록)
    tremor_amp=0.0,     # 떨림 진폭
    tremor_freq=9.0,    # 떨림 주파수
    baseline_drift=0.0, # 베이스라인 기울기
    overshoot=0.0,      # 획 끝 넘김
    char_gap=0.30,
    seed=0,
)

def catmull_rom(pts, n=80):
    """제어점을 부드러운 곡선으로. 글자의 '획'을 만든다."""
    P = np.asarray(pts, float)
    if len(P) < 2: return P
    P = np.vstack([P[0], P, P[-1]])
    out = []
    for i in range(len(P)-3):
        p0,p1,p2,p3 = P[i],P[i+1],P[i+2],P[i+3]
        t = np.linspace(0,1,max(2,n//(len(P)-3)))[:,None]
        out.append(0.5*((2*p1) + (-p0+p2)*t
                   + (2*p0-5*p1+4*p2-p3)*t**2
                   + (-p0+3*p1-3*p2+p3)*t**3))
    return np.vstack(out)

def apply_style(stroke, S, rng, x_off):
    """스켈레톤 획 하나에 스타일을 입힌다. 여기가 '개인 필체'."""
    P = catmull_rom(stroke)
    x, y = P[:,0].copy(), P[:,1].copy()
    s = np.linspace(0, 1, len(x))

    # overshoot: 획 끝을 진행방향으로 살짝 넘김
    if S['overshoot'] != 0 and len(x) > 2:
        dx, dy = x[-1]-x[-2], y[-1]-y[-2]
        x[-1] += dx*S['overshoot']*6; y[-1] += dy*S['overshoot']*6

    # curvature: 획을 법선방향으로 부풀림
    if S['curvature'] != 0:
        tx, ty = np.gradient(x), np.gradient(y)
        L = np.hypot(tx,ty)+1e-9
        nx, ny = -ty/L, tx/L
        bulge = np.sin(np.pi*s)*S['curvature']
        x += nx*bulge; y += ny*bulge

    # tremor: 사람 손의 미세 떨림
    if S['tremor_amp'] != 0:
        ph = rng.uniform(0, 2*np.pi, 2)
        x += S['tremor_amp']*np.sin(2*np.pi*S['tremor_freq']*s + ph[0])
        y += S['tremor_amp']*np.sin(2*np.pi*S['tremor_freq']*s + ph[1])

    # 크기
    x *= S['width_ratio']; y *= S['x_height']
    # slant: y 에 비례해 x 를 밀기
    x += y*np.tan(np.deg2rad(S['slant_deg']))
    # 문자 위치 + baseline drift
    x += x_off
    y += x_off*S['baseline_drift']
    return x, y

def render_text(text, style=None):
    S = dict(DEFAULT_STYLE); S.update(style or {})
    rng = np.random.default_rng(S['seed'])
    strokes, x_off = [], 0.0
    for ch in text:
        if ch not in SKEL:
            x_off += S['char_gap']; continue
        for st in SKEL[ch]:
            strokes.append(apply_style(st, S, rng, x_off))
        x_off += S['width_ratio'] + S['char_gap']
    return strokes

def ascii_plot(strokes, w=76, h=17, label=""):
    xs = np.concatenate([s[0] for s in strokes]); ys = np.concatenate([s[1] for s in strokes])
    xr, yr = max(xs.max()-xs.min(),1e-9), max(ys.max()-ys.min(),1e-9)
    sc = min((w-1)/xr, (h-1)/yr)
    g = [[' ']*w for _ in range(h)]
    for X, Y in strokes:
        # 획 내부를 촘촘히 보간해서 끊김 방지
        for xi, yi in zip(X, Y):
            cx = int((xi-xs.min())*sc); cy = int((1-(yi-ys.min())/yr)*(h-1))
            if 0 <= cx < w and 0 <= cy < h: g[cy][cx] = '#'
    print(f"  {label}")
    for r in g: print('  |'+''.join(r)+'|')

print("="*82)
print("검증 A: 스켈레톤이 있으면 '읽을 수 있는 글자'가 나오는가?")
print("="*82)
ascii_plot(render_text("0123"), label="'0123' 기본 스타일")
print()
ascii_plot(render_text("ㄱㅣㅁ"), label="'ㄱㅣㅁ' (한글 자모)")

print()
print("="*82)
print("검증 B: 같은 텍스트 + 다른 스타일 파라미터 = 다른 필체가 나오는가?")
print("="*82)
STYLES = {
 "기본":        {},
 "기울임 25도": {'slant_deg':25},
 "떨리는 손":   {'tremor_amp':.035,'tremor_freq':11,'seed':3},
 "납작+흘림":   {'x_height':.65,'width_ratio':1.25,'curvature':.07,'overshoot':.05},
}
for name, st in STYLES.items():
    ascii_plot(render_text("017", st), h=13, label=f"'017' — {name}  {st}")
    print()

print("="*82)
print("검증 C: 스타일 파라미터가 고정되면, 다른 텍스트도 같은 필체로 나오는가?")
print("       (= 사용자가 원한 '파라미터 베이스로 다양한 text 생성')")
print("="*82)
persona = {'slant_deg':18,'tremor_amp':.02,'curvature':.05,'x_height':.9,'seed':7}
for t in ["012", "731", "ㄱㅣㅁ"]:
    ascii_plot(render_text(t, persona), h=12, label=f"'{t}' — 동일 필체 파라미터")
    print()
