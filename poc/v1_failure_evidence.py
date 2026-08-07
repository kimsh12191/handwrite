"""
Phase 1 dynamics.py 의 ODE 를 numpy 로 그대로 재현해서,
실제로 '글자'가 나오는지 검증한다.

dynamics.py 의 식:
  dx/dt = vx
  dy/dt = vy
  dvx/dt = -alpha*vx - beta*x + gamma*sin(omega*t) + c_text_x * text_feat
  dvy/dt = -alpha*vy - beta*y + gamma*sin(omega*t) + c_text_y * text_feat
  dp/dt  = -lambda_p*p + mu_p + sigma_p*sqrt(vx^2+vy^2)
  dtheta/dt = kappa*(atan2(vy,vx) - theta) + rho

text_feat 은 model.py 에서 text_features[..., :1] 로 들어감 = 스칼라 상수 하나.
"""
import numpy as np

def simulate(text_feat, params, T=2.0, N=400, h0=None):
    a  = params['alpha']; b = params['beta']
    g  = params['gamma']; w = params['omega']
    lp = params['lambda_p']; mp = params['mu_p']; sp = params['sigma_p']
    ka = params['kappa']; rho = params['rho']
    cx = params['c_text_x']; cy = params['c_text_y']

    dt = T / N
    if h0 is None:
        h0 = np.array([0.0, 0.0, 0.1, 0.1, 0.3, 0.0])
    h = h0.copy()
    traj = np.zeros((N, 6))

    for i in range(N):
        t = i * dt
        x, y, vx, vy, p, th = h
        osc = g * np.sin(w * t)
        dx  = vx
        dy  = vy
        dvx = -a*vx - b*x + osc + cx*text_feat
        dvy = -a*vy - b*y + osc + cy*text_feat
        speed = np.sqrt(vx*vx + vy*vy + 1e-8)
        dp  = -lp*p + mp + sp*speed
        dth = ka*(np.arctan2(vy, vx+1e-8) - th) + rho
        h = h + dt*np.array([dx, dy, dvx, dvy, dp, dth])
        traj[i] = h
    return traj


DEFAULT = dict(alpha=0.3, beta=0.1, gamma=0.05, omega=2.0,
               lambda_p=0.5, mu_p=0.3, sigma_p=0.1,
               kappa=0.5, rho=0.0, c_text_x=0.1, c_text_y=0.1)

print("=" * 70)
print("검증 1: 서로 다른 텍스트가 서로 다른 궤적을 만드는가?")
print("=" * 70)
# model.py: text_features = embedding(ord(c)%256), 그 중 [...,:1] 첫 차원.
# 임베딩은 랜덤 초기화이므로, 텍스트마다 다른 스칼라가 나온다고 가정하고
# 넓은 범위로 스윕해본다.
for name, tf in [("text A", -0.5), ("text B", 0.0), ("text C", 0.5), ("text D", 2.0)]:
    tr = simulate(tf, DEFAULT)
    x, y = tr[:, 0], tr[:, 1]
    print(f"{name:8s} (feat={tf:+.2f}): "
          f"x=[{x.min():+.3f},{x.max():+.3f}] y=[{y.min():+.3f},{y.max():+.3f}] "
          f"  x==y 상관계수={np.corrcoef(x, y)[0,1]:+.6f}")

print()
print("=" * 70)
print("검증 2: 궤적의 실제 모양 (ASCII 플롯)")
print("=" * 70)

def ascii_plot(traj, w=60, h=20, label=""):
    x, y = traj[:, 0], traj[:, 1]
    xr = x.max() - x.min(); yr = y.max() - y.min()
    if xr < 1e-9: xr = 1.0
    if yr < 1e-9: yr = 1.0
    grid = [[' '] * w for _ in range(h)]
    for xi, yi in zip(x, y):
        cx = int((xi - x.min()) / xr * (w - 1))
        cy = int((1 - (yi - y.min()) / yr) * (h - 1))
        grid[cy][cx] = '#'
    print(f"--- {label} ---")
    for row in grid:
        print('|' + ''.join(row) + '|')

ascii_plot(simulate(0.5, DEFAULT), label="text_feat=0.5, 기본 파라미터")
print()
ascii_plot(simulate(2.0, DEFAULT), label="text_feat=2.0, 기본 파라미터")

print()
print("=" * 70)
print("검증 3: x(t) 와 y(t) 는 독립적인 자유도를 갖는가?")
print("=" * 70)
tr = simulate(0.5, DEFAULT)
x, y = tr[:, 0], tr[:, 1]
print(f"x(t) 와 y(t) 의 피어슨 상관계수: {np.corrcoef(x, y)[0,1]:+.8f}")
print(f"max|x(t) - y(t)| = {np.abs(x - y).max():.8f}")
print()
print("해석: c_text_x == c_text_y 이고 나머지 계수가 x,y 에 대해 완전히 대칭이므로")
print("      x(t) 와 y(t) 는 초기조건이 같으면 '완전히 동일한' 함수가 된다.")
print("      => 궤적은 항상 y=x 직선 위에만 존재. 2D 글자를 그릴 자유도가 없음.")

print()
print("=" * 70)
print("검증 4: 초기조건을 다르게 줘도 글자가 되는가?")
print("=" * 70)
for h0 in [np.array([0., 0., 1.0, 0.2, 0.3, 0.]),
           np.array([0., 0., 0.2, 1.0, 0.3, 0.]),
           np.array([0.5, -0.5, 0.3, -0.3, 0.3, 0.])]:
    tr = simulate(0.5, DEFAULT, h0=h0)
    x, y = tr[:, 0], tr[:, 1]
    # 궤적이 몇 번 방향을 바꾸는지 = 글자다움의 아주 거친 지표
    dx = np.diff(x); dy = np.diff(y)
    turns = np.sum(np.diff(np.sign(dx)) != 0) + np.sum(np.diff(np.sign(dy)) != 0)
    print(f"h0={h0[:4]}: 방향전환 {turns}회, "
          f"x범위={x.max()-x.min():.3f}, y범위={y.max()-y.min():.3f}")
ascii_plot(simulate(0.5, DEFAULT, h0=np.array([0., 0., 1.0, 0.2, 0.3, 0.])),
           label="초기속도 (1.0, 0.2)")
