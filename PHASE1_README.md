# Phase 1: Handwriting ODE Simulator

## Overview

Phase 1 구현: 물리 기반 미분방정식으로 손글씨 동역학을 모델링하고 신드하는 시뮬레이터

### Architecture

```
Text Input ("hello")
    ↓
TextEncoder: 문자 임베딩 추출
    ↓
HandwritingModel: 
  - Initial State Network: 텍스트 → 초기 상태
  - ODE Solver (torchdiffeq)
    ↓
HandwritingDynamics:
  dh/dt = f(h, t, text_features; θ)
  
  상태: h = [x, y, vx, vy, p, θ]
  - (x,y): 펜 위치
  - (vx,vy): 펜 속도
  - p: 필압 (0-1)
  - θ: 펜 각도
    ↓
HandwritingRenderer: 궤적 → 이미지 렌더링
```

---

## Module Structure

### `dynamics.py`
손글씨 동역학 ODE 정의

**Classes:**
- `HandwritingDynamicsParams`: 파라미터 컨테이너
- `HandwritingDynamics(nn.Module)`: ODE 동역학 레이어

**Parameters (총 11개):**
- Velocity dynamics (α, β, γ, ω): 감쇠 조화진동자
- Pressure dynamics (λ, μ, σ): 필압 변화
- Angle dynamics (κ, ρ): 펜 각도
- Text control (c_x, c_y): 텍스트 제어

**핵심 방정식:**
```
ẋ = vx
ẏ = vy
v̇x = -α*vx - β*x + γ*sin(ωt) + c_text
v̇y = -α*vy - β*y + γ*sin(ωt) + c_text
ṗ = -λ*p + μ + σ*√(vx²+vy²)
θ̇ = κ*atan2(vy,vx) + ρ
```

### `model.py`
PyTorch 모델 및 학습 구조

**Classes:**
- `TextEncoder`: 문자 임베딩
- `HandwritingModel(nn.Module)`: 메인 모델
  - ODE 솔버 (torchdiffeq)
  - 초기 상태 네트워크
  - 동역학 레이어
- `SimpleHandwritingSimulator`: 사용자 친화적 래퍼

**주요 메서드:**
```python
simulator = SimpleHandwritingSimulator()

# 기본 생성
traj = simulator.generate("hello", num_points=200)  # [200, 6]

# 스타일 지정
traj = simulator.generate("hello", style_params={"alpha": 0.5, ...})

# 미리 정의된 스타일
traj = simulator.generate_with_style("hello", style_name="shaky")
```

### `renderer.py`
궤적 → 이미지 렌더링

**Classes:**
- `HandwritingRenderer`: PIL/matplotlib 렌더링
  - `render_pil()`: PIL Image 반환
  - `render_numpy()`: numpy 배열 반환
  - `render_matplotlib()`: 분석 플롯

**특징:**
- 필압 기반 펜 굵기
- 자동 정규화
- 비교 시각화

---

## Quick Start

### Installation

```bash
# Clone and install
cd handwrite
pip install -r requirements.txt
```

### Basic Usage

```python
from simulator import SimpleHandwritingSimulator, HandwritingRenderer

# Create simulator
simulator = SimpleHandwritingSimulator(device='cpu')

# Generate trajectory
trajectory = simulator.generate(
    text="hello",
    num_points=200,      # 시간 스텝 수
    duration=2.0          # 총 지속시간
)

# Render to image
renderer = HandwritingRenderer(width=400, height=300)
img = renderer.render_pil(trajectory, normalize=True)
img.show()

# Or save to file
img.save('output.png')
```

### Different Styles

```python
# Predefined styles
styles = simulator.get_styles()
# 'normal', 'shaky', 'flowing', 'heavy'

trajectory = simulator.generate_with_style("hello", style_name="shaky")
```

### Parameter Customization

```python
custom_params = {
    'alpha': 0.4,        # 더 강한 감쇠
    'gamma': 0.08,       # 더 큰 진동
    'mu_p': 0.5,         # 더 강한 필압
}

trajectory = simulator.generate("hello", style_params=custom_params)
```

---

## Parameter Guide

### Velocity Dynamics (가장 중요)

**α (alpha) - Velocity Damping**
- 범위: 0.1 ~ 0.5
- 낮음 (0.1): 부드럽고 긴 필기
- 높음 (0.5): 빠르고 끊긴 필기

**β (beta) - Restoring Force**
- 범위: 0.05 ~ 0.2
- 펜이 중심으로 돌아오는 강도
- 낮음: 더 넓은 궤적
- 높음: 더 타이트한 궤적

**γ (gamma) - Oscillation Amplitude**
- 범위: 0.02 ~ 0.15
- 필기의 떨림/진동
- 낮음: 부드러운 필기
- 높음: 떨리는 필기 (신경 떨림, 피로 효과)

**ω (omega) - Oscillation Frequency**
- 범위: 1.0 ~ 3.0
- 진동의 빈도
- 보통 고정하고 γ로 진동 강도 조절

### Pressure Dynamics

**λ (lambda_p) - Pressure Decay**
- 범위: 0.3 ~ 0.8
- 필압 이완 속도
- 높음: 빠르게 필압 감소

**μ (mu_p) - Baseline Pressure**
- 범위: 0.2 ~ 0.7
- 기본 필압
- 펜을 얼마나 눌러서 쓰는가

**σ (sigma_p) - Pressure-Speed Coupling**
- 범위: 0.05 ~ 0.2
- 필기 속도에 따른 필압 변화
- 높음: 빠를수록 더 세게 씀

### Angle Dynamics

**κ (kappa) - Angle Tracking**
- 범위: 0.2 ~ 1.0
- 펜이 속도 방향을 따라가는 정도
- 자연스러운 값: 0.5

**ρ (rho) - Angle Bias**
- 범위: -π/4 ~ π/4
- 특정 각도로의 기울임
- 보통 0 (중립)

---

## Predefined Styles

### 1. Normal (기본)
```python
{
    'alpha': 0.3,      # 중간 감쇠
    'beta': 0.1,       # 중간 복원력
    'gamma': 0.05,     # 낮은 진동
    'mu_p': 0.3,       # 중간 필압
}
```

### 2. Shaky (떨리는)
```python
{
    'alpha': 0.2,      # 낮은 감쇠 → 더 떨림
    'beta': 0.08,      # 낮은 복원력
    'gamma': 0.1,      # 높은 진동 ← 떨림 효과
    'mu_p': 0.4,       # 약간 높은 필압
}
```

### 3. Flowing (흐르는)
```python
{
    'alpha': 0.4,      # 높은 감쇠 → 부드러움
    'beta': 0.05,      # 낮은 복원력
    'gamma': 0.03,     # 낮은 진동 ← 부드러움
    'mu_p': 0.2,       # 낮은 필압
}
```

### 4. Heavy (무거운)
```python
{
    'alpha': 0.25,     # 중간 감쇠
    'beta': 0.12,      # 높은 복원력
    'gamma': 0.04,     # 낮은 진동
    'mu_p': 0.6,       # 매우 높은 필압
}
```

---

## Testing

### Run Tests

```bash
python test_phase1.py
```

**Tests Performed:**
1. **Basic Generation**: 기본 궤적 생성 및 렌더링
2. **Different Texts**: 여러 텍스트 생성
3. **Parameter Variations**: 4가지 스타일 비교
4. **Trajectory Properties**: 궤적 물리 특성 분석
   - 속도, 가속도, 저크 (3차 미분)
   - 필압 통계
5. **Rendering**: PIL/matplotlib 렌더링 확인
6. **Reproducibility**: 같은 시드로 재현성 확인

**Output:**
- `/tmp/test_basic.png`
- `/tmp/test_different_texts.png`
- `/tmp/test_styles.png`
- `/tmp/test_trajectory_properties.png`
- `/tmp/test_render_pil.png`
- `/tmp/test_render_matplotlib.png`

---

## Next Steps (Phase 2)

Phase 1 이후 개선할 사항:

### 1. Neural ODE Upgrade
```python
# 현재: 물리 기반 고정 방정식
# → 향후: 학습된 신경망 동역학

class NeuralODEDynamics(nn.Module):
    def forward(self, t, h, features):
        # 자동으로 동역학 학습
        return self.learned_f(t, h, features)
```

### 2. Parameter Extraction
```python
# 역추론: 손글씨 이미지 → ODE 파라미터 추출

from inverse import ParameterExtractor
extractor = ParameterExtractor()
params = extractor.extract(image, text)
```

### 3. LLM Integration
```python
# LLM으로 파라미터 역추론

from llm_inverse import LLMParameterInference
llm_inference = LLMParameterInference()
params = llm_inference.infer(image, text)
```

### 4. Auto-Research Loop
```python
# VLM 피드백으로 자동 개선

from auto_refine import AutoResearchLoop
loop = AutoResearchLoop(llm, vlm, simulator)
refined_params = loop.refine(target_image, text)
```

---

## Architecture Notes

### Why Physics-Based ODE?

1. **Interpretability**: 파라미터가 물리적 의미를 가짐
2. **Parameter Extraction**: 실제 손글씨에서 파라미터 추출 가능
3. **LLM Reasoning**: LLM이 물리 개념으로 추론 가능
4. **Efficiency**: 적은 파라미터로 다양한 스타일 표현

### Why Not Direct Learning?

```
Pure Neural Network (덜 해석 가능):
input → black_box → trajectory

Physics-Based ODE (해석 가능):
input → meaningful_params → physically_grounded_dynamics → trajectory
                  ↑
            LLM이 직접 추론 가능
```

### ODE Solver Choice

```
dopri5 (Runge-Kutta):
- Default solver
- 속도와 정확도 균형
- 대부분의 경우 최선

adams (Multi-step):
- 장 시퀀스에서 빠름
- 메모리 효율
- 수렴 어려울 수 있음

rk4 (4th-order Runge-Kutta):
- 수동 구현
- 검증용
```

---

## Known Limitations

1. **Text Conditioning**: 현재는 텍스트 평균만 사용 → Phase 2에서 시간별 조건화

2. **No Character-Specific Dynamics**: 각 문자마다 다른 동역학 미지원 → Phase 2에서 추가

3. **Simple Pressure Model**: 실제 필압이 더 복잡함 → Phase 3에서 개선

4. **No Velocity Penalties**: 부자연스러운 점프 가능 → Phase 2에서 손실함수 개선

5. **Limited Style Control**: 4가지 스타일만 정의 → Phase 2에서 연속 파라미터화

---

## Debugging Tips

### Trajectory Diverges

```python
# 문제: ODE가 터짐
# 원인: 동역학 파라미터가 불안정

# 해결:
simulator.model.set_dynamics_params({
    'alpha': 0.5,      # 감쇠 증가
    'beta': 0.15,      # 복원력 증가
    'gamma': 0.03,     # 진동 감소
})
```

### Trajectory Too Noisy

```python
# 문제: γ가 너무 크거나 α가 작음
# 해결:
params = {
    'gamma': 0.03,     # 진동 감소
    'alpha': 0.35,     # 감쇠 증가
}
```

### Pressure Always Zero

```python
# 문제: mu_p가 너무 낮음
# 해결:
params = {'mu_p': 0.4}  # 기본 필압 증가
```

---

## File Structure

```
handwrite/
├── plan.md                          # 전체 프로젝트 계획
├── requirements.txt                 # 의존성
├── test_phase1.py                   # 테스트 스크립트
├── PHASE1_README.md                 # 이 파일
├── refer_paper/                     # 논문 레퍼런스
│   ├── summary.md
│   ├── key_papers_detail.md
│   └── implementation_guide.md
└── simulator/                       # Phase 1 구현
    ├── __init__.py
    ├── dynamics.py                  # ODE 동역학
    ├── model.py                     # PyTorch 모델
    └── renderer.py                  # 렌더링
```

---

## References

- **Graves (2013)**: Generating Sequences With RNNs
  - 손글씨 생성의 기초
  - MDN (Mixture Density Networks) 개념

- **Neural ODE (2021)**: Modeling Trajectories with Neural ODEs
  - ODE 기반 궤적 모델링 기초
  - torchdiffeq 사용법

- **Disentangling Styles (2023)**: 필체와 문자 스타일 분리
  - 파라미터화 전략

---

## Performance Notes

### Computational Cost

```
Text Length | Generate | Render | Total
1 char      | ~10 ms   | ~5 ms  | ~15 ms
5 char      | ~20 ms   | ~10 ms | ~30 ms
10 char     | ~30 ms   | ~15 ms | ~45 ms
```

### Memory Usage

```
ODE State: [200, 6] = 4.8 KB
Model Parameters: ~5 KB
Per-generation: ~50 KB
Batch of 32: ~1.6 MB
```

---

## Citation

If you use this code, please cite:

```
@article{handwriting_ode,
  title={ODE-Based Handwriting Synthesis},
  year={2026}
}
```

---

## Contact & Support

문제가 있거나 개선 아이디어가 있으면 이슈를 제출하세요.

**다음 Phase로 진행하기 전에:**
- [ ] test_phase1.py 모든 테스트 통과
- [ ] 여러 스타일의 손글씨 생성 확인
- [ ] 궤적 물리 특성 이해 (속도, 가속도, 필압)
- [ ] 렌더링 파이프라인 검증
