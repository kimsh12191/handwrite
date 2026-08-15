# HANDOFF.md — Claude / Coding Agent Handoff Guide

## 0. 이 프로젝트를 한 문장으로 정의하면

**이 프로젝트의 1차 목표는 handwriting dataset 분석 자체도, 손글씨 생성 모델 자체도 아니다.**

실제 사람들의 숫자·영문·한글 손글씨 데이터를 분석해서,

> **인간 손글씨의 형태와 필기 습관을 충분히 표현할 수 있는 해석 가능한(interpretable) parametric renderer의 표현 공간을 데이터 기반으로 설계하는 것**

이 핵심 목표다.

최종적으로는 다음 형태를 지향한다.

```text
text
  ↓
layout program
  ↓
glyph / jamo geometry
  ↓
stroke trajectory / motor program
  ↓
brush / raster renderer
  ↓
handwriting image
```

그리고 이 renderer가 실제 사람 손글씨를 얼마나 표현할 수 있는지를 정량적으로 검증한다.

---

# 1. 가장 중요한 전제

최적화나 VLM fitting보다 renderer가 먼저다.

최종적으로는 다음 inverse problem을 풀고 싶다.

\[
I = R(\theta)
\]

- `I`: 실제 손글씨 이미지
- `R`: parametric handwriting renderer
- `θ`: 개인의 handwriting parameter

실제 참조 이미지가 renderer의 표현 집합 안에 들어 있지 않으면,

\[
I_{ref} \notin \{R(\theta)\}
\]

아무리 좋은 optimizer나 VLM을 붙여도 정답을 찾을 수 없다.

따라서 현재 단계에서 가장 중요한 질문은:

> **"이 renderer가 우리가 다루려는 실제 인간 손글씨 family를 충분히 표현할 수 있는가?"**

다.

---

# 2. 지금 당장 renderer 파라미터를 상상해서 추가하지 말 것

잘못된 접근:

```text
ㄱ은 h_len이 필요할 것 같다.
ㅇ은 circle_x가 필요할 것 같다.
ㅅ은 angle이 필요할 것 같다.
→ 바로 renderer에 구현
```

권장 접근:

```text
실제 handwriting corpus
        ↓
문자 / 자모 / context별 분리
        ↓
shape / trajectory feature 추출
        ↓
PCA / clustering / distribution 분석
        ↓
실제 주요 variation axis 발견
        ↓
renderer parameter taxonomy 설계
```

즉 **parameter-first가 아니라 data-first**다.

---

# 3. 전체 프로젝트 루프

이 프로젝트의 핵심 개발 루프는 다음이다.

```text
실제 Handwriting Dataset
        ↓
문자 / 자모 / trajectory 분리
        ↓
실제 variation 분석
        ↓
Renderer parameter space 설계
        ↓
Renderer 구현
        ↓
실제 손글씨 fitting
        ↓
Coverage error 측정
        ↓
Failure sample 수집
        ↓
Failure clustering
        ↓
부족한 표현 자유도 발견
        ↓
Renderer 확장
        ↺
```

renderer가 충분히 좋아진 뒤에야 VLM 기반 fitting loop를 본격적으로 붙인다.

---

# 4. 현재 우선순위

반드시 다음 순서를 지킨다.

## Phase 1 — 공개 handwriting 데이터 수집

현재 bootstrap에서 다루는 대상:

- HangulDB PE92 / SERI95
- UJI Pen Characters v2
- EMNIST
- Naver handwriting font probe
- 수동 투입 가능한 Clova 나눔손글씨 109종
- 승인 후 투입 가능한 AI-Hub 71307

목표:
- 숫자
- 영문 대/소문자
- 한글 자모 및 자모가 초성/중성/종성에서 변형되는 방식
- trajectory / stroke topology

를 확보한다.

## Phase 2 — 실제 variation 분석

문자별로 최소한 다음을 본다.

```text
aspect
centroid
principal angle
moment ratio
left/right asymmetry
top/bottom asymmetry
compactness
connected components
hole count
stroke count
path length
stroke topology
```

trajectory 데이터가 있는 경우:

```text
duration
velocity profile
velocity peak position
acceleration
pressure
pressure-velocity relationship
corner slowdown
overshoot
pen-up / pen-down
```

도 본다.

## Phase 3 — renderer parameter taxonomy 설계

예를 들어 실제 `ㅇ` 데이터에서:

```text
PC1: width / height
PC2: rotation
PC3: left-right asymmetry
PC4: closure
```

가 주요 variation이라면 renderer에:

```text
circle_aspect
circle_rotation
circle_asymmetry
circle_closure
```

같은 자유도를 만든다.

중요:
PCA 이름 자체를 그대로 parameter로 쓰라는 뜻은 아니다.
실제 variation을 사람이 해석 가능한 생성 규칙으로 변환해야 한다.

## Phase 4 — renderer 구현

계층은 대략 다음과 같이 유지한다.

```text
Global / Page
├─ baseline
├─ spacing
├─ global scale / slant
└─ line-level drift

Character / Syllable
├─ width / height
├─ local rotation
├─ local offset
└─ component packing

Jamo / Glyph Shape
├─ segment length ratios
├─ corners
├─ angles
├─ loops
├─ closure
├─ asymmetry
└─ local deformation

Motor
├─ stroke order
├─ duration
├─ velocity profile
├─ corner slowdown
├─ overshoot
├─ pen lift
└─ micro variation

Brush
├─ width
├─ pressure response
├─ opacity
└─ edge softness
```

## Phase 5 — coverage test

실제 sample `I_i`마다:

\[
e_i = \min_\theta D(R(\theta), I_i)
\]

를 측정한다.

단순 pixel L2 하나만 쓰지 말고, 가능하면 다음을 나눈다.

- contour similarity
- skeleton similarity
- layout similarity
- topology
- perceptual similarity

## Phase 6 — failure clustering

renderer가 잘 못 맞추는 상위 5~10% sample을 모은다.

예:

```text
failure cluster A
→ 열린 ㅇ

failure cluster B
→ 비대칭 타원형 ㅇ

failure cluster C
→ 시작점/끝점이 겹치는 ㅇ
```

이 경우 renderer에 부족한 자유도가 있다는 뜻이다.

새 parameter를 추가하고 다시 coverage test를 한다.

---

# 5. 한글을 다룰 때 반드시 지킬 점

한글 완성형 11,172자를 독립적인 glyph renderer로 만들면 안 된다.

한글은 조합 구조를 이용해야 한다.

```text
초성
중성
종성
  ↓
자모 shape
  ↓
context-dependent deformation
```

특히 같은 자모라도 역할이 다르면 모양이 달라질 수 있다.

예:

```text
ㄱ as initial
ㄱ as final consonant
```

는 동일한 shape distribution이라고 가정하면 안 된다.

따라서 probe set에도:

- 모든 초성
- 모든 중성
- 모든 종성

을 동일 문맥에서 관찰할 수 있도록 context-controlled syllable을 포함한다.

---

# 6. 숫자 / 영문도 단순 class list로만 보지 말 것

문자별 분석과 함께 shape family를 함께 본다.

예:

```text
loop family:
0 O o a e d g q 6 8 9

vertical / stem family:
1 I l i t

angular family:
A V W X Y K 4 7

hump family:
m n h u

descender family:
g j p q y
```

renderer의 primitive가 여러 script에 공통적으로 재사용 가능한지 확인한다.

---

# 7. Motor와 Shape를 섞지 말 것

offline image에서:

- stroke order
- true velocity
- pen pressure
- pen lift timing

을 확정적으로 역추정할 수 있다고 가정하지 않는다.

따라서:

```text
offline image corpus
→ geometry / layout space

online trajectory corpus
→ motor prior / stroke topology
```

로 역할을 분리한다.

한국어 motor 분석에는 AI-Hub 71307이 가장 중요하다.

---

# 8. AI-Hub 71307 관련 현재 확인사항

공식 schema 기준:

```text
x: 0~756
y: 0~88

action:
DOWN
UP
MOVE

fields include:
pressure
velocity_magnitude
acceleration_magnitude
angle_consecutive_samples
log_curvature_radius
first / second differences
stroke_length_ratio
```

주의:
이전 대화 중 trajectory y가 이미지 height를 초과한다는 잘못된 해석이 있었으나,
공식 schema 확인 결과 해당 주장은 폐기한다.

---

# 9. VLM의 역할

VLM은 renderer가 충분한 표현 공간을 확보한 뒤에 사용한다.

VLM을 직접 30차원 optimizer처럼 쓰지 않는다.

권장 구조:

```text
REFERENCE + CURRENT
        ↓
VLM Verifier
        ↓
semantic discrepancy
"받침이 너무 넓다"
        ↓
parameter mapper
        ↓
1~2개 parameter axis
        ↓
code-generated candidate renders
        ↓
VLM candidate selection
        ↓
reference / previous / current 판정
```

VLM 역할:

- 시각적 차이 기술
- 큰 mismatch 우선순위 판단
- candidate 비교

코드 역할:

- parameter mapping
- search range
- candidate generation
- history
- accept / reject
- stopping rule

---

# 10. 기존 fitting protocol

향후 VLM fitting을 붙일 때 다음 프로토콜을 유지한다.

## Generator ↔ Verifier 분리

Verifier:
- 참조와 렌더의 차이만 기술
- parameter name 금지

Generator / Mapper:
- discrepancy를 parameter axis로 대응
- 값 또는 candidate 생성

## Staged decomposition

한 회차:
- mismatch 1개
- axis 2~3개 이하

순서:

```text
배치
→ 자소 / 문자 크기
→ glyph shape
→ motor
→ texture
```

## Trajectory memory

`trajectory.md`에:

- 변경 축
- 선택값
- 개선 여부
- 되돌린 변경
- 실패 이유

를 모두 남긴다.

## 3자 비교

항상:

```text
REFERENCE / PREVIOUS / CURRENT
```

로 판정한다.

---

# 11. 이 단계에서 하지 말아야 할 것

다음은 금지 또는 후순위다.

### 하지 말 것

- renderer를 먼저 대규모로 구현
- 11,172개 한글 완성형을 독립 shape로 수작업 구현
- VLM에게 모든 parameter를 한 번에 추측시키기
- 실제 variation 분석 없이 parameter를 계속 추가
- handwriting font를 motor ground truth로 사용
- offline image에서 실제 stroke order를 확정적으로 추론했다고 간주

### 아직 후순위

- DiffVG
- full differentiable renderer
- Sigma-Lognormal 전면 적용
- 고급 brush texture
- 대규모 VLM fitting

현재는 **representation-space discovery** 단계다.

---

# 12. 현재 코드베이스를 이어서 작업할 때의 기본 질문

어떤 기능을 추가하기 전에 항상 다음을 묻는다.

1. 이 작업은 실제 handwriting variation을 측정하는 데 도움이 되는가?
2. 이 작업은 renderer parameter taxonomy를 결정하는 근거가 되는가?
3. 이 작업은 coverage를 정량화하는 데 도움이 되는가?
4. 아니면 단지 지금 renderer를 더 예쁘게 만드는 작업인가?

4번이면 대부분 현재 단계에서는 하지 않는다.

---

# 13. 완료 조건

Renderer Space Discovery 단계가 끝났다고 볼 조건:

- 숫자 / 영문 / 한글 주요 class에 대해 실제 shape variation을 분석함
- 주요 shape family가 정의됨
- 한글 초성/중성/종성 context effect가 분석됨
- online trajectory에서 주요 motor variable이 정리됨
- renderer parameter taxonomy v1이 데이터 근거와 함께 만들어짐

Renderer v1 완료 조건:

- 실제 sample fitting 가능
- 문자별 coverage error 측정 가능
- failure sample clustering 가능
- parameter 추가 시 coverage 개선을 정량적으로 확인 가능

그 뒤에 VLM optimization 단계로 이동한다.

---

# 14. 최종 목표

최종적으로는 한 사람의 여러 handwriting sample에서 공통 parameter를 찾아:

\[
\theta_{writer}
\]

를 얻고,

그 사람이 한 번도 쓰지 않은 문장도:

```text
text
+
personal handwriting parameters
→
human-like handwriting trajectory/image
```

로 생성할 수 있는 구조를 목표로 한다.

핵심은 단순 style transfer가 아니라:

> **개인의 글쓰기 규칙을 해석 가능한 geometry / motor / brush parameter로 표현하는 것**

이다.
