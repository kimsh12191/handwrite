# 리서치 요약 — 손글씨 동역학 모델링

**검증 등급 표기**
- `[검증]` — 실제 바이트를 받아서 확인 (GitHub raw, PyPI API), 또는 수식을 직접 재유도·수치검증
- `[색인]` — 여러 검색에서 저자·연도·DOI가 일관되게 나왔으나 **논문 본문을 열지 못함**
- `[미확인]` — 확인 실패. 그대로 믿지 말 것

**접근 제약**: 사내 egress 정책으로 arxiv.org, Springer, ScienceDirect, IEEE Xplore, ACM DL, Google Scholar 가 모두 403 차단됨. github.com, raw.githubusercontent.com, pypi.org 만 직접 접근 가능. 따라서 논문은 대부분 `[색인]` 등급이며, **arXiv ID 는 전부 열어보지 못했으므로 인용 전 직접 확인 필요**.

---

## 1. Sigma-Lognormal (ΣΛ) — 이 프로젝트의 핵심 이론

Plamondon 의 **Kinematic Theory of Rapid Human Movements**. "사람마다 미분방정식을 세워놓고 그 기준으로 다양한 글씨를 생성한다"는 우리 컨셉의 40년 된 선행 연구이며, 가장 근접한 정답.

### 1.1 기본 문헌 `[색인]`

| 문헌 | 저자 | 출처 |
|---|---|---|
| A kinematic theory of rapid human movements, Part I | R. Plamondon | Biol. Cybern. 72(4):295–307, 1995. DOI 10.1007/BF00202785 |
| Part II. Movement time and control | R. Plamondon | Biol. Cybern. 72(4):309–320, 1995. DOI 10.1007/BF00202786 |
| Part III. Kinetic outcomes | Plamondon et al. | DOI 10.1007/s004220050420 |
| Part IV. Formal mathematical proof | Plamondon et al. | DOI 10.1007/s00422-003-0407-9 |
| Development of a Sigma-Lognormal representation for on-line signatures | O'Reilly & Plamondon | Pattern Recognition, 2009. DOI 10.1016/j.patcog.2008.10.005 |
| The lognormal handwriter: learning, performing, declining | Plamondon, O'Reilly, Rémi, Duval | Front. Psychol., 2013 (PMC3867641) |

**용어 주의**: *Delta-Lognormal* (ΔΛ, 1995 원본) 은 두 lognormal 의 **차** (주동근-길항근). *Sigma-Lognormal* (ΣΛ) 은 획마다 하나의 lognormal 을 **벡터 합**. **우리가 쓸 것은 ΣΛ.**

### 1.2 핵심 수식 `[검증 — 해석적 재유도 + 수치 확인]`

> 웹 검색 결과에는 앞에 마이너스 부호가 붙고 지수가 양수인 **틀린 형태**가 돌아왔음. 아래가 올바른 형태.

**획 하나의 속력** — lognormal PDF 그 자체 (적분 = 1):

$$\Lambda(t; t_0,\mu,\sigma) = \frac{1}{\sigma\sqrt{2\pi}\,(t-t_0)}\exp\left(-\frac{(\ln(t-t_0)-\mu)^2}{2\sigma^2}\right),\quad t>t_0$$

$$\|\mathbf v_i(t)\| = D_i\,\Lambda(t;t_{0i},\mu_i,\sigma_i)$$

따라서 $\int\|\mathbf v_i\|dt = D_i$ — **$D_i$ 는 획의 호 길이 그 자체.**

**방향** — 각 획은 원호. lognormal **CDF** 가 각도를 $\theta_s$ 에서 $\theta_e$ 로 이동시킴:

$$\Phi_i(t)=\frac12\left[1+\operatorname{erf}\left(\frac{\ln(t-t_{0i})-\mu_i}{\sigma_i\sqrt2}\right)\right],\qquad \theta_i(t)=\theta_{si}+(\theta_{ei}-\theta_{si})\Phi_i(t)$$

**위치 — 닫힌 형태** (획별 수치적분 불필요). 호 반지름 $r_i = D_i/(\theta_{ei}-\theta_{si})$:

$$\mathbf p_i(t)=\frac{D_i}{\theta_{ei}-\theta_{si}}\begin{bmatrix}\sin\theta_i(t)-\sin\theta_{si}\\ -\cos\theta_i(t)+\cos\theta_{si}\end{bmatrix}$$

$\theta_{ei}\to\theta_{si}$ 일 때 발산하므로 직선으로 특수처리: $D_i\Phi_i(t)[\cos\theta_s,\sin\theta_s]^T$

**Σ — 획 합성**:

$$\mathbf v(t)=\sum_{i=1}^N D_i\Lambda(t;t_{0i},\mu_i,\sigma_i)\begin{bmatrix}\cos\theta_i(t)\\\sin\theta_i(t)\end{bmatrix},\qquad \mathbf p(t)=\mathbf p_0+\sum_i \mathbf p_i(t)$$

**이 중첩이 전부다.** 획들이 **시간축에서 겹친다** — 획 $i$ 가 끝나기 전에 획 $i+1$ 이 발화. 이 겹침이 이산적 원호들을 매끄러운 흘림체로 바꾼다. 겹침 0 = 로봇 같은 분절 동작, 겹침 큼 = 빠르고 둥근 흘림. **겹침 정도가 가장 표현력 있는 스타일 노브 중 하나.**

**해석적 특징점** `[검증 — 직접 유도]`. $w=(\ln(t-t_0)-\mu)/\sigma$ 치환 시:

| 점 | 의미 | $w$ |
|---|---|---|
| $p_3$ | 최대 속력 | $-\sigma$ |
| $p_2$ | 1차 변곡 (최대 전) | $\frac{-3\sigma-\sqrt{\sigma^2+4}}{2}$ |
| $p_4$ | 2차 변곡 (최대 후) | $\frac{-3\sigma+\sqrt{\sigma^2+4}}{2}$ |

$$t_{\max}=t_0+e^{\mu-\sigma^2},\qquad \|\mathbf v\|_{\max}=\frac{D}{\sigma\sqrt{2\pi}}e^{\sigma^2/2-\mu}$$

### 1.3 파라미터의 물리적 의미 — **이 프로젝트에서 가장 중요한 표**

| 파라미터 | 물리적 의미 | 분류 |
|---|---|---|
| $D_i$ | 운동 명령의 진폭 = 획의 호 길이 | **내용** (action plan) |
| $\theta_{si},\theta_{ei}$ | 원호의 시작/끝 방향. 차이 = 중심각 = 곡률 | **내용** (action plan) |
| $t_{0i}$ | 중추신경계가 명령을 내린 시각 (움직임 관찰 전, 음수 가능) | **타이밍** → 획 겹침 |
| $\mu_i$ | 로그 시간지연 — 신경근계 응답 지연의 로그. 작을수록 빠른 반응 | **스타일** (개인 고유) |
| $\sigma_i$ | 로그 응답시간 — 속도 종모양의 **비대칭/왜도**. 클수록 긴 꼬리 | **스타일** (개인 고유) |

> **$(\mu,\sigma)$ 는 생물학적·개인 고유 파라미터로 비교적 안정적. $(D,\theta_s,\theta_e)$ 는 "무엇을 쓰는가". $t_0$ 는 시퀀싱.**
> 이 분리가 우리가 원하는 "스타일 vs 내용" 분해를 **이론적으로 정당화**해준다. 자의적 설계가 아님.

$\sigma$ 는 피로·노화·신경운동 장애에서 강하게 변동 (파킨슨 진단에 활용됨).

### 1.4 실제 궤적에서 파라미터 추출 `[검증 — 코드 확인 + 재유도]`

**Robust XZERO / INFLEX** (Djioua & Plamondon) 계열:

1. **전처리** — 균일 $\Delta t$ 리샘플, 저역통과, $\mathbf v(t)$·속력·각도 계산
2. **특징점 탐색** — 속력 프로파일에서 `극소 → 변곡 → 극대 → 변곡 → 극소` 패턴 → $[p_1..p_5]$
3. **$\sigma$ 닫힌 해.** $L=\ln(v(p_a)/v(p_b))$ 일 때:
   $$\sigma^2_{(2,3)}=-2-2L-\tfrac1{2L},\quad \sigma^2_{(2,4)}=-2+2\sqrt{L^2+1},\quad \sigma^2_{(3,4)}=-2+2L+\tfrac1{2L}$$
   (세 식 모두 대수적으로 검증. $\sigma=0.3$ 대입 시 $\sigma^2=0.0900$ 복원 확인)
4. **$\mu,t_0$** — $a(p)=e^{\sigma w_p}$ 로 두면 $e^\mu = \frac{t_{p_a}-t_{p_b}}{a(p_a)-a(p_b)}$, $t_0=t_p-e^\mu a(p)$ ($p_3$ 권장)
5. **$D$** — 속력식을 $p_3$ 에서 역산
6. **각도** — $\Delta\theta=\frac{\theta(p_2)-\theta(p_4)}{\Phi(p_2)-\Phi(p_4)}$ 후 외삽. $\Phi(p_1)=0,\Phi(p_5)=1$ 하드코딩이 흔한 안정화
7. **인간 타당성 게이트** — 구현체에서 확인된 실측 범위:
   - $p_2$: 속력비 0.44–0.54, 최대속력 **30–140 ms 전**
   - $p_4$: 속력비 0.66–0.74, 최대속력 **25–130 ms 후**
8. **잔차 반복** — 맞춘 획의 속도를 빼고 재추출, SNR 포화까지

**추출기 2종**
- **ScriptStudio** (Plamondon 그룹, Polytechnique Montréal) — 레퍼런스 플랫폼. **속도만** 피팅. 공개 배포 확인 안 됨. `.ana` 출력 포맷. `[색인]`
- **iDeLog** — Ferrer, Diaz, Carmona-Duarte, Plamondon, **IEEE TPAMI 42(1):114–125, 2020**, DOI 10.1109/TPAMI.2018.2879312. **ScriptStudio 는 속도만 맞추고 궤적이 흘러가는 문제**가 있는데, iDeLog 는 궤적과 속도를 **동시에** 맞춤 — 궤적에서 action plan(가상 목표점)을, 속도에서 lognormal 열을 뽑고, 가상 목표점을 반복 이동시켜 양쪽 오차를 함께 최소화. **공간 충실도가 중요하면 이쪽** (= 읽을 수 있는 글자를 만들려면 필수)

### 1.5 ΣΛ 기반 합성 데이터 생성 — 우리 용도의 직접 선행연구 `[색인]`

- **Galbally, Plamondon, Fierrez, Ortega-Garcia**, "Synthetic on-line signature generation"
  - Part I: Pattern Recognition 45(7):2610–2621, 2012. DOI 10.1016/j.patcog.2011.12.011
  - Part II: 45(7):2622–2632, 2012. DOI 10.1016/j.patcog.2011.12.007

**"Duplicated signatures" 메커니즘 — 우리 프로젝트의 핵심 레시피**. 두 가지 섭동 전략:
1. **획 단위** — 각 획의 $(D,t_0,\mu,\sigma,\theta_s,\theta_e)$ 에 가우시안 노이즈. 단 **파라미터별 민감도가 크게 다름** — $\sigma$ 와 $\theta$ 가 $D$ 보다 훨씬 민감
2. **목표점 단위** — action plan 의 가상 목표점을 섭동 후 lognormal 재계산. **가독성 보존에 훨씬 유리** (동역학이 아니라 기하를 건드리므로)

추출된 ΣΛ 파라미터 집합이 **필기자 운동 프로그램의 템플릿** 역할을 하며, 그 주변을 샘플링하면 일반적 노이즈가 아닌 **진짜 개인 내 변동성**이 재현됨.

**서베이 앵커**: Diaz, Ferrer, Impedovo, Malik, Pirlo, Plamondon, "A Perspective Analysis of Handwritten Signature Technology", **ACM Computing Surveys 51(6) Art.117, 2019**, DOI 10.1145/3274658. **여기서 시작할 것.**

관련: "A sigma-lognormal model for handwritten **text** CAPTCHA generation" — 서명이 아니라 **텍스트**라는 점에서 우리와 가까움 `[색인]`

### 1.6 온라인 vs 오프라인 — **계획에 결정적인 제약**

> **ΣΛ 는 근본적으로 온라인(시간 정보 있는) 데이터를 요구한다.** 속도 모델이므로, 스캔 이미지에는 속도가 없다.

세 가지 우회로 (우리에게 유용한 순):

1. **온라인 생성 → 오프라인 렌더링 (최선)**. ΣΛ 궤적을 합성한 뒤 **잉크 침착 모델**로 래스터화.
   Ferrer, Diaz-Cabrera, Morales, Galbally, Gomez-Barrero, "Realistic synthetic off-line signature generation based on synthetic on-line data", **ICCST 2013**, DOI 10.1109/CCST.2013.6922041. 볼펜 잉크 침착 모델 사용, 실제 데이터와 개인 간/내 변동성이 일치한다고 보고. `[색인]`
   **정답을 우리가 완전히 통제한다 — 학습 데이터 생성에 정확히 필요한 성질.**
2. **이미지에서 궤적 복원 후 ΣΛ 피팅**. 세선화 → 획 순서 추정 → 펜 경로 복원 → 속도 프로파일 부여. 손실이 크고 오류가 잦음, 복원된 타이밍은 추측.
3. **오프라인 파라미터 공간 증강**. "Intrapersonal Parameter Optimization for Offline Handwritten Signature Augmentation", arXiv 2010.06663 `[색인, 열지 못함]`

### 1.7 오픈소스 구현

| 저장소 | 등급 | 비고 |
|---|---|---|
| `github.com/gpds-ulpgc/iDeLog` | `[검증]` README 200 | **MATLAB**, 논문 저자 그룹(ULPGC). "ScriptStudio 의 대안" 자칭. **RAR 안의 `.p` 난독화 파일** — 소스 안 읽힘. "G1 fitting with clothoids" 툴박스 필요. **라이선스 명시 없음 → gpds@gi.ulpgc.es 문의 전까지 연구용으로만 취급** |
| `github.com/andrew-healey/sigma-lognormal` | `[검증]` `README.md`, `lognormal.py`, `action_plan.py`, `speed_extract.py` 전문 확인 | **Python/NumPy/SciPy**, ~12 모듈. "최초 공개 구현" 자칭. **별 5개, 포크 0, 라이선스 파일 없음 — 취미 수준.** 그러나 **수식은 정확** (직접 재유도로 확인). 위 §1.2/§1.4 전부 이 소스로 교차검증함. **가장 읽을 만한 참조 구현** |

> **프로덕션급·허용적 라이선스의 ΣΛ 라이브러리는 없다. 직접 구현해야 함** — 다만 §1.2/§1.4 가 사실상 완결이라 수백 줄 수준.

---

## 2. 비신경망 대안 모델

### 2.1 진동자 모델
- **Hollerbach, "An oscillation theory of handwriting", Biol. Cybern. 39:139–156, 1981** `[색인]`. 직교하는 두 질량-스프링 진동자(손목=수직, 손가락=수평) + 좌→우 등속 평행이동. **약점: 궤적에서 파라미터 추출 불가, x/y 비대칭.**
- **André, Kostrubiec, Buisson, Albaret, Zanone, "A parsimonious oscillatory model of handwriting" (POMH), Biol. Cybern. 108(3):321–336, 2014**, DOI 10.1007/s00422-014-0600-z `[색인]`. **위 두 약점을 모두 수정한 현대판** — 궤적에서 파라미터 추출 가능, x/y 대칭 회복. 아동 난필증 진단에 적용됨.

> **ΣΛ vs 진동자**: ΣΛ 는 *이산 사건* (탄도적 명령의 열), 진동자는 *연속 리듬*. **숫자는 펜 들림이 있는 소수의 이산 획으로 쓰므로 ΣΛ 가 적합.** 반복적 흘림체에는 진동자가 유리.

### 2.2 절차적 스켈레톤 섭동 `[색인/일반지식]`
`글리프 스켈레톤 → 인스턴스별 파라미터 재추첨 → 가변폭 브러시 렌더링`

반복적으로 등장하는 스타일 파라미터: slant(전단), x-height/cap-height 비, 획 굵기 + 필압 연동, 제어점 곡률 이득, 연결(ligature) 규칙, baseline drift(저주파 흔들림), tremor(고주파 — 고령/장애 필기의 지배적 사실감 단서), 문자·단어 간격 지터, 문자별 회전.

> **구현 요점**: 이 값들을 문서 단위가 아니라 **문자 인스턴스마다** 재추첨할 것. 인접 문자가 상관되도록 **Perlin/simplex 노이즈** 사용이 일반적. 안 그러면 페이지의 모든 "5"가 똑같아지고, **이것이 합성 손글씨의 가장 명백한 티**다.

---

## 3. 실무에서 쓰이는 생성기

| 도구 | 등급 | 방식 |
|---|---|---|
| `Belval/TextRecognitionDataGenerator` (TRDG) | `[검증]` PyPI `trdg` **1.8.0, MIT** | **표준 주력 도구.** 폰트 렌더 + 왜곡(기울임/블러/배경). `-hw` 로 손글씨 모드(Grzego 모델). 비라틴 지원. 파이썬 모듈로 사용 가능 |
| `clovaai/synthtiger` | `[검증]` README 200 | **네이버 클로바.** ECCV/ICDAR 계열. **클로바는 한국 조직 — 한글 지원 가능성이 가장 높음. 한글용으로 먼저 평가할 것** |
| `sjvasquez/handwriting-synthesis` | `[검증]` | **별 4.8k, 포크 674 — 가장 널리 쓰임.** Graves(2013) 의 TF 구현. **SVG 출력, 사전학습 모델 포함**, `bias`(단정함)·`style`(필기자) 노출 |
| `IBM/tensorflow-hangul-recognition` | `[검증]` README 200 | 한글 인식. **폰트 기반 한글 이미지 생성기 포함** — 한글 렌더링 출발점으로 유용 |
| `koninik/awesome-handwritten-text-generation` | `[검증]` 2025-12-05 갱신 | HTG 큐레이션. GAN(ScrabbleGAN, GANwriting, VATr) / Diffusion(WordStylist, DiffusionPen, One-DM) / AR(Emuru) / VAE 분류 |

**현재 실제 사용 빈도 순**: (1) 폰트+왜곡 — OCR 학습데이터의 현실적 다수, (2) diffusion — 2023–25 사실감 SOTA, (3) GAN — 밀려나는 중, (4) 획/궤적 모델(ΣΛ, RNN) — 소수파이나 **해석 가능한 파라미터를 주는 유일한 부류**

**한글**: `[색인]` **PHD08** 이 표준 한국어 문자 DB — 2,350 클래스 × 2,187 = 5,139,450 이미지. **단 "P"는 Printed — 인쇄체이지 손글씨가 아님** (중요한 단서). GAN 증강 한글 연구에서 CNN 정확도 92.45% → 96.86% 보고 `[색인]`.

> **한글 + ΣΛ 의 구조적 이점**: 한글은 조합형 — 초성 19 + 중성 21 + 종성 28 이 11,172 음절을 만든다. **음절이 아니라 자모 단위로 ΣΛ 획을 모델링해야 한다.** 약 68개 자모 획 집합이 전체를 커버. 라틴 흘림체보다 우리 접근이 **훨씬 유리**하며, **정확히 이걸 한 발표 연구를 찾지 못함 — 빈 구멍으로 보임.**

---

## 4. 금액 인식과 Prior

### 4.1 정전 문헌 `[색인]`
**1997년 IJPRAI 수표 처리 특집호**가 이 분야의 분기점:
- "Check Amount Recognition Based on the Cross Validation of Courtesy and Legal Amount Fields", DOI 10.1142/S0218001497000275
- "Bankcheck Recognition using Cross Validation Between Legal and Courtesy Amounts", DOI 10.1142/S0218001497000287
- "Handwritten Bank Check Recognition of Courtesy Amounts", IJIG, DOI 10.1142/S0219467804001373

> 참고: 수표 판독이 1990년대 AT&T 벨연구소에서 **LeCun 의 CNN 연구(LeNet, MNIST)를 견인한 응용**이다. MNIST 는 말 그대로 이 문제 때문에 존재한다.

### 4.2 CAR/LAR 교차검증이 prior 로 작동하는 방식
- **Courtesy amount (CAR)** = 숫자 필드 (`₩1,250,000`). **Legal amount (LAR)** = 문자 표기 (`일백이십오만원`). **같은 값을 두 개의 독립 채널로 인코딩.**
- 각 인식기가 **n-best + 신뢰도**를 내고, 두 리스트를 교집합 후 **결합 신뢰도로 재순위**
- 이득이 비대칭적으로 큼: **LAR 단독 정확도는 낮지만**, 그 오류가 CAR 오류와 **통계적으로 독립**이므로 둘의 일치는 매우 강한 증거. 문헌 보고치: **1% 오류율에서 ~67% 인식, reject 없이 ~85%**
- 실무적으로: **불일치 → 사람에게 라우팅.** 이 reject 옵션이 은행 운영의 핵심 — 시스템의 가치는 **모를 때 모른다고 아는 것**에 있다

### 4.3 어휘·문법 제약 인식 `[색인]`
- "Lexicon-driven HMM decoding for large vocabulary handwriting recognition", IJDAR, DOI 10.1007/s10032-003-0113-0. 문자=HMM 을 연결해 단어 모델 구성, **어휘를 prefix tree** 로 두어 접두사 공유. LDLBA 로 디코딩, 레벨 간 bigram 이 단어 내 스타일 변동 포착. 팩스 문서에서 상용 패키지 대비 **오류 1/4**
- "Lexicon-driven segmentation and recognition of handwritten character strings for **Japanese address reading**" — **CJK + 닫힌 필드 어휘**라는 점에서 한국어 금융 필드와 가장 가까운 공개 유사 사례

> **핵심 원리**: 필드 타입이 알려지면 탐색 공간이 급격히 붕괴한다. 한국 통화 금액 필드는 **정규 언어**이므로 DFA 를 손으로 쓸 수 있고, 인식은 "DFA 내부에서 최선 경로 찾기"가 되어 개방 어휘 디코딩보다 훨씬 쉬워진다.

### 4.4 현대 VLM 의 constrained decoding `[색인 + 광범위 교차확인]`
디코딩 매 스텝에서, 문법(regex/CFG/JSON schema)을 위반하게 되는 토큰의 logit 을 $-\infty$ 로 마스킹.

| 백엔드 | 비고 |
|---|---|
| **Outlines** (Willard & Louf, 2023) | FSM 기반, 어휘 인덱스 구축. 가장 널리 쓰임 |
| **XGrammar** | vLLM/SGLang 의 문법 백엔드 |
| **llguidance** | vLLM·SGLang 양쪽에서 XGrammar 를 일관되게 상회한다고 보고. **JSONSchemaBench** (2025-01, 실사용 스키마 ~10k, 6개 프레임워크) 에서 **비제약 생성보다도 토큰당 지연이 낮음** |

vLLM·SGLang 모두 choice/JSON schema/regex/문법 지원. 금융 필드에는 **regex 가 최적점** — DFA 는 선형 시간이라 추론 비용이 사실상 0. 금액 필드를 `^[0-9]{1,3}(,[0-9]{3})*$` 로 묶으면 **형식 오류를 낼 수 없다.**

> ⚠️ **중요한 반대 증거**: constrained decoding 의 부작용에 대한 2025–26 연구가 진행 중 — "From Hallucination to Structure Snowballing: The Alignment Tax of Constrained Decoding" (arXiv 2604.06066), "AdapTrack: Constrained Decoding without Distorting LLM's Output Intent" (arXiv 2510.17376) `[둘 다 미확인 — 열지 못함]`.
> 우려: 유효한 형식을 강제하면 정직한 "못 읽겠다"가 **자신 있게 형식만 맞는 오답**으로 바뀐다. **은행 응용에서는 실제 위험.**
> 완화: 문법에 **명시적 `illegible`/`null` 분기**를 넣어 모델이 합법적으로 기권할 길을 열어두고, constrained logprob 을 신뢰도 신호로 삼아 reject 옵션에 연결. — **1997년 CAR/LAR reject 로직을 그대로 현대로 이식하는 것.**

---

## 5. 이 프로젝트에 대한 결론

1. **ΣΛ 가 해석 가능한 스타일 파라미터화의 정답**이고, 수식은 §1.2/§1.4 로 이미 손에 있다. 직접 구현 (수백 줄 NumPy). `andrew-healey/sigma-lognormal` 을 읽기용 참조로, iDeLog 논문을 알고리즘 목표로.
2. **스타일 벡터를 모델이 나누는 방식 그대로 나눌 것**: 필기자 고유 $(\mu,\sigma)$ + 획 겹침 통계 = **스타일**, $(D,\theta_s,\theta_e)$ action plan = **내용**. 여기에 §2.2 의 전역 절차적 파라미터(slant, baseline drift, tremor, 획 굵기)를 얹으면 싸게 사실감을 얻는다.
3. **"온라인 생성 → 잉크 침착 모델로 오프라인 렌더링" 파이프라인**을 쓸 것. 스캔 이미지에서 ΣΛ 를 추출하려 하지 말고, 대상 필기자에게서 소량의 펜 태블릿 데이터를 받는 편이 낫다. ΣΛ 의 강점이 **극소량 실제 샘플**로 동작한다는 것이다 — duplicated-signature 문헌은 **피험자당 실제 1장**에서 작동한다.
4. **한글은 자모 단위로 모델링.** 가용한 선택지 중 레버리지가 가장 크고, 발표된 선행연구가 없다.
5. **숫자가 쉬운 승리** — 10 클래스, 획 1–3개, 닫힌 형태 원호 합성으로 깔끔히 처리. **여기서 시작해 검증 후 한글로.**
6. **인식 측은 두 prior 를 함께**: 디코딩 시 필드 타입 regex/DFA 제약 + CAR/LAR 식 독립 채널 교차검증 + 명시적 reject. 1997년의 통찰 — 가치는 **언제 사람에게 넘길지 아는 것** — 은 변하지 않았다.

---

## 6. 직접 확인이 필요한 항목

- 이 문서의 **모든 arXiv ID** (하나도 열지 못함)
- ScriptStudio 배포/라이선스 여부
- iDeLog 라이선스 (gpds@gi.ulpgc.es 문의)
- `clovaai/synthtiger` 가 한글 에셋을 실제로 포함하는지, 지원만 하는지
