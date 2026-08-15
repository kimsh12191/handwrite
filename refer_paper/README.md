# 선행연구 정리

## 문서

| 파일 | 다루는 것 | 상태 |
|---|---|---|
| `sigma_lognormal_and_motor.md` | ΣΛ(Sigma-Lognormal) 운동학 모델, 파라미터 추출 알고리즘, 진동자 모델, 실무 생성기, 금액 인식 prior | 이전 브랜치에서 복원. 자체 검증 등급 표기 있음 |
| `shape_parameterization.md` | 형태(offline) 파라미터화 계열 — PDM/ASM, BPL, 폰트 다양체, 한글 조합형 생성 | 이 문서 아래 요약 |

`sigma_lognormal_and_motor.md`는 저장소 초기화 때 사라졌던 것을 git 이력에서
되살렸다. `reports/taxonomy_v2.md`의 L7(모터) 층이 이 문서에 의존한다.
문서 자체가 `[검증]/[색인]/[미확인]` 등급을 달고 있으므로 인용 전에 등급을 볼 것.

---

# 형태 파라미터화 계열

ΣΛ는 **온라인(시간 정보 있는) 궤적**을 요구한다. 현재 손에 있는 PE92·SERI95는
offline 이미지뿐이라 ΣΛ를 바로 적용할 수 없다. 지금 쓸 수 있는 것은 아래 계열이다.

## 1. 통계적 형상 모델 (PDM / Active Shape Model) — 지금 바로 쓸 것

Cootes, Taylor, Cooper, Graham, "Active Shape Models — Their Training and
Application", CVIU 61(1):38–59, 1995.

절차:

```text
landmark 지정 → 샘플 간 점 대응 → Procrustes 정렬 → PCA → 형상 파라미터 b
```

형상 `x = x̄ + Σ b_k φ_k`. `b`가 곧 해석 가능한 형태 파라미터이고, 각 모드의
분산 `λ_k`가 그 축이 실제로 얼마나 쓰이는지를 알려준다.

**이 프로젝트에 중요한 이유**: HANDOFF Phase 2·3의 `feature → PCA → 파라미터 축`
절차가 사실 이 레시피인데, **점 대응 단계가 빠져 있었다.** 대응 없이 PCA를 돌리면
분산이 형태가 아니라 어긋남(misregistration)에 쓰인다. 이 저장소에서 그 증상을
실제로 측정했다 — `reports/phase2_findings.md` 2절.

- [An Introduction to Active Shape Models (Cootes)](https://personalpages.manchester.ac.uk/staff/timothy.f.cootes/papers/asm_overview.pdf)
- [A Brief Introduction to Statistical Shape Analysis](https://graphics.stanford.edu/courses/cs164-09-spring/Handouts/paper_shape_spaces_imm403.pdf)

관련: Procrustes 분석, Dryden & Mardia "Statistical Shape Analysis";
얼굴의 morphable model(Blanz & Vetter 1999)도 같은 구조 — **조밀 대응 후 PCA**.

## 2. 확률적 프로그램 (BPL) — 구조가 가장 닮음

Lake, Salakhutdinov, Tenenbaum, "Human-level concept learning through
probabilistic program induction", Science 350(6266):1332–1338, 2015.

문자를 **프로그램**으로 본다.

```text
문자 = 획들의 집합
획   = sub-stroke 열 (각각 스플라인)
관계 = 이 획이 앞 획의 어디에 붙는가
       (independent / start / end / along at τ)
```

- 조합성(compositionality) — 원형에서 조립
- 인과성(causality) — 어떻게 그려졌는지를 모델링
- 한 장에서 학습 가능

**이 프로젝트에 중요한 이유**: HANDOFF가 목표로 한 계층
(`glyph → jamo → stroke → motor`)과 구조가 거의 같다. 특히 **관계(relation)** 가
§5의 context-dependent deformation을 표현하는 정공법이다.
`taxonomy_v2.md`의 L2가 여기서 왔다.

- [Science 논문](https://www.semanticscholar.org/paper/Human-level-concept-learning-through-probabilistic-Lake-Salakhutdinov/815c84ab906e43f3e6322f2ca3fd5e1360c64285)
- [brendenlake/BPL 구현](https://github.com/brendenlake/BPL)
- [Omniglot challenge 3-year report](https://www.cs.princeton.edu/~bl8144/papers/LakeEtAlOmniglotProgress.pdf)

Omniglot 자체도 유용하다 — MNIST의 전치(클래스 많고 샘플 적음)이며 **온라인 획
데이터**를 포함한다. 라틴 alphabet의 획 구조를 볼 수 있다.

## 3. 파라메트릭 폰트 — 설계자 정의 축 (데이터 기반 아님)

- **METAFONT** (Knuth) — 펜 기반 파라메트릭 글리프 기술
- **Variable fonts / Multiple Masters** — 설계자가 정한 축(weight, width, slant)

축이 사람이 정한 것이라 개인 필체 동일성을 담기엔 축이 너무 적다(보통 2~5개).
**참고용이지 답이 아니다.**

## 4. 한글 조합형 생성 — 구조는 참고, 방법은 아님

한글의 초성/중성/종성 조합성을 명시적으로 쓰는 few-shot 폰트 생성 연구가 있다.

| 모델 | 출처 |
|---|---|
| DM-Font | [Few-shot Compositional Font Generation with Dual Memory, ECCV 2020](https://arxiv.org/pdf/2005.10510) |
| LF-Font | [Few-shot Font Generation with Localized Style Representations and Factorization, AAAI 2021](https://arxiv.org/pdf/2009.11042) |
| MX-Font | ICCV 2021 |
| 구현 모음 | [clovaai/fewshot-font-generation](https://github.com/clovaai/fewshot-font-generation) (GitHub은 이 세션에서 접근 가능) |

DM-Font는 한글처럼 **완전 조합형 스크립트**에 한정해 컴포넌트별 지역 특징을
메모리에 저장하고 조합한다.

**한계**: 신경망이라 파라미터가 해석 불가능하다. HANDOFF의 "해석 가능한
parametric renderer" 목표에 그대로 쓸 수 없다. **참고할 것은 컴포넌트 분해
구조이지 생성 방법이 아니다.** 또한 폰트 데이터라 실제 손동작 ground truth도 아니다.

## 5. 계열 비교

| 계열 | 파라미터 | 필요 데이터 | 해석성 | 지금 가능? |
|---|---|---|---|---|
| PDM / ASM | landmark PCA 계수 | offline + **대응** | 중 (모드 관찰 후) | **가능** |
| BPL | 획·sub-stroke 제어점·관계 | online 획 | 높음 | Omniglot으로 부분 가능 |
| ΣΛ | D, t₀, μ, σ, θs, θe / 획 | online 궤적 + 시간 | 높음 (생리학적) | AI-Hub 대기 |
| 진동자 | 진폭·주파수·위상 | online | 중 | 대기 |
| 신경망 폰트 생성 | 잠재 벡터 | 폰트 | 없음 | 구조만 참고 |

## 6. 확인 필요

`sigma_lognormal_and_motor.md` §6의 미확인 항목에 더해:

- ASM을 손글씨 골격에 적용한 사례 (얼굴·의료영상 사례는 많으나 문자 사례 미확인)
- CJK 문자 offline 이미지에서 획 추출·대응을 다룬 문헌 (존재할 것으로 보이나 미조사)
- Omniglot 한글 포함 여부 및 획 데이터 형식
