# 지금 바로 가능한 범위

이 문서는 bootstrap 시점의 계획이었다. 실제로 무엇이 되고 무엇이 막혔는지를
각 항목 아래에 적는다. 측정 결과 본문은 `phase2_findings.md`다.

## A. 한글 geometry
PE92 + SERI95 test subset부터 분석한다.

1차 질문:
- 동일 음절 내 aspect/rotation/asymmetry 분산이 얼마나 큰가
- hole/component topology가 몇 가지 family로 갈리는가
- 현재 renderer의 scale/slant/bend만으로 PCA top axis를 설명 가능한가

**상태: 완료. 단 test subset이 아니라 PE92 train + SERI95 test로 했다.**
PE92 test는 클래스당 ~10장이라 클래스별 PCA를 하나도 통과하지 못한다.

세 번째 질문의 답이 이번 단계의 결론이다. **설명 불가능하다.**
scale/slant 류의 전역 변형은 같은 척도에서 4~5성분으로 측정되는데
실제 클래스 내부 변동은 28성분이다 (`phase2_findings.md` 2·3절).

첫 번째 질문은 다시 물어야 한다. 요약통계 PCA에서는 축이 갈리지 않아서
"aspect 분산이 얼마"라는 형태의 답이 나오지 않는다 (1절).

## B. 숫자/영문 geometry
EMNIST ByClass에서 class당 200개를 먼저 추출한다.

1차 질문:
- 0/O/o처럼 loop 계열에 rotation/asymmetry/closure가 얼마나 필요한가
- A/K/M/W 같은 angular glyph의 segment ratio 자유도가 얼마나 필요한가

**상태: 미착수.** `biometrics.nist.gov`가 이 환경의 egress 정책에 막혀
EMNIST를 받지 못했다. 스크립트(`extract_emnist.py`)는 그대로 있고,
데이터가 들어오면 `bootstrap_after_download.py`가 자동으로 이 단계를 켠다.

## C. 숫자/영문 motor
UJI v2에서 writer별:
- stroke count
- point count
- path length
- bounding aspect
- stroke length distribution
을 추출한다.

UJI는 timing/pressure가 없으므로 velocity model 학습용이 아니라 **stroke topology prior**로 쓴다.

**상태: 미착수.** `archive.ics.uci.edu`가 막혀 있다.
UJI는 별도의 이유로도 중요하다 — **writer id가 있는 유일한 확보 후보 데이터**다.
PE92·SERI95에는 writer id가 없어서 `θ_writer`의 재현성을 현재 검증할 수 없다
(`taxonomy_v1.md` 4절).

## D. 한국어 motor
AI-Hub 71307이 들어오면 별도 단계로 연다.
공식 schema에는 실제 velocity/acceleration/pressure가 있으므로 UJI보다 훨씬 강하다.

**상태: 변동 없음. 승인 대기.**

## E. 한글 font stress test
Clova 109종 TTF를 투입하면 같은 probe 음절을 109 style로 렌더하여
'실제 handwriting + generated font style'의 합집합으로 geometry stress test를 한다.

폰트는 실제 손동작 ground truth가 아니므로 motor 분석에는 사용하지 않는다.

**상태: 미착수.** Clova는 원래 수동이고, 자동 대상이던 Nanum 번들도
`cdn.naver.com`이 막혀 미확보다.

폰트가 들어오면 할 일이 하나 늘었다. 폰트 글리프를 `shape_space.py`에 통과시키면
**같은 척도에서 폰트가 몇 성분으로 읽히는지**를 얻는다. 실제 손글씨의 28과
비교하면 "폰트가 못 덮는 폭"이 수치가 된다. 폰트는 motor ground truth로는
쓸 수 없지만 이 비교에는 쓸 수 있다.
