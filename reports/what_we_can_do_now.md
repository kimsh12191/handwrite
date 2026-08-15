# 지금 바로 가능한 범위

## A. 한글 geometry
PE92 + SERI95 test subset부터 분석한다.

1차 질문:
- 동일 음절 내 aspect/rotation/asymmetry 분산이 얼마나 큰가
- hole/component topology가 몇 가지 family로 갈리는가
- 현재 renderer의 scale/slant/bend만으로 PCA top axis를 설명 가능한가

## B. 숫자/영문 geometry
EMNIST ByClass에서 class당 200개를 먼저 추출한다.

1차 질문:
- 0/O/o처럼 loop 계열에 rotation/asymmetry/closure가 얼마나 필요한가
- A/K/M/W 같은 angular glyph의 segment ratio 자유도가 얼마나 필요한가

## C. 숫자/영문 motor
UJI v2에서 writer별:
- stroke count
- point count
- path length
- bounding aspect
- stroke length distribution
을 추출한다.

UJI는 timing/pressure가 없으므로 velocity model 학습용이 아니라 **stroke topology prior**로 쓴다.

## D. 한국어 motor
AI-Hub 71307이 들어오면 별도 단계로 연다.
공식 schema에는 실제 velocity/acceleration/pressure가 있으므로 UJI보다 훨씬 강하다.

## E. 한글 font stress test
Clova 109종 TTF를 투입하면 같은 probe 음절을 109 style로 렌더하여
'실제 handwriting + generated font style'의 합집합으로 geometry stress test를 한다.

폰트는 실제 손동작 ground truth가 아니므로 motor 분석에는 사용하지 않는다.
