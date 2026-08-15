# 먼저 읽을 문서

- **Claude / Coding Agent:** `HANDOFF.md`
- **사용자:** `USER_GUIDE.md`

`HANDOFF.md`에는 프로젝트의 핵심 의도, 우선순위, 금지사항, renderer 설계 원칙이 정리되어 있다.
이 프로젝트를 다른 에이전트에게 넘길 때는 ZIP 전체와 함께 `HANDOFF.md`를 먼저 읽도록 지시한다.

진행 상황은 `HANDOFF.md` §15에 있다. 측정 결과는 다음 둘이다.

- `reports/phase2_findings.md` — 실제 variation 측정 결과와 그것이 바꾸는 결정
- `reports/taxonomy_v1.md` — 근거 있는 축만 넣은 parameter taxonomy 초안

---

# Renderer Space Public Bootstrap v2

목적: **승인 없이 공개 다운로드 가능한 데이터부터** renderer 표현 공간 분석을 시작한다.

## 1. 자동 수집 대상

```bash
pip install -r requirements.txt
python scripts/download_public.py
python scripts/bootstrap_after_download.py
```

자동 다운로드 대상으로 등록한 것:

- HangulDB 전체 (PE92 / SERI95 / HanDB) — `git clone`
- UJI Pen Characters v2
- EMNIST
- Naver Nanum legacy font bundle (NanumPen / NanumBrush 포함)

HangulDB를 파일 단위 URL이 아니라 clone으로 받는 이유:
`PE92_test.zip`은 클래스당 약 10장뿐이라 클래스별 PCA(`--min-n 15`)를 하나도
통과하지 못한다. train split이 있어야 2,350클래스 × 약 81장이 되고, 그것이
저장소 안에 같이 들어 있다.

`download_public.py`는 막힌 호스트를 우회하지 않고 요약에 그대로 보고한다.
Phase 1 중 무엇이 실제로 손에 있는지가 이후 결론의 범위를 결정하기 때문이다.

## 2. 별도 수동 투입

### Clova 나눔손글씨 109종

공식 페이지:
`https://clova.ai/handwriting/list.html`

현재 안정적인 공식 bulk-download URL을 manifest에 고정하지 않았다.
TTF들을 받은 뒤 아래에 넣는다.

```text
data/fonts_manual/
```

그 후:

```bash
python scripts/bootstrap_after_download.py
```

### AI-Hub 71307

승인이 필요한 데이터이므로 자동 수집하지 않는다.

공식 schema상:
- x: 0~756
- y: 0~88
- DOWN / UP / MOVE
- pressure
- velocity
- acceleration
- angle
- curvature
- 1차/2차 difference
- stroke length ratio

데이터를 받으면 별도 motor parser를 연결한다.

## 3. probe set

`make_probe_set.py`는 다음을 만든다.

- 숫자 10
- 영문 대/소문자 52
- 한글 compatibility 자모
- 모든 초성을 동일 문맥(ㅏ)에서 관찰하는 음절
- 모든 중성을 동일 문맥(ㅇ 초성)에서 관찰하는 음절
- 모든 종성을 동일 문맥(ㄱ+ㅏ)에서 관찰하는 음절

즉 한글은 **고립 자모만 보지 않고 초성/중성/종성 역할별 변형까지** 본다.

## 4. 이번 단계에서 구하려는 것

각 글자/음절 sample에서:

- aspect
- centroid
- principal angle
- moment ratio
- 좌우/상하 asymmetry
- compactness
- connected component
- hole count

를 뽑고 문자별 PCA를 한다.

원래 기대한 결과는 이런 모양이었다.

```text
ㅇ 실제 데이터
  PC1 aspect
  PC2 rotation
  PC3 asymmetry
  PC4 closure
```

이런 결과가 나오면 renderer에 그 자유도를 넣는다는 계획이었다.
**실제로는 이렇게 갈리지 않는다.** 이 요약통계 기저에서는 PC1이 22.8%에
그치고 loading이 섞여서 축에 이름이 붙지 않는다.
`reports/phase2_findings.md` 1절과 `USER_GUIDE.md` §7을 볼 것.

## 5. 최종 renderer 설계 규칙

먼저 parameter를 상상해서 만들지 않는다.

```text
공개 handwriting corpus
→ 문자별 variation 분석
→ PCA / clustering
→ 필요한 parameter axis 정의
→ renderer 구현
→ 실제 sample fitting
→ failure cluster 분석
→ renderer space 확장
```

## 주의

HangulDB README는 PE92에 일부 mislabel이 있다고 경고한다.
따라서 PE92는 최종 quantitative benchmark보다 **shape-space 발견용**으로 먼저 쓴다.
