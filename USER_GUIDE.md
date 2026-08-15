# USER_GUIDE.md — 사용자 실행 및 판단 가이드

## 이 ZIP은 무엇인가

이 ZIP은 완성된 손글씨 renderer가 아니다.

현재 목적은:

> **실제 손글씨 데이터를 모아서, renderer가 어떤 모양과 움직임을 표현할 수 있어야 하는지를 먼저 측정하는 bootstrap 프로젝트**

다.

따라서 지금 결과물이 바로 예쁜 손글씨를 생성하지 않아도 정상이다.

---

# 1. 가장 빠른 실행

인터넷이 되는 Python 환경에서:

```bash
pip install -r requirements.txt

python scripts/download_public.py
python scripts/bootstrap_after_download.py
```

자동 수집 대상으로 등록된 공개 데이터:

- PE92 test
- SERI95 test
- UJI Pen Characters v2
- EMNIST
- Naver Nanum legacy font bundle

---

# 2. 결과 위치

```text
data/raw/
```

다운로드한 원본 데이터

```text
data/extracted/
```

압축해제 / 문자별 분리 결과

```text
data/features/
```

shape / trajectory feature CSV

```text
reports/
```

PCA 및 분석 결과

---

# 3. 한글 분석

PE92 / SERI95를 받아 처리하면:

```bash
python scripts/extract_hgu1.py data/raw/PE92_test.zip --out data/extracted/pe92
python scripts/analyze_images.py data/extracted/pe92 --out data/features/pe92_shapes.csv
python scripts/pca_report.py data/features/pe92_shapes.csv --out reports/pe92_pca.json
```

SERI95도 동일하게 처리한다.

주의:
HangulDB README는 PE92에 오라벨 샘플이 일부 있다고 경고한다.
따라서 초기에는 renderer space 발견용으로 사용하고,
정밀 benchmark로 바로 사용하지 않는다.

---

# 4. 숫자 / 영문 분석

EMNIST:

```bash
python scripts/extract_emnist.py data/raw/emnist_gzip.zip \
  --split byclass \
  --per-class 200 \
  --out data/extracted/emnist_byclass

python scripts/analyze_images.py data/extracted/emnist_byclass \
  --out data/features/emnist_shapes.csv

python scripts/pca_report.py data/features/emnist_shapes.csv \
  --out reports/emnist_pca.json
```

UJI Pen:

```bash
python scripts/parse_uji_v2.py \
  data/raw/uji_pen_characters_v2.zip \
  --out data/features/uji_v2_motor.csv
```

UJI는 실제 timing과 pressure가 없으므로:
- velocity 학습
- pressure model

용으로 사용하지 않는다.

주 용도는:
- stroke count
- stroke topology
- writer variation
- path geometry

다.

---

# 5. Clova 나눔손글씨 109종을 추가하려면

공식 페이지에서 TTF를 받은 뒤:

```text
data/fonts_manual/
```

아래에 넣는다.

그 후:

```bash
python scripts/bootstrap_after_download.py
```

를 다시 실행한다.

이 데이터는:
- geometry stress-test

에는 좋지만,
- 실제 motor trajectory

로 취급하면 안 된다.

---

# 6. AI-Hub 71307을 추가하려면

AI-Hub에서 승인 후 데이터를 내려받는다.

이 데이터는 한국어 motor 분석에서 핵심이다.

확인해야 할 주요 항목:

```text
x / y
DOWN / UP / MOVE
pressure
velocity
acceleration
angle
curvature
difference
stroke-length ratio
writer metadata
```

이 데이터가 들어오면 다음 분석을 추가하는 것이 우선이다.

- writer별 velocity profile
- pressure response
- corner slowdown
- stroke count
- path length
- pen lift
- intra-writer variance
- inter-writer variance

---

# 7. 결과를 어떻게 읽나

예를 들어 특정 문자 데이터에서 PCA 결과가:

```text
PC1 aspect          42%
PC2 rotation        21%
PC3 asymmetry       15%
PC4 compactness      9%
```

라면,

현재 renderer가 최소한 다음 자유도를 표현할 필요가 있다는 뜻이다.

```text
aspect
rotation
asymmetry
shape compactness / closure
```

단, PCA feature를 그대로 renderer parameter로 복사하지 않는다.

예를 들어 `compactness`가 실제로는:
- loop closure
- corner rounding
- stroke overlap

중 무엇 때문인지 샘플을 확인해서 사람이 해석 가능한 parameter로 바꾼다.

---

# 8. 언제 renderer 구현을 시작할까

다음 정도가 확보되면 시작한다.

- `ㄱ / ㄴ / ㄹ / ㅁ / ㅅ / ㅇ` 주요 variation 확인
- 숫자 loop / angular / stem family variation 확인
- 영문 loop / hump / angular / descender family 확인
- 한글 초성/중성/종성 context effect 확인
- trajectory에서 stroke topology 확인

그 전에 renderer를 계속 확장하면 다시 추측 기반 설계가 된다.

---

# 9. Renderer 구현 이후 해야 할 평가

실제 손글씨 sample마다:

```text
reference
  ↓
renderer parameter fitting
  ↓
best render
  ↓
coverage error
```

를 계산한다.

못 맞춘 sample을 따로 저장한다.

예:

```text
failures/
├─ ㅇ_open_loop/
├─ ㅇ_asymmetric/
├─ ㄱ_rounded_corner/
└─ ㅅ_asymmetric_legs/
```

이렇게 failure cluster가 생기면 새로운 자유도를 추가할 근거가 된다.

---

# 10. VLM은 언제 사용하나

Renderer가 충분한 공간을 확보한 뒤에 사용한다.

현재 예정된 VLM 환경:

- vLLM
- Qwen VL 3.5 397B
- OpenAI-compatible API

역할:

```text
참조 vs 렌더 차이 관찰
→ 가장 큰 mismatch 1개 선택
→ parameter group 선택
→ 후보 render 비교
→ 개선 여부 판단
```

VLM에게 전체 parameter vector를 한 번에 추측시키지 않는다.

---

# 11. 사용자가 진행 상황을 판단하는 기준

잘 진행되고 있는 상태:

- 데이터가 늘어날수록 필요한 renderer parameter가 더 명확해짐
- parameter 추가 후 coverage error가 실제로 감소함
- 같은 사람의 반복 필기에서 유사한 parameter가 나옴
- 실패 sample이 점점 특정 작은 cluster로 좁혀짐

잘못 진행되고 있는 상태:

- renderer parameter만 계속 늘어나는데 근거가 없음
- 예쁜 sample 몇 개만 맞고 전체 coverage는 측정하지 않음
- VLM prompt 튜닝만 계속함
- 실제 데이터보다 font sample에 과도하게 의존함
- shape와 motor를 한 번에 섞어 원인을 구분할 수 없음

---

# 12. 핵심 순서

다시 요약하면:

```text
1. 데이터 확보
2. variation 측정
3. renderer parameter taxonomy
4. renderer 구현
5. coverage test
6. failure clustering
7. renderer 확장
8. VLM fitting
```

현재는 **1~3 사이**다.
