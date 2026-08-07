"""
LLM/VLM 계약.

루프:
  [LLM] 파라미터 생성/수정 (JSON)
    -> [PY] 시뮬레이터 실행 + 렌더
    -> [VLM] 타겟 vs 렌더 비교, 차이 서술 + 개선 방향 (자연어)
    -> [LLM] 방향을 파라미터 수치로 옮김
    -> 반복

모델은 '이미지 보고 텍스트/JSON 뱉기'만 한다. 렌더링·비교·검증은 전부 파이썬.
따라서 내부망 397B VLM 으로 그대로 교체 가능하다.
"""
import json
import re

from .simulator import STYLE_RANGES, ALLOGRAPHS

# ---------------------------------------------------------------------------
# 파라미터 명세 — LLM 이 읽는 유일한 문서. 값을 올리면 무엇이 커지는지 명시.
# ---------------------------------------------------------------------------
PARAM_SPEC = {
    'slant_deg':    ('도',  '글자 기울기. 올리면 오른쪽으로 눕는다. 음수는 왼쪽으로 눕는다.'),
    'aspect':       ('배',  '가로/세로 비. 올리면 옆으로 퍼지고 낮추면 홀쭉해진다.'),
    'curvature':    ('em',  '획의 부풀림. 올리면 획이 바깥으로 볼록해지고, 음수면 안으로 오목해진다.'),
    'tremor_amp':   ('em',  '손떨림 진폭. 올리면 획이 잔물결처럼 떨린다. 0이면 매끈하다.'),
    'tremor_freq':  ('회',  '손떨림 주파수. 올리면 떨림이 더 촘촘해진다. tremor_amp가 0이면 효과 없음.'),
    'overshoot':    ('em',  '획 끝 넘김. 올리면 획 끝이 진행방향으로 삐져나온다. 음수면 못 미친다.'),
    'corner_round': ('0-1', '모서리 둥글기. 1이면 곡선으로 부드럽게, 0이면 각지게 꺾인다.'),
    'stroke_w':     ('em',  '획 굵기. 올리면 펜이 굵어진다. 너무 크면 고리 안쪽이 메워진다.'),
    'width_var':    ('0-1', '굵기 변동. 올리면 획 중간이 굵고 끝이 가늘어진다(필압 효과).'),
    'ink_noise':    ('0-1', '잉크 얼룩·끊김. 올리면 농도가 얼룩지고 획이 군데군데 끊긴다.'),
    'rotation_deg': ('도',  '글자 전체 회전. 올리면 반시계로 돈다. slant와 달리 형태가 안 눕는다.'),
    'ctrl_jitter':  ('em',  '제어점 흔들림. 올리면 글자 골격 자체가 삐뚤어진다(모양 개인차).'),
    'endpoint_gap': ('em',  '획 시작/끝 잘림. 양수면 획이 짧아져 o가 안 닫히고, 음수면 길어져 겹친다.'),
}


def param_doc():
    """LLM 프롬프트에 넣을 파라미터 명세 표."""
    lines = ['| 이름 | 범위 | 단위 | 올리면 |', '|---|---|---|---|']
    for k, (unit, desc) in PARAM_SPEC.items():
        lo, hi = STYLE_RANGES[k]
        lines.append(f'| {k} | {lo} ~ {hi} | {unit} | {desc} |')
    return '\n'.join(lines)


def allograph_doc(ch):
    """해당 글자의 이체 목록 설명."""
    n = len(ALLOGRAPHS[ch])
    return (f"'{ch}' 의 이체는 {n}가지이며 allo=0..{n-1} 로 지정한다. "
            f"이체는 글자의 '형태' 자체가 다르다(예: 4의 위가 열렸는가 닫혔는가). "
            f"파라미터로는 형태를 바꿀 수 없으므로, 골격이 다르면 이체를 바꿔야 한다.")


# ---------------------------------------------------------------------------
# 프롬프트 빌더
# ---------------------------------------------------------------------------
def prompt_describe(ch):
    """[VLM] 1단계: 타겟 필체를 서술."""
    return f"""첨부 이미지는 사람이 손으로 쓴 '{ch}' 입니다.
이 필체의 특징을 관찰해 서술하세요. 다음을 반드시 포함하세요.

- 기울기: 오른쪽/왼쪽으로 얼마나 누웠는가 (대략 몇 도)
- 굵기: 획이 굵은가 가는가, 굵기가 일정한가 변하는가
- 비율: 가로로 퍼졌는가 홀쭉한가
- 곡률: 획이 곧은가 휘었는가, 모서리가 각진가 둥근가
- 골격: 이 글자를 어떤 형태로 썼는가 (획 구성)
- 기타: 떨림, 잉크 끊김, 획 끝이 삐져나오거나 못 미치는지

추측은 추측이라고 밝히세요. 파라미터 값을 제안하지는 마세요."""


def prompt_propose(ch, description):
    """[LLM] 2단계: 서술 -> 파라미터 JSON."""
    return f"""손글씨 시뮬레이터의 파라미터를 정하는 작업입니다.

## 타겟 '{ch}' 의 필체 서술
{description}

## 조종 가능한 파라미터
{param_doc()}

## 이체
{allograph_doc(ch)}

위 서술과 가장 가까운 결과가 나오도록 파라미터를 정하세요.
JSON 하나만 출력하세요. 설명은 JSON 앞에 두세요.

{{"allo": <정수>, "params": {{"slant_deg": <수>, ...}}}}

params 에는 기본값에서 바꿀 것만 넣으면 됩니다. 범위를 벗어나면 잘립니다."""


def prompt_critique(ch):
    """[VLM] 3단계: 타겟 vs 렌더 비교 -> 차이 + 개선 방향."""
    return f"""첨부 이미지는 손으로 쓴 '{ch}' 두 개를 나란히 놓은 것입니다.
왼쪽이 TARGET(사람이 쓴 진짜), 오른쪽이 시뮬레이터가 만든 것입니다.

두 글자의 차이를 관찰해 서술하세요.

1. 가장 눈에 띄는 차이 3가지를 중요한 순서로
2. 각 차이에 대해 오른쪽을 어느 '방향'으로 바꿔야 왼쪽에 가까워지는지
   (예: "더 눕혀야 한다", "획을 가늘게", "위쪽 고리를 작게")
3. 골격 자체가 다른가? (파라미터로 못 고치고 이체를 바꿔야 하는 종류의 차이인가)
4. 이미 충분히 비슷하면 그렇다고 말하세요

파라미터 이름이나 수치는 쓰지 마세요. 보이는 것만 말하세요."""


def prompt_revise(ch, params, allo, critique):
    """[LLM] 4단계: 개선 방향 -> 수정된 파라미터."""
    return f"""손글씨 시뮬레이터의 파라미터를 수정하는 작업입니다.

## 현재 설정
allo = {allo}
params = {json.dumps(params, ensure_ascii=False)}

## 관찰된 차이와 개선 방향
{critique}

## 조종 가능한 파라미터
{param_doc()}

## 이체
{allograph_doc(ch)}

개선 방향을 반영하도록 설정을 수정하세요.
한 번에 너무 크게 바꾸지 말고, 방향이 맞는지 확인할 만큼만 움직이세요.
골격이 다르다는 지적이 있으면 allo 를 바꾸세요.

JSON 하나만 출력하세요. 무엇을 왜 바꿨는지는 JSON 앞에 쓰세요.

{{"allo": <정수>, "params": {{...}}}}"""


# ---------------------------------------------------------------------------
# 파싱/검증 — 모델 출력을 신뢰하지 않는다
# ---------------------------------------------------------------------------
def parse(text, ch):
    """
    모델 출력에서 마지막 JSON 객체를 뽑아 검증·클램프.
    returns (allo, params, warnings)
    """
    blocks = re.findall(r'\{(?:[^{}]|\{[^{}]*\})*\}', text, re.S)
    if not blocks:
        raise ValueError('JSON 을 찾지 못함')
    obj = None
    for b in reversed(blocks):
        try:
            o = json.loads(b)
        except json.JSONDecodeError:
            continue
        if isinstance(o, dict) and ('params' in o or 'allo' in o):
            obj = o
            break
    if obj is None:
        raise ValueError('allo/params 를 가진 JSON 이 없음')

    warn = []
    allo = int(obj.get('allo', 0))
    n = len(ALLOGRAPHS[ch])
    if not 0 <= allo < n:
        warn.append(f'allo={allo} 범위 밖 -> 0 으로')
        allo = 0

    params = {}
    for k, v in (obj.get('params') or {}).items():
        if k not in STYLE_RANGES:
            warn.append(f'알 수 없는 파라미터 {k} 무시')
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            warn.append(f'{k} 값이 수가 아님 무시')
            continue
        lo, hi = STYLE_RANGES[k]
        if not lo <= v <= hi:
            warn.append(f'{k}={v} 범위 밖 -> {min(max(v, lo), hi)} 로 클램프')
            v = min(max(v, lo), hi)
        params[k] = v
    return allo, params, warn
