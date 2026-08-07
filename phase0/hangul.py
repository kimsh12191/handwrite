"""
한글 음절 <-> 자모 분해, 그리고 자모 배치 유형.

Q1 이 필요로 하는 것: '방' 이라는 라벨에서 ㅂ/ㅏ/ㅇ 세 자모와
"초성은 왼쪽, 중성은 오른쪽, 종성은 아래" 라는 배치를 얻는 것.
배치 유형은 **중성이 결정한다** — 조합형 폰트가 쓰는 규칙과 같다.
"""

BASE = 0xAC00
N_MED, N_FIN = 21, 28

CHO = list('ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ')
JUNG = list('ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ')
JONG = [''] + list('ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ')

# 겹자모 -> 낱자모. 재사용 단위는 낱자모이므로 쪼갤 수 있으면 쪼갠다.
# 다만 이미지에서 두 조각으로 나누는 것은 별도 문제라, 분해는 표기만 하고
# 이미지 분할에서는 통으로 다룬다 (CLUSTER 참조).
CLUSTER = {
    'ㄳ': 'ㄱㅅ', 'ㄵ': 'ㄴㅈ', 'ㄶ': 'ㄴㅎ', 'ㄺ': 'ㄹㄱ', 'ㄻ': 'ㄹㅁ',
    'ㄼ': 'ㄹㅂ', 'ㄽ': 'ㄹㅅ', 'ㄾ': 'ㄹㅌ', 'ㄿ': 'ㄹㅍ', 'ㅀ': 'ㄹㅎ',
    'ㅄ': 'ㅂㅅ',
}

# 배치 유형 — 중성의 모양이 결정한다
V_MEDIAL = set('ㅏㅐㅑㅒㅓㅔㅕㅖㅣ')      # 세로로 긴 중성 -> 초성 왼쪽 | 중성 오른쪽
H_MEDIAL = set('ㅗㅛㅜㅠㅡ')             # 가로로 긴 중성 -> 초성 위 / 중성 아래
M_MEDIAL = set('ㅘㅙㅚㅝㅞㅟㅢ')          # 섞임 -> 중성이 ㄴ 자로 감쌈

LAYOUT_V, LAYOUT_H, LAYOUT_M = 'V', 'H', 'M'


def is_syllable(ch):
    return len(ch) == 1 and BASE <= ord(ch) < BASE + 11172


def decompose(ch):
    """'방' -> ('ㅂ','ㅏ','ㅇ').  종성 없으면 ''."""
    if not is_syllable(ch):
        raise ValueError(f'한글 음절이 아님: {ch!r}')
    i = ord(ch) - BASE
    return CHO[i // (N_MED * N_FIN)], JUNG[(i // N_FIN) % N_MED], JONG[i % N_FIN]


def compose(cho, jung, jong=''):
    return chr(BASE + (CHO.index(cho) * N_MED + JUNG.index(jung)) * N_FIN
               + JONG.index(jong))


def layout_of(jung):
    if jung in V_MEDIAL:
        return LAYOUT_V
    if jung in H_MEDIAL:
        return LAYOUT_H
    return LAYOUT_M


def split_cluster(j):
    """겹자모면 낱자모 문자열, 아니면 그대로."""
    return CLUSTER.get(j, j)


def parts(ch):
    """
    음절 -> [(자모, 역할), ...] + 배치 유형.
    역할: 'cho' | 'jung' | 'jong'
    returns (parts, layout)
    """
    cho, jung, jong = decompose(ch)
    p = [(cho, 'cho'), (jung, 'jung')]
    if jong:
        p.append((jong, 'jong'))
    return p, layout_of(jung)


ROMAN = dict(zip(
    'ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ',
    ['g', 'kk', 'n', 'd', 'tt', 'r', 'm', 'b', 'pp', 's', 'ss', 'ng', 'j',
     'jj', 'ch', 'k', 't', 'p', 'h',
     'a', 'ae', 'ya', 'yae', 'eo', 'e', 'yeo', 'ye', 'o', 'wa', 'wae', 'oe',
     'yo', 'u', 'wo', 'we', 'wi', 'yu', 'eu', 'ui', 'i']))


def roman(j):
    """자모 -> 로마자. 한글 폰트가 없는 환경에서 라벨용."""
    return ''.join(ROMAN.get(c, c) for c in split_cluster(j))


def jamo_counts(text):
    """전사 텍스트의 자모 빈도. 겹받침은 낱자모로 펴서 센다."""
    from collections import Counter
    c = Counter()
    for ch in text:
        if not is_syllable(ch):
            continue
        for j in decompose(ch):
            if not j:
                continue
            for k in split_cluster(j):
                c[k] += 1
    return c
