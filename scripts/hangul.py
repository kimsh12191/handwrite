"""Hangul syllable decomposition and layout typing.

HANDOFF.md §5 forbids treating the 11,172 precomposed syllables as independent
glyphs: the renderer has to work from 초성/중성/종성 plus context-dependent
deformation. Everything downstream that needs to know which jamo a sample
contains, or how those jamo are packed into the square, comes through here.
"""

BASE = 0xAC00
LAST = 0xD7A3

CHO = ["ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
       "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
JUNG = ["ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ",
        "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"]
JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ",
        "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]

# Where the 중성 sits relative to the 초성. This is the standard typographic
# split and it is what makes a syllable's ink layout change shape, so it is the
# grouping variable for the layout measurements.
VERTICAL_VOWELS = {"ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅣ"}      # 중성 to the right
HORIZONTAL_VOWELS = {"ㅗ", "ㅛ", "ㅜ", "ㅠ", "ㅡ"}                          # 중성 below
MIXED_VOWELS = {"ㅘ", "ㅙ", "ㅚ", "ㅝ", "ㅞ", "ㅟ", "ㅢ"}                    # wraps right and below


def decompose(ch):
    """('ㄱ','ㅏ','ㄱ') for 각. Returns None for anything not a precomposed syllable."""
    if len(ch) != 1 or not (BASE <= ord(ch) <= LAST):
        return None
    i = ord(ch) - BASE
    return CHO[i // 588], JUNG[(i % 588) // 28], JONG[i % 28]


def vowel_class(jung):
    if jung in VERTICAL_VOWELS:
        return "vertical"
    if jung in HORIZONTAL_VOWELS:
        return "horizontal"
    if jung in MIXED_VOWELS:
        return "mixed"
    return "unknown"


def layout_type(ch):
    """One of the six packing types: {vertical,horizontal,mixed} x {open,closed}.

    'closed' means a 종성 is present, which forces the 초성/중성 to give up
    vertical room. That compression is the effect measured in jamo_context.py.
    """
    d = decompose(ch)
    if d is None:
        return None
    cho, jung, jong = d
    return f"{vowel_class(jung)}_{'closed' if jong else 'open'}"


def compose(cho, jung, jong=""):
    return chr(BASE + CHO.index(cho) * 588 + JUNG.index(jung) * 28 + JONG.index(jong))


def syllable_from_label(label):
    """Recover the syllable from an extractor directory name like 'UAC00_가'."""
    if "_" in label:
        tail = label.split("_", 1)[1]
        if len(tail) == 1 and BASE <= ord(tail) <= LAST:
            return tail
    if len(label) == 1 and BASE <= ord(label) <= LAST:
        return label
    return None
