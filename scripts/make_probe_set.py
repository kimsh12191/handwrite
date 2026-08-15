import json
from pathlib import Path

CHO = ["ㄱ","ㄲ","ㄴ","ㄷ","ㄸ","ㄹ","ㅁ","ㅂ","ㅃ","ㅅ","ㅆ","ㅇ","ㅈ","ㅉ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"]
JUNG = ["ㅏ","ㅐ","ㅑ","ㅒ","ㅓ","ㅔ","ㅕ","ㅖ","ㅗ","ㅘ","ㅙ","ㅚ","ㅛ","ㅜ","ㅝ","ㅞ","ㅟ","ㅠ","ㅡ","ㅢ","ㅣ"]
JONG = ["","ㄱ","ㄲ","ㄳ","ㄴ","ㄵ","ㄶ","ㄷ","ㄹ","ㄺ","ㄻ","ㄼ","ㄽ","ㄾ","ㄿ","ㅀ","ㅁ","ㅂ","ㅄ","ㅅ","ㅆ","ㅇ","ㅈ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"]

def syllable(ci, vi, fi=0):
    return chr(0xAC00 + ci*588 + vi*28 + fi)

def main():
    # Position-controlled coverage:
    # initial: every choseong + ㅏ + no final
    initials=[syllable(i,0,0) for i in range(19)]
    # vowel: ㅇ + each vowel + no final
    vowels=[syllable(11,v,0) for v in range(21)]
    # final: ㄱ-base initial + ㅏ + every final
    finals=[syllable(0,0,f) for f in range(1,28)]

    latin=list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    digits=list("0123456789")
    compat=list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")

    data={
        "digits":digits,
        "latin":latin,
        "hangul_compat_jamo":compat,
        "hangul_initial_context":initials,
        "hangul_vowel_context":vowels,
        "hangul_final_context":finals,
        "all_probe_chars": list(dict.fromkeys(digits+latin+compat+initials+vowels+finals))
    }
    Path("metadata/probe_set.json").write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
    Path("metadata/probe_chars.txt").write_text("".join(data["all_probe_chars"]),encoding="utf-8")
    print({k:len(v) for k,v in data.items() if isinstance(v,list)})

if __name__=="__main__":
    main()
