"""
python postprocess/main.py input.tsv output.tsv
"""
import argparse
import re
from typing import Final

from tqdm import tqdm

VOWELS: Final[tuple[str, ...]] = ("a", "e", "i", "o", "u")
CONSONANTS: Final[tuple[str, ...]] = (
    "b", "v", "d", "h", "z", "χ", "t", "j", "k", "l", "m", "n", "s", "f", "p",
    "ts", "tʃ", "w", "ʔ", "ɡ", "ʁ", "ʃ", "ʒ", "dʒ",
)
STRESS_MARKS: Final[tuple[str, ...]] = ("ˈ",)

HEBREW_TO_IPA_ONSET: Final[dict[str, tuple[str, ...]]] = {
    "א": ("ʔ",), "ב": ("b", "v"), "ג": ("ɡ", "dʒ"),
    "ד": ("d",), "ה": ("h",), "ו": ("v", "w"),
    "ז": ("z", "ʒ"), "ח": ("χ",), "ט": ("t",), "י": ("j"),
    "כ": ("k", "χ"), "ך": ("k", "χ"), "ל": ("l",), "מ": ("m",), "ם": ("m",),
    "נ": ("n",), "ן": ("n",), "ס": ("s",), "ע": ("ʔ",),
    "פ": ("p", "f"), "ף": ("p", "f"), "צ": ("ts", "tʃ"), "ץ": ("ts", "tʃ"),
    "ק": ("k",), "ר": ("ʁ",), "ש": ("ʃ", "s"), "ת": ("t",),
}

_IPA_UNITS: Final[tuple[str, ...]] = tuple(sorted(CONSONANTS + VOWELS + STRESS_MARKS, key=len, reverse=True))
_IPA_PATTERN: re.Pattern = re.compile("|".join(re.escape(u) for u in _IPA_UNITS))
_CONSONANT_SET: Final[frozenset[str]] = frozenset(CONSONANTS)
_VOWEL_SET: Final[frozenset[str]] = frozenset(VOWELS)
_HEBREW_RE: re.Pattern = re.compile(r"^[א-תׁ-ׂ\s]+$")
_STRIP_PUNCT: re.Pattern = re.compile(r"[.,?!;:\-]+")
_PUNCT_AT_END: re.Pattern = re.compile(r"[.,?!;:]+$")


def ipa_word_onset(word: str) -> str:
    """First IPA phoneme, ignoring stress marks."""
    word = _STRIP_PUNCT.sub("", word).replace("ˈ", "")
    for unit in _IPA_UNITS:
        if word.startswith(unit):
            return unit
    return ""


def onset_match(heb_word: str, ipa_word: str) -> bool:
    first = next((ch for ch in heb_word if ch in HEBREW_TO_IPA_ONSET), "")
    if not first:
        return True
    return ipa_word_onset(ipa_word) in HEBREW_TO_IPA_ONSET[first]


def filter_ipa_word(word: str) -> bool:
    word = _STRIP_PUNCT.sub("", word)
    if not word:
        return True
    tokens = _IPA_PATTERN.findall(word)
    if "".join(tokens) != word:
        return False
    if word.count("ˈ") != 1:
        return False
    phonemes = [t for t in tokens if t in _CONSONANT_SET or t in _VOWEL_SET]
    for a, b in zip(phonemes, phonemes[1:]):
        if a == b:
            return False
    return True


def lcs_align(heb_words: list[str], ipa_words: list[str]) -> list[tuple[int, int]]:
    """LCS by onset match, returns list of aligned (heb_idx, ipa_idx) pairs."""
    n, m = len(heb_words), len(ipa_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if onset_match(heb_words[i - 1], ipa_words[j - 1]):
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        if onset_match(heb_words[i - 1], ipa_words[j - 1]) and dp[i][j] == dp[i - 1][j - 1] + 1:
            pairs.append((i - 1, j - 1))
            i -= 1; j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def extract_sentences(heb_text: str, ipa_text: str) -> list[tuple[str, str]]:
    heb_words = heb_text.split()
    ipa_words = ipa_text.split()
    if not heb_words or not ipa_words:
        return []

    pairs = lcs_align(heb_words, ipa_words)
    if not pairs:
        return []

    # Build map: heb_idx -> ipa_idx for aligned words
    heb_to_ipa = dict(pairs)

    sentences = []
    h_start = 0
    last_ipa_idx = -1

    for h_idx, heb_word in enumerate(heb_words):
        is_last = h_idx == len(heb_words) - 1
        has_punct = bool(_PUNCT_AT_END.search(heb_word))

        if not (has_punct or is_last):
            continue

        h_end = h_idx + 1

        # Find IPA range: from after last used ipa up to furthest aligned ipa in this heb range
        aligned_ipa_in_range = [ipa_idx for hh, ipa_idx in pairs if h_start <= hh < h_end]
        if not aligned_ipa_in_range:
            h_start = h_end
            continue

        i_end = max(aligned_ipa_in_range) + 1
        i_start = last_ipa_idx + 1

        if i_end <= i_start:
            h_start = h_end
            continue

        heb_seg_words = heb_words[h_start:h_end]
        ipa_seg_words = ipa_words[i_start:i_end]

        # Quality: enough of the Hebrew words are aligned
        seg_matches = len(aligned_ipa_in_range)
        if seg_matches / len(heb_seg_words) < 0.5:
            h_start = h_end
            last_ipa_idx = i_end - 1
            continue

        # Strip punctuation from words for output
        heb_clean = " ".join(_STRIP_PUNCT.sub("", w) for w in heb_seg_words)
        ipa_clean = " ".join(_STRIP_PUNCT.sub("", w) for w in ipa_seg_words)

        # Text must be pure Hebrew
        if not _HEBREW_RE.match(heb_clean):
            h_start = h_end
            last_ipa_idx = i_end - 1
            continue

        # IPA words must contain only valid phonemes
        if not all(filter_ipa_word(w) for w in ipa_seg_words):
            h_start = h_end
            last_ipa_idx = i_end - 1
            continue

        # Drop sentences with any single-char word on either side
        if any(len(_STRIP_PUNCT.sub("", w)) <= 1 for w in heb_seg_words + ipa_seg_words):
            h_start = h_end
            last_ipa_idx = i_end - 1
            continue

        if heb_clean and ipa_clean and len(heb_seg_words) == len(ipa_seg_words):
            sentences.append((heb_clean, ipa_clean))

        h_start = h_end
        last_ipa_idx = i_end - 1

    return sentences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input TSV file")
    parser.add_argument("output", help="output TSV file")
    args = parser.parse_args()

    written = 0
    total_chunks = 0

    with open(args.input, encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        lines = fin.readlines()
        for line in tqdm(lines, unit="chunk"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue

            filename, text, ipa = parts
            total_chunks += 1

            for text_seg, ipa_seg in extract_sentences(text, ipa):
                fout.write(f"{filename}\t{text_seg}\t{ipa_seg}\n")
                written += 1

    print(f"Processed {total_chunks} chunks.")
    print(f"Written {written} rows to {args.output}")


if __name__ == "__main__":
    main()
