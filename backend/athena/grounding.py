"""Grounding check: is every specific claim in the answer backed by a source?

A small local model is most dangerous when it is confidently wrong: the
Milestone 1 eval caught gemma3:4b writing "page authority 28" with a
citation, when the cited page said 31. Checking free-form meaning needs
another model, but the claims that hurt most are *specific tokens*:
numbers, versions, codes and acronyms. Those can be checked exactly.

For each answer sentence, the specific tokens must appear in the sources
that sentence cites (or, if it cites nothing, in any source given to the
model). Number words are normalised, so "fifteen" matches "15".
"""

import re
from dataclasses import dataclass, field

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}
_NUMBER_WORD_RE = re.compile(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b", re.IGNORECASE)
# Tokens that contain a digit: 15, 2.0, 800-63B, AAL2, 1,000, 25%, 2024
_DIGIT_TOKEN = re.compile(r"\b[A-Za-z]*\d[\w.,\-]*\b")
# Acronyms / codes: NIST, OECD, CBRN, GV.OC (two or more capitals)
_ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]{1,}(?:[.\-][A-Z0-9]+)*\b")
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[*-])|\n+")

# Capitalised words a model writes that are never "claims".
_IGNORE_ACRONYMS = {"AI", "OK", "NOT_FOUND", "I", "A", "II", "FAQ", "PDF", "DOCX"}


@dataclass
class Grounding:
    supported: bool
    unsupported: list[str] = field(default_factory=list)  # tokens not found in the sources
    checked: int = 0  # specific tokens checked
    uncited_sentences: int = 0  # sentences with claims but no [n]
    invalid_citations: list[int] = field(default_factory=list)  # [n] beyond the source list


def _normalise(text: str) -> str:
    text = _NUMBER_WORD_RE.sub(lambda m: str(_NUMBER_WORDS[m.group(1).lower()]), text)
    return re.sub(r"\s+", " ", text).lower()


def _canonical(token: str) -> str:
    token = token.strip(".,-").lower()
    if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", token):
        token = token.replace(",", "")
    return token


def specific_tokens(sentence: str) -> list[str]:
    text = _CITATION.sub(" ", sentence)
    # List markers ("1.", "2)", "-") are formatting, not claims.
    text = re.sub(r"^\s*(?:\d{1,2}[.)]|[-*•])\s+", "", text)
    tokens = [m.group(0) for m in _DIGIT_TOKEN.finditer(_NUMBER_WORD_RE.sub(
        lambda m: str(_NUMBER_WORDS[m.group(1).lower()]), text))]
    tokens += [m.group(0) for m in _ACRONYM.finditer(text) if m.group(0) not in _IGNORE_ACRONYMS]
    seen, out = set(), []
    for token in tokens:
        canonical = _canonical(token)
        if canonical and canonical not in seen:
            seen.add(canonical)
            out.append(token.strip(".,"))
    return out


def split_sentences(answer: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(answer) if s and s.strip()]


def cited_numbers(sentence: str) -> list[int]:
    return [int(n) for group in _CITATION.findall(sentence) for n in group.split(",")]


def check_grounding(answer: str, sources: list[str]) -> Grounding:
    """`sources[i]` is the text of source number i+1 in the prompt."""
    normalised = [_normalise(s) for s in sources]
    everything = " ".join(normalised)
    result = Grounding(supported=True)

    for sentence in split_sentences(answer):
        tokens = specific_tokens(sentence)
        numbers = cited_numbers(sentence)
        result.invalid_citations += [n for n in numbers if not 1 <= n <= len(sources)]
        valid = [n for n in numbers if 1 <= n <= len(sources)]
        if tokens and not valid:
            result.uncited_sentences += 1
        haystack = " ".join(normalised[n - 1] for n in valid) if valid else everything
        for token in tokens:
            result.checked += 1
            canonical = _canonical(token)
            pattern = r"(?<![\w.])" + re.escape(canonical) + r"(?![\w])"
            if not re.search(pattern, haystack) and not re.search(pattern, haystack.replace(",", "")):
                # Cited the wrong source but the fact is elsewhere in the context:
                # a citation slip, not an invented fact.
                if valid and re.search(pattern, everything):
                    continue
                result.unsupported.append(token)

    result.supported = not result.unsupported and not result.invalid_citations
    return result
