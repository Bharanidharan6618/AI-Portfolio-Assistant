"""Small text helpers used for retrieval."""

import re

STOPWORDS= {
    "a","an","the","and","or","of","in","on","at","by","for","to","from","with","as",
    "by","for","to","from","with","as","is","are","was","were","be","been","am","do",
    "does","did","have","had","had","i","me","my","you","your","he","his","she","her",
    "they","their","it","its","this","that","what","which","who","whom","where","when",
    "how","can","could","would","should","will","please","tell","give","show","any","all",
    "s","t","d","ll","re","ve","m",
}

_WORD_PATTERN = re.compile(r"[a-z0-9+#]+")

def _stem(word: str) -> str:
    """Very light stemming: 'projects' -> 'project'."""
    if len(word)>3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word

def tokenize(text: str) -> str:
    """Turn text into a list of lowercase keywords."""
    words = _WORD_PATTERN.findall(text.lower())
    return [_stem(word) for word in words if word not in STOPWORDS]


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

def clean_question(text: str, max_length: int) -> str:
    """Make visitor input safe: remove control characters and < > symbols,
    collapse extra spaces, and cut it to the maximum length."""
    text = _CONTROL_CHARS.sub(" ", text)
    text = text.replace("<", " ").replace(">", " ")
    text = " ".join(text.split())
    return text[:max_length]