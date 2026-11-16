"""Cheap first-line defences against prompt injection and secret leaks.

These filters only catch obvious attacks. The main protections are the strict
system instruction, the retrieval step (Gemini only sees matching portfolio
text) and the fact that no secrets are ever placed in the prompt.
"""

import re
import unicodedata

REFUSAL_MESSAGE = "Sorry, I can only answer questions about this portfolio."

_ZERO_WIDTH = re.compile("[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")

_INJECTION_PATTERNS = [
    re.compile(pattern)
    for pattern in (
        # "ignore all previous instructions", "disregard the above rules"
        r"\b(ignore|disregard|forget)\b.{0,30}"
        r"\b(previous|prior|above|earlier|all|any|your|the|these|those|system)\b.{0,30}"
        r"\b(instructions?|rules?|prompts?|guidelines?|restrictions?)\b",
        # "reveal your system prompt", "what are your hidden instructions"
        r"\b(reveal|show|print|repeat|display|output|leak|tell|give|share|dump"
        r"|what is|what are|what's)\b.{0,30}"
        r"\b(your|the)\s+(system|hidden|initial|secret|developer)\s+"
        r"(prompt|instructions?|message|rules)\b",
        # "you are now a ..."
        r"\byou are now (a|an|the|free|no longer|in|acting|my)\b",
        # "pretend you are ..."
        r"\b(pretend|roleplay|role-play)\b.{0,20}\b(you are|to be|you're|you have no)\b",
        # well-known jailbreak phrases
        r"\b(jailbreak|jailbroken|developer mode|god mode|do anything now)\b",
        # asking for the bot's own secrets
        r"\byour\s+(own\s+)?(api[ -]?keys?|secrets?|passwords?|credentials?"
        r"|access tokens?|environment variables?)\b",
        r"\.env\b",
    )
]

_LEAK_MARKERS = (
    "portfolio_context",
    "visitor_question",
    "reply with exactly this sentence",
)
_KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z_\-]{30,}")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH.sub("", text).replace("\u2019", "'")
    return " ".join(text.lower().split())


def is_suspicious_question(question: str) -> bool:
    """True if the question contains an obvious injection or secret-probing phrase."""
    normalized = _normalize(question)
    return any(pattern.search(normalized) for pattern in _INJECTION_PATTERNS)


def is_safe_answer(answer: str, secrets: tuple[str, ...] = ()) -> bool:
    """False if the answer looks like it leaks instructions or secrets."""
    lowered = answer.lower()
    if any(marker in lowered for marker in _LEAK_MARKERS):
        return False
    if _KEY_PATTERN.search(answer):
        return False
    return not any(secret and secret in answer for secret in secrets)