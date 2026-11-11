"""Loads portfolio_knowledge.txt and finds the parts relevant to a question."""

import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.config import get_settings
from app.utils.logger import get_logger
from app.utils.text_utils import tokenize

logger = get_logger(__name__)

MAX_CHUNK_CHARS = 1500
DEFAULT_TOP_K = 3
HEADING_BONUS = 2

# Groups of related words. If the question contains one word from a group,
# the whole group is searched. (Written as plain text, then tokenized so the
# stemming matches.)
_SYNONYM_WORDS = [
    "education college university degree school study studied qualification "
    "academic gpa cgpa graduate graduation",
    "skill technology tech stack language tool framework proficient expertise",
    "experience internship intern work job company role employment",
    "project built build developed application app",
    "certification certificate course licence license credential",
    "contact email phone mobile number reach hire location address",
    "github linkedin link repository repo website url profile",
    "introduction intro summary yourself background profile overview",
]
SYNONYM_GROUPS = [frozenset(tokenize(words)) for words in _SYNONYM_WORDS]


class KnowledgeBaseError(Exception):
    """Raised when the portfolio document is missing, unreadable or empty."""


@dataclass(frozen=True)
class Chunk:
    text: str
    tokens: frozenset[str]
    heading_tokens: frozenset[str]


def load_document(path: Path) -> str:
    """Read the portfolio file, or raise a clear error."""
    if not path.is_file():
        raise KnowledgeBaseError(f"Knowledge file not found: {path}")

    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise KnowledgeBaseError("Knowledge file must be saved as UTF-8.") from exc

    text = text.replace("\r\n", "\n").strip()
    if not text:
        raise KnowledgeBaseError("Knowledge file is empty.")
    return text


def _is_heading(block: str) -> bool:
    """A heading is a short single line, such as 'EDUCATION' or 'Skills:'."""
    return (
        "\n" not in block
        and len(block) <= 60
        and len(block.split()) <= 6
        and not block.endswith((".", "!", "?"))
        and ":" not in block[:-1]
    )


def _split_long_chunk(chunk: str) -> list[str]:
    """Break an oversized chunk into pieces at line boundaries."""
    if len(chunk) <= MAX_CHUNK_CHARS:
        return [chunk]

    parts: list[str] = []
    current = ""
    for line in chunk.split("\n"):
        if current and len(current) + len(line) + 1 > MAX_CHUNK_CHARS:
            parts.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        parts.append(current)
    return parts


def split_into_chunks(text: str) -> list[str]:
    """Split at blank lines and keep each heading with the text below it."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]

    merged: list[str] = []
    pending_heading = ""
    for block in blocks:
        if _is_heading(block):
            pending_heading = f"{pending_heading}\n{block}".strip()
            continue
        merged.append(f"{pending_heading}\n{block}".strip() if pending_heading else block)
        pending_heading = ""
    if pending_heading:
        merged.append(pending_heading)

    chunks: list[str] = []
    for item in merged:
        chunks.extend(_split_long_chunk(item))
    return chunks


def _make_chunk(text: str) -> Chunk:
    first_line = text.split("\n", 1)[0]
    return Chunk(
        text=text,
        tokens=frozenset(tokenize(text)),
        heading_tokens=frozenset(tokenize(first_line)),
    )


def _expand_query(tokens: set[str]) -> set[str]:
    """Add related words so 'college' also finds 'education'."""
    expanded = set(tokens)
    for group in SYNONYM_GROUPS:
        if tokens & group:
            expanded |= group
    return expanded


class KnowledgeService:
    def __init__(self, knowledge_file: Path) -> None:
        text = load_document(knowledge_file)
        self._chunks = [_make_chunk(c) for c in split_into_chunks(text)]
        logger.info("Loaded %d knowledge chunks.", len(self._chunks))

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def search(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[str]:
        """Return the best-matching chunks (best first). Empty list if none."""
        query_tokens = _expand_query(set(tokenize(question)))
        if not query_tokens:
            return []

        scored: list[tuple[int, int]] = []
        for index, chunk in enumerate(self._chunks):
            score = len(query_tokens & chunk.tokens)
            score += HEADING_BONUS * len(query_tokens & chunk.heading_tokens)
            if score > 0:
                scored.append((score, index))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [self._chunks[index].text for _, index in scored[:top_k]]

    def get_relevant_context(self, question: str) -> str:
        """Return matching chunks as one text block, or '' if nothing matches."""
        return "\n\n---\n\n".join(self.search(question))


@lru_cache
def get_knowledge_service() -> KnowledgeService:
    """Create the service once and reuse it."""
    return KnowledgeService(get_settings().knowledge_file)


if __name__ == "__main__":
    # Quick manual test:
    #   python -m app.services.knowledge_service "What are your skills?"
    user_question = " ".join(sys.argv[1:]).strip()
    if not user_question:
        print('Usage: python -m app.services.knowledge_service "your question"')
        sys.exit(1)

    service = get_knowledge_service()
    results = service.search(user_question)
    print(f"Loaded {service.chunk_count} chunks. Matches: {len(results)}\n")
    for number, text in enumerate(results, start=1):
        print(f"--- Match {number} ---\n{text}\n")