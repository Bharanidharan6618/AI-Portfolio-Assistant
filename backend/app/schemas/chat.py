"""Request and response models for the chat endpoint."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import get_settings

_MAX_QUESTION_LENGTH = get_settings().max_question_length


class ChatRequest(BaseModel):
    # Trim spaces around the question, and reject unexpected extra fields.
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    question: str = Field(
        ...,
        min_length=1,
        max_length=_MAX_QUESTION_LENGTH,
        description="The visitor's question about the portfolio.",
        examples=["What are your technical skills?"],
    )

    @field_validator("question")
    @classmethod
    def must_contain_text(cls, value: str) -> str:
        if not any(ch.isalnum() for ch in value):
            raise ValueError("Question must contain letters or numbers.")
        return value


class ChatResponse(BaseModel):
    answer: str = Field(
        ...,
        description="Answer generated from the portfolio information.",
        examples=["The portfolio lists skills in Python, FastAPI and SQL."],
    )


class ErrorResponse(BaseModel):
    detail: str = Field(..., examples=["The AI service is temporarily unavailable."])