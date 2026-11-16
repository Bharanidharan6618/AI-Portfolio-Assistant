"""Chat endpoint: question in, portfolio-grounded answer out."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import enforce_rate_limit
from app.schemas.chat import ChatRequest, ChatResponse, ErrorResponse
from app.services.gemini_service import GeminiService, get_gemini_service
from app.services.knowledge_service import KnowledgeService, get_knowledge_service

router = APIRouter(prefix="/api", tags=["Chat"])

KnowledgeDep = Annotated[KnowledgeService, Depends(get_knowledge_service)]
GeminiDep = Annotated[GeminiService, Depends(get_gemini_service)]


@router.post(
    "/chat",
    response_model=ChatResponse,
    dependencies=[Depends(enforce_rate_limit)],
    responses={
        429: {"model": ErrorResponse, "description": "Rate or daily limit reached"},
        502: {"model": ErrorResponse, "description": "AI service request failed"},
        503: {"model": ErrorResponse, "description": "Service temporarily unavailable"},
        504: {"model": ErrorResponse, "description": "AI service timed out"},
    },
)
async def chat(
    request: ChatRequest,
    knowledge: KnowledgeDep,
    gemini: GeminiDep,
) -> ChatResponse:
    context = knowledge.get_relevant_context(request.question)
    answer = await gemini.generate_answer(request.question, context)
    return ChatResponse(answer=answer)