"""Portfolio information endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.schemas.chat import ErrorResponse
from app.schemas.portfolio import PortfolioResponse
from app.services.knowledge_service import KnowledgeService, get_knowledge_service

router = APIRouter(prefix="/api", tags=["Portfolio"])

KnowledgeDep = Annotated[KnowledgeService, Depends(get_knowledge_service)]


@router.get(
    "/portfolio",
    response_model=PortfolioResponse,
    responses={503: {"model": ErrorResponse}},
)
async def get_portfolio(knowledge: KnowledgeDep) -> PortfolioResponse:
    sections = knowledge.section_titles
    return PortfolioResponse(sections=sections, total_sections=len(sections))