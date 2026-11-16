"""Response model for the portfolio endpoint."""

from pydantic import BaseModel, Field


class PortfolioResponse(BaseModel):
    sections: list[str] = Field(
        ..., description="Topics covered by the portfolio.",
        examples=[["Education", "Skills", "Projects"]],
    )
    total_sections: int = Field(..., examples=[3])
    message: str = "Ask about any of these topics using POST /api/chat."