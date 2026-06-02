from typing import List, Optional
from pydantic import BaseModel
from app.models.domain import Restaurant

class RecommendationDetail(BaseModel):
    rank: int
    restaurant: Restaurant
    explanation: str

class APIRecommendationResponse(BaseModel):
    summary: Optional[str] = None
    recommendations: List[RecommendationDetail]
    meta: dict
