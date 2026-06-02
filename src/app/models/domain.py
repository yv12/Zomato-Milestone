from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class Restaurant(BaseModel):
    id: str
    name: str
    location: str
    cuisines: List[str]
    rating: float
    estimated_cost: float
    budget_band: Literal["low", "medium", "high", "unknown"]
    votes: int
    dish_liked: Optional[List[str]] = None
    text_representation: Optional[str] = None
    metadata: Optional[dict] = None

class UserPreferences(BaseModel):
    location: str = Field(..., description="Target city or locality")
    budget: Literal["low", "medium", "high"] = Field(..., description="User's budget constraint")
    cuisine: str = Field(..., description="Preferred cuisine")
    min_rating: float = Field(default=0.0, ge=0.0, le=5.0, description="Minimum acceptable rating")
    additional_preferences: Optional[str] = Field(default=None, description="Free text for extra constraints")
    top_n: int = Field(default=5, ge=1, le=10, description="Number of recommendations to return (capacity constraint)")

class FilterCriteria(BaseModel):
    location: str
    budget: Literal["low", "medium", "high"]
    cuisine: str
    min_rating: float

class Recommendation(BaseModel):
    restaurant_id: str
    rank: int
    explanation: str
    
class RecommendationResponse(BaseModel):
    summary: Optional[str] = None
    recommendations: List[Recommendation]
