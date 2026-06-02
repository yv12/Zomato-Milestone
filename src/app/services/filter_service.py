from typing import List, Dict, Any
from app.config import settings
from ..models.domain import Restaurant, UserPreferences, FilterCriteria
from ..data.repository import RestaurantRepository

class FilterResult:
    def __init__(self, candidates: List[Restaurant], total_before_cap: int, applied_filters: Dict[str, Any]):
        self.candidates = candidates
        self.total_before_cap = total_before_cap
        self.applied_filters = applied_filters

class FilterService:
    def __init__(self, repository: RestaurantRepository):
        self.repository = repository
        self.max_candidates = settings.max_candidates

    def filter(self, preferences: UserPreferences) -> FilterResult:
        # 1. Map Preferences to Criteria
        criteria = FilterCriteria(
            location=preferences.location,
            budget=preferences.budget,
            cuisine=preferences.cuisine,
            min_rating=preferences.min_rating
        )
        
        # 2. Query Repository for deterministic matches
        candidates = self.repository.filter(criteria)
        total_before_cap = len(candidates)
        
        # 3. Pre-sort by rating and votes descending to get high quality candidates
        candidates.sort(key=lambda r: (r.rating, r.votes), reverse=True)
        
        # 4. Cap Candidates to prevent blowing up the LLM token context limit
        capped_candidates = candidates[:self.max_candidates]
        
        applied_filters = {
            "location": criteria.location,
            "budget": criteria.budget,
            "cuisine": criteria.cuisine,
            "min_rating": criteria.min_rating,
            "max_candidates_cap": self.max_candidates
        }
        
        return FilterResult(
            candidates=capped_candidates,
            total_before_cap=total_before_cap,
            applied_filters=applied_filters
        )
