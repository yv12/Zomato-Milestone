from typing import List, Dict, Any
from ..models.domain import UserPreferences, RecommendationResponse, Recommendation, Restaurant
from .filter_service import FilterService
from .vector_search import VectorSearchEngine
from .prompt_builder import PromptBuilder
from .llm_client import LLMClient
from .response_parser import ResponseParser

class RecommendationOrchestrator:
    def __init__(
        self, 
        filter_service: FilterService, 
        vector_search_engine: VectorSearchEngine, 
        llm_client: LLMClient
    ):
        self.filter_service = filter_service
        self.vector_search_engine = vector_search_engine
        self.llm_client = llm_client

    def execute(self, preferences: UserPreferences) -> RecommendationResponse:
        print(f"Orchestrating recommendation for location: '{preferences.location}', cuisine: '{preferences.cuisine}'")
        
        # 1. Deterministic Sequential Filtering
        filter_result = self.filter_service.filter(preferences)
        candidates = filter_result.candidates
        print(f"Hierarchical pre-filtering matched {filter_result.total_before_cap} items, capped to {len(candidates)} candidates.")
        
        # Early short-circuit if candidate pool is completely empty
        if not candidates:
            return RecommendationResponse(
                summary="No restaurants matched your exact criteria. Try broadening your location, rating, or budget parameters.",
                recommendations=[]
            )
            
        # 2. Local Semantic Cosine Vector Similarity Search
        print(f"Executing local semantic vector search against {len(candidates)} candidates for qualitative preferences...")
        semantic_candidates = self.vector_search_engine.search(
            query=preferences.additional_preferences,
            candidates=candidates,
            top_n=preferences.top_n
        )
        print(f"Semantic search selected top {len(semantic_candidates)} candidate matches.")
        
        # 3. Prompt Construction
        prompt = PromptBuilder.build(preferences, semantic_candidates)
        
        # 4. Request LLM Completion with Grounding Retries & Fallback Safeguards
        try:
            raw_response = self.llm_client.complete(prompt)
            parsed_json = ResponseParser.parse(raw_response)
            response = self._merge(parsed_json, semantic_candidates)
        except Exception as e:
            print(f"Warning: Connection timeout or LLM failure encountered: {e}. Activating rating-based fallback engine...")
            
            # Formulate fallback default recommendations based on semantic candidates
            fallback_recs = []
            for rank, res in enumerate(semantic_candidates, start=1):
                cuisines_str = ", ".join(res.cuisines)
                fallback_recs.append(
                    Recommendation(
                        restaurant_id=res.id,
                        rank=rank,
                        explanation=f"Highly popular restaurant in {res.location} matching your requirements, rated {res.rating} stars."
                    )
                )
            
            response = RecommendationResponse(
                summary="AI Custom explanations are currently offline. Serving direct database-ranked restaurant listings.",
                recommendations=fallback_recs
            )
            
        return response

    def _merge(self, parsed_json: Dict[str, Any], candidates: List[Restaurant]) -> RecommendationResponse:
        summary = parsed_json.get("summary", "Here are your personalized restaurant selections.")
        raw_recommendations = parsed_json.get("recommendations", [])
        
        # Lookup dictionary for fast O(1) grounding validation
        candidate_dict = {c.id: c for c in candidates}
        
        final_recommendations = []
        for rec in raw_recommendations:
            rec_id = rec.get("restaurant_id")
            
            # Grounding: Only allow IDs that were present in our vector candidate subset!
            if rec_id in candidate_dict:
                final_recommendations.append(
                    Recommendation(
                        restaurant_id=rec_id,
                        rank=rec.get("rank", 99),
                        explanation=rec.get("explanation", "Recommended by AI based on your preferences.")
                    )
                )
            else:
                print(f"Warning: LLM hallucinated unknown restaurant_id '{rec_id}'. Skipping entry to maintain grounding.")
                
        # Sort recommendations strictly by rank
        final_recommendations.sort(key=lambda r: r.rank)
        
        return RecommendationResponse(
            summary=summary,
            recommendations=final_recommendations
        )
