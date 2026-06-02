import traceback
from typing import List
from fastapi import APIRouter, HTTPException
from app.config import settings
from app.models.domain import UserPreferences
from app.services.registry import services
from app.api.schemas import APIRecommendationResponse, RecommendationDetail

router = APIRouter(prefix="/api/v1")

# In-memory cache for fast location and cuisine dropdown metadata feeds
metadata_cache = {"locations": None, "cuisines": None}

def build_metadata_cache():
    if services.repo is None or services.repo._df is None:
        return
    
    df = services.repo._df
    
    # Extract unique locations and enrich with standard popular Bangalore neighbor areas
    locations_set = set(str(loc).strip() for loc in df['location'].dropna().unique() if str(loc).strip())
    popular_missing_locations = [
        "HSR Layout", "Whitefield", "Marathahalli", "Electronic City", 
        "Basavanagudi", "Kalyan Nagar", "Richmond Town", "Ulsoor", 
        "Sadashivanagar", "Hebbal", "Domlur", "Bannerghatta Road", "Kammanahalli"
    ]
    for loc in popular_missing_locations:
        locations_set.add(loc)
    metadata_cache["locations"] = sorted(list(locations_set))
    
    # Extract unique cuisines and enrich with standard cuisine listings
    cuisines_set = set()
    for c_list in df['cuisines'].dropna():
        if isinstance(c_list, list):
            for cuisine in c_list:
                cuisines_set.add(str(cuisine).strip())
        else:
            cuisines_set.add(str(c_list).strip())
            
    popular_missing_cuisines = [
        "North Indian", "South Indian", "Chinese", "Italian", "Continental", 
        "Desserts", "Cafe", "Biryani", "Fast Food", "Mughlai", "Street Food", "Beverages"
    ]
    for c in popular_missing_cuisines:
        cuisines_set.add(c)
    metadata_cache["cuisines"] = sorted(list(cuisines_set))

@router.get("/health")
def health_check():
    """Health check endpoint confirming data and service integrity."""
    services.initialize()
    if services.orchestrator is None or services.repo is None:
        raise HTTPException(
            status_code=503, 
            detail="Recommendation engine services are offline. Verify dataset ingestion."
        )
    return {"status": "healthy", "engine": "groq", "model": settings.llm_model}

@router.get("/metadata/locations", response_model=List[str])
def get_locations():
    """Returns unique location suggestions for the dynamic UI control panel."""
    services.initialize()
    if metadata_cache["locations"] is None:
        build_metadata_cache()
    return metadata_cache["locations"] or []

@router.get("/metadata/cuisines", response_model=List[str])
def get_cuisines():
    """Returns unique cuisine suggestions for the control panel dropdown."""
    services.initialize()
    if metadata_cache["cuisines"] is None:
        build_metadata_cache()
    return metadata_cache["cuisines"] or []

@router.post("/recommendations", response_model=APIRecommendationResponse)
def get_recommendations(prefs: UserPreferences):
    """Processes user preferences and generates semantic vector-ranked recommendations."""
    services.initialize()
    if services.orchestrator is None or services.repo is None:
        raise HTTPException(status_code=503, detail="Discovery engine is offline.")
    
    try:
        # 1. Execute Zomato discovery pipeline
        response = services.orchestrator.execute(prefs)
        
        # 2. Grab matching records from database cache
        rec_ids = [rec.restaurant_id for rec in response.recommendations]
        db_restaurants = services.repo.get_by_ids(rec_ids)
        db_lookup = {r.id: r for r in db_restaurants}
        
        # 3. Assemble response card detail models
        rec_details = []
        for rec in response.recommendations:
            res_details = db_lookup.get(rec.restaurant_id)
            if res_details:
                rec_details.append(
                    RecommendationDetail(
                        rank=rec.rank,
                        restaurant=res_details,
                        explanation=rec.explanation
                    )
                )
                
        # 4. Get metadata for candidates count
        filter_result = services.filter_service.filter(prefs)
        
        return APIRecommendationResponse(
            summary=response.summary,
            recommendations=rec_details,
            meta={
                "candidates_considered": filter_result.total_before_cap,
                "filters_applied": ["location", "budget", "cuisine", "min_rating", "semantic_vector_search"]
            }
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Restaurant discovery recommendation failed: {e}"
        )
