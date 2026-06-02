import json
from typing import List
from ..models.domain import Restaurant, UserPreferences

class PromptBuilder:
    @staticmethod
    def build(preferences: UserPreferences, candidates: List[Restaurant]) -> str:
        # 1. Format Candidates with rich metadata fields for LLM preference analysis
        candidates_list = []
        for c in candidates:
            candidates_list.append({
                "restaurant_id": c.id,
                "name": c.name,
                "location": c.location,
                "cuisines": c.cuisines,
                "rating": c.rating,
                "cost_for_two": c.estimated_cost,
                "budget_band": c.budget_band,
                "dishes_liked": c.dish_liked or []
            })
        
        candidates_json = json.dumps(candidates_list, indent=2)
        
        # 2. Build the System Instructions & Grounding constraints
        system_prompt = f"""You are an expert restaurant advisor. 
Your task is to recommend restaurants to a user based on their preferences.
You MUST ONLY recommend restaurants from the provided Candidate List below. Do not invent or recommend any restaurant not in the list.

USER PREFERENCES:
- Target Location: {preferences.location}
- Budget Category: {preferences.budget}
- Preferred Cuisine: {preferences.cuisine}
- Minimum Quality Rating: {preferences.min_rating}
- Qualitative / Additional Requests: {preferences.additional_preferences or "None"}

CANDIDATE LIST:
{candidates_json}

INSTRUCTIONS:
1. Analyze the candidate list and select the top {preferences.top_n} restaurants that best match the user's explicit preferences and qualitative "Additional Requests".
2. Rank them from 1 to {preferences.top_n}.
3. Provide a brief, persuasive explanation (strictly 1 sentence, maximum 20 words) for each recommendation explaining why it fits. Do not overflow or write excessive text.
4. Output your response STRICTLY as a JSON object matching the following structure exactly (do not output any conversational wrapper text before or after the JSON block):

{{
  "summary": "A brief 1-2 sentence overview of your selections.",
  "recommendations": [
    {{
      "restaurant_id": "the_exact_id_from_candidate_list",
      "rank": 1,
      "explanation": "Brief 1-sentence justification matching user profile."
    }}
  ]
}}
"""
        return system_prompt
