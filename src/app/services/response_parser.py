import json
import re
from typing import Dict, Any

class ResponseParser:
    @staticmethod
    def parse(raw_response: str) -> Dict[str, Any]:
        """
        Extracts and parses JSON from the LLM response.
        Handles cases where the LLM might have wrapped the JSON in markdown blocks.
        """
        if not raw_response:
            return {"summary": "Empty response received.", "recommendations": []}
            
        # Try to parse directly first
        try:
            return json.loads(raw_response)
        except json.JSONDecodeError:
            pass
            
        # If direct parse fails, try to extract JSON using regex (between { and })
        try:
            # Look for ```json ... ``` or just {...}
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Fallback to finding the first { and last }
                start = raw_response.find('{')
                end = raw_response.rfind('}') + 1
                if start != -1 and end != 0:
                    json_str = raw_response[start:end]
                else:
                    raise ValueError("No JSON object found in response.")
                    
            return json.loads(json_str)
        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
            print(f"Raw response was: {raw_response[:200]}...")
            # Return empty skeleton if parsing completely fails
            return {"summary": "Failed to generate recommendations due to an AI error.", "recommendations": []}
