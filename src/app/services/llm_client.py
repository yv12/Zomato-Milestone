import time
from groq import Groq
from app.config import settings

class LLMClient:
    def __init__(self):
        self.model = settings.llm_model.strip()
        if not self.model:
            self.model = "llama-3.3-70b-versatile"
            
        api_key = settings.groq_api_key
        if not api_key or api_key == "your_groq_api_key_here":
            raise ValueError("GROQ_API_KEY is missing or placeholder in .env file.")
            
        self.client = Groq(api_key=api_key)

    def complete(self, prompt: str) -> str:
        print(f"Sending prompt to Groq API using model {self.model}...")
        
        for attempt in range(2):
            try:
                completion = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model=self.model,
                    temperature=0.3, # Low temperature for structured JSON
                    response_format={"type": "json_object"}
                )
                return completion.choices[0].message.content
            except Exception as e:
                print(f"Groq API Error (Attempt {attempt+1}): {str(e)}")
                if attempt == 1:
                    raise
                time.sleep(2)
        return "{}"
