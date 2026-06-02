import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Determine the absolute directory representing the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

class Settings(BaseSettings):
    # LLM Settings
    groq_api_key: Optional[str] = Field(default=None, validation_alias="GROQ_API_KEY")
    llm_model: str = Field(default="llama-3.3-70b-versatile", validation_alias="LLM_MODEL")

    # Local Vector Search & Embeddings
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", 
        validation_alias="EMBEDDING_MODEL_NAME"
    )

    # Data Pathing & Capacity
    data_path: str = Field(default="data/processed/restaurants.sqlite", validation_alias="DATA_PATH")
    max_candidates: int = Field(default=30, validation_alias="MAX_CANDIDATES")

    # Budget Ranges
    budget_low_max: float = Field(default=500.0, validation_alias="BUDGET_LOW_MAX")
    budget_medium_max: float = Field(default=1500.0, validation_alias="BUDGET_MEDIUM_MAX")

    model_config = SettingsConfigDict(
        env_file=os.path.join(PROJECT_ROOT, ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def absolute_data_path(self) -> str:
        """Returns the fully qualified absolute path to the data storage file."""
        if os.path.isabs(self.data_path):
            return self.data_path
        return os.path.abspath(os.path.join(PROJECT_ROOT, self.data_path))

# Instantiate a global settings object for importing across modules
settings = Settings()
