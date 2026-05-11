from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "DataSmith AI"
    API_V1_STR: str = "/api/v1"
    
    # LLM Settings
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL_NAME: str = "llama-3.3-70b-versatile"
    GROQ_MAX_TOKENS: int = 1024
    GROQ_TOOL_MAX_TOKENS: int = 512
    GROQ_MAX_RETRIES: int = 2
    GROQ_REQUESTS_PER_SECOND: float = 1.0
    GROQ_SERVICE_TIER: str = "on_demand"
    
    # Model Costs (Groq Llama 3.3 70B pricing: ~$0.59 per 1M input tokens)
    INPUT_TOKEN_COST: float = 0.00000059
    OUTPUT_TOKEN_COST: float = 0.00000079
    
    # RAG Settings
    QDRANT_PATH: str = "./qdrant_db"
    QDRANT_COLLECTION: str = "datasmith"
    BGE_M3_MODEL_PATH: str = "D:/EmbeddingModels/bge-m3"
    RAG_CHUNK_SIZE: int = 500
    RAG_CHUNK_OVERLAP: int = 50
    RAG_TOP_K: int = 3
    
    # Audio & Extraction Settings
    WHISPER_MODEL_NAME: str = "base"
    PDF_OCR_THRESHOLD: int = 50
    
    # Agent Hyperparameters
    LLM_AGENT_TEMPERATURE: float = 0.1

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

settings = Settings()
