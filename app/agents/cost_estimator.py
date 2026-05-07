import tiktoken
from app.core.config import settings

class CostEstimator:
    @staticmethod
    def estimate_cost(text: str, model_name: str = settings.GROQ_MODEL_NAME) -> float:
        try:
            encoding = tiktoken.encoding_for_model(model_name)
            token_count = len(encoding.encode(text))
            
            # This is a rough estimation purely based on input tokens
            estimated_cost = token_count * settings.INPUT_TOKEN_COST
            return round(estimated_cost, 6)
        except Exception:
            # Fallback if model not found in tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            token_count = len(encoding.encode(text))
            return round(token_count * settings.INPUT_TOKEN_COST, 6)
