from pydantic import BaseModel, Field
from typing import Optional
from langchain_groq import ChatGroq
from app.core.config import settings
from app.core.logger import get_logger
from app.agents.prompts import PLANNER_PROMPT

logger = get_logger(__name__)

class PlanOutput(BaseModel):
    is_ambiguous: bool = Field(description="True if the user's intent is unclear or if multiple distinct tasks are plausible. False if the intent is clear.")
    clarification_question: Optional[str] = Field(None, description="If ambiguous, the follow-up question to ask the user.")
    selected_task: Optional[str] = Field(None, description="If clear, one of: 'summarization', 'sentiment_analysis', 'code_explanation', 'conversational', 'extract_action_items', 'rag_qa'. Null/empty if ambiguous.")

    @classmethod
    def model_validator_is_ambiguous(cls, v):
        if isinstance(v, str):
            return v.lower() not in ("false", "0", "no", "null", "none", "")
        return v

class PlannerAgent:
    def __init__(self):
        self.llm = ChatGroq(
            model=settings.GROQ_MODEL_NAME,
            api_key=settings.GROQ_API_KEY,
            temperature=settings.LLM_PLANNER_TEMPERATURE,
            max_retries=0
        ).with_structured_output(PlanOutput)

    async def plan(self, content: str) -> PlanOutput:
        prompt = PLANNER_PROMPT.format(content=content)
        
        # TODO: Consider passing conversation history into the planner for multi-turn awareness.
        logger.info("Planner analyzing intent...")
        result = await self.llm.ainvoke(prompt)
        return result
