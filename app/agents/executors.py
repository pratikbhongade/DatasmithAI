from langchain_groq import ChatGroq
from app.core.config import settings
from app.core.logger import get_logger
from app.agents.prompts import (
    SUMMARIZATION_PROMPT, SENTIMENT_ANALYSIS_PROMPT, CODE_EXPLANATION_PROMPT,
    EXTRACT_ACTION_ITEMS_PROMPT, CONVERSATIONAL_PROMPT, RAG_QA_PROMPT
)

logger = get_logger(__name__)

class Executors:
    def __init__(self):
        self.llm = ChatGroq(
            model=settings.GROQ_MODEL_NAME,
            api_key=settings.GROQ_API_KEY,
            temperature=settings.LLM_EXECUTOR_TEMPERATURE,
            max_retries=0
        )

    async def _stream_text(self, prompt: str, on_token=None) -> str:
        final_text = ""
        async for chunk in self.llm.astream(prompt):
            text_chunk = chunk.content
            if not isinstance(text_chunk, str):
                text_chunk = str(text_chunk)
            final_text += text_chunk
            if on_token:
                await on_token(text_chunk)
        return final_text

    async def execute_summarization(self, content: str, on_token=None) -> str:
        prompt = SUMMARIZATION_PROMPT.format(content=content)
        return await self._stream_text(prompt, on_token)

    async def execute_sentiment_analysis(self, content: str, on_token=None) -> str:
        prompt = SENTIMENT_ANALYSIS_PROMPT.format(content=content)
        return await self._stream_text(prompt, on_token)

    async def execute_code_explanation(self, content: str, on_token=None) -> str:
        prompt = CODE_EXPLANATION_PROMPT.format(content=content)
        return await self._stream_text(prompt, on_token)

    async def execute_extract_action_items(self, content: str, on_token=None) -> str:
        prompt = EXTRACT_ACTION_ITEMS_PROMPT.format(content=content)
        return await self._stream_text(prompt, on_token)

    async def execute_conversational(self, content: str, on_token=None) -> str:
        prompt = CONVERSATIONAL_PROMPT.format(content=content)
        return await self._stream_text(prompt, on_token)

    async def execute_rag_qa(self, query: str, context: str, on_token=None) -> str:
        prompt = RAG_QA_PROMPT.format(context=context, query=query)
        return await self._stream_text(prompt, on_token)
