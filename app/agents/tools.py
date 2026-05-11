# tools.py
# Defines LangChain tool wrappers for the available capabilities.

from langchain_core.tools import tool
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_groq import ChatGroq
from app.core.config import settings
from app.core.logger import get_logger
from app.agents.prompts import (
    SUMMARIZATION_PROMPT,
    SENTIMENT_ANALYSIS_PROMPT,
    CODE_EXPLANATION_PROMPT,
    EXTRACT_ACTION_ITEMS_PROMPT,
    CONVERSATIONAL_PROMPT,
    RAG_QA_PROMPT,
)

logger = get_logger(__name__)

# Content truncation limit — keeps tool calls within LLM context windows.
_MAX_CONTENT_CHARS = 12_000

_TOOL_RATE_LIMITER = InMemoryRateLimiter(
    requests_per_second=settings.GROQ_REQUESTS_PER_SECOND,
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)


def _make_llm() -> ChatGroq:
    """Lightweight factory — ChatGroq is a config object, not a loaded model."""
    return ChatGroq(
        model=settings.GROQ_MODEL_NAME,
        api_key=settings.GROQ_API_KEY,
        temperature=settings.LLM_AGENT_TEMPERATURE,
        max_retries=settings.GROQ_MAX_RETRIES,
        max_tokens=settings.GROQ_TOOL_MAX_TOKENS,
        service_tier=settings.GROQ_SERVICE_TIER,
        rate_limiter=_TOOL_RATE_LIMITER,
    )


def build_tools(rag_service):
    """
    Build tool functions bound to a shared RAG service instance.

    The returned tools are called by the execution graph at runtime.
    """

    @tool
    async def summarize_content(content: str) -> str:
        """
        Generates a structured summary of the provided content.
        Use this when the user wants a summary, overview, TL;DR, or digest of any
        text, document, audio transcript, or video transcript.
        Returns a 1-line headline, 3 key bullet points, and a paragraph summary.
        """
        logger.info("[Tool] summarize_content called.")
        llm = _make_llm()
        prompt = SUMMARIZATION_PROMPT.format(content=content[:_MAX_CONTENT_CHARS])
        result = await llm.ainvoke(prompt)
        return result.content

    @tool
    async def analyze_sentiment(content: str) -> str:
        """
        Analyzes the emotional tone/sentiment of the provided text.
        Use this when the user wants to know if content is positive, negative, or
        neutral — or when they ask for sentiment analysis, mood detection, or tone analysis.
        Returns: Label (positive/negative/neutral), Confidence, and a one-sentence Reason.
        """
        logger.info("[Tool] analyze_sentiment called.")
        llm = _make_llm()
        prompt = SENTIMENT_ANALYSIS_PROMPT.format(content=content[:_MAX_CONTENT_CHARS])
        result = await llm.ainvoke(prompt)
        return result.content

    @tool
    async def explain_code(content: str) -> str:
        """
        Breaks down and explains a code snippet.
        Use this when the user uploads or pastes code and wants it explained,
        reviewed, or analyzed. Covers: language, purpose, bugs, and complexity.
        """
        logger.info("[Tool] explain_code called.")
        llm = _make_llm()
        prompt = CODE_EXPLANATION_PROMPT.format(content=content[:_MAX_CONTENT_CHARS])
        result = await llm.ainvoke(prompt)
        return result.content

    @tool
    async def extract_action_items(content: str) -> str:
        """
        Extracts action items, tasks, or to-dos from the provided content.
        Use this for meeting notes, emails, project documents, or any content
        where the user wants a list of things to do or follow up on.
        Returns a bullet-point list of action items (or a note if none exist).
        """
        logger.info("[Tool] extract_action_items called.")
        llm = _make_llm()
        prompt = EXTRACT_ACTION_ITEMS_PROMPT.format(content=content[:_MAX_CONTENT_CHARS])
        result = await llm.ainvoke(prompt)
        return result.content

    @tool
    async def search_and_answer(query: str) -> str:
        """
        Searches the uploaded document store (RAG) and answers a specific question.
        Use this when the user asks a question about a document they previously uploaded
        (PDF, image, audio, video). Performs hybrid dense+sparse retrieval then generates
        a grounded answer. Do NOT use for general knowledge questions.
        """
        logger.info(f"[Tool] search_and_answer called with query: {query}")
        context = rag_service.hybrid_search(query)
        if not context or context.startswith("Failed") or context.startswith("RAG Search"):
            return "I couldn't find relevant content in the uploaded documents to answer that question."
        llm = _make_llm()
        prompt = RAG_QA_PROMPT.format(context=context, query=query)
        result = await llm.ainvoke(prompt)
        return result.content

    @tool
    async def respond_conversationally(message: str) -> str:
        """
        Responds to general chat messages, greetings, and open-ended knowledge questions.
        Use this when the user is having a conversation, asking general knowledge questions
        (e.g., 'What is LangGraph?', 'How does gravity work?'), or when no document
        has been uploaded and no specific analysis task is requested.
        """
        logger.info("[Tool] respond_conversationally called.")
        llm = _make_llm()
        prompt = CONVERSATIONAL_PROMPT.format(content=message[:_MAX_CONTENT_CHARS])
        result = await llm.ainvoke(prompt)
        return result.content

    @tool
    async def request_clarification(question: str) -> str:
        """
        Signals that the user's intent is ambiguous and a clarification question
        should be posed to the user. Use this ONLY when the request is genuinely
        unclear and no tool choice can be made with confidence.
        The 'question' argument should be a specific, helpful question for the user.
        Example: 'I see meeting notes here. Would you like a summary, action item extraction, or sentiment analysis?'
        """
        logger.info(f"[Tool] request_clarification called: {question}")
        # The return value goes back into the agent context.
        # The orchestrator detects this tool call from the event stream and sets
        # needs_clarification=True on the final result.
        return f"CLARIFICATION_NEEDED: {question}"

    return [
        summarize_content,
        analyze_sentiment,
        explain_code,
        extract_action_items,
        search_and_answer,
        respond_conversationally,
        request_clarification,
    ]
