# orchestrator.py
# Orchestrates requests and routes work through the tool execution graph.

import warnings
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langgraph.prebuilt import create_react_agent

from app.agents.tools import build_tools
from app.agents.prompts import REACT_SYSTEM_PROMPT
from app.agents.cost_estimator import CostEstimator
from app.services.rag_service import RAGService
from app.api.websocket import manager
from app.core.logger import get_logger
from app.core.config import settings
from app.models.schemas import ExtractionResult, AgentResponse

logger = get_logger(__name__)


class AgentOrchestrator:
    MAX_AGENT_CONTENT_CHARS = 10_000
    MAX_AGENT_EVENTS = 400
    MAX_AGENT_OUTPUT_CHARS = 20_000

    def __init__(self):
        # RAGService is expensive (loads BGE-M3 model) — one instance shared
        # across all requests, also passed into tools so they don't reload it.
        self.rag_service = RAGService()

        self.rate_limiter = InMemoryRateLimiter(
            requests_per_second=settings.GROQ_REQUESTS_PER_SECOND,
            check_every_n_seconds=0.1,
            max_bucket_size=1,
        )

        self.llm = ChatGroq(
            model=settings.GROQ_MODEL_NAME,
            api_key=settings.GROQ_API_KEY,
            temperature=settings.LLM_AGENT_TEMPERATURE,
            max_retries=settings.GROQ_MAX_RETRIES,
            max_tokens=settings.GROQ_MAX_TOKENS,
            service_tier=settings.GROQ_SERVICE_TIER,
            rate_limiter=self.rate_limiter,
        )

        tools = build_tools(self.rag_service)

        # Build the LangGraph execution graph and attach the tool set.
        self.agent = create_react_agent(
            model=self.llm,
            tools=tools,
            prompt=SystemMessage(content=REACT_SYSTEM_PROMPT),
        )

        logger.info("ReAct agent initialised with %d tools.", len(tools))

    async def process_request(
        self,
        ext_data: ExtractionResult,
        session_id: str,
        raw_prompt: str = "",
    ) -> AgentResponse:

        final_text = ""
        est_cost = 0.0
        needs_clarification = False
        clarification_question = None

        try:
            # ── 1. Log extraction ────────────────────────────────────────────
            await manager.send_log(
                session_id,
                "Extraction",
                f"Extracted {len(ext_data.content)} characters from {ext_data.source_type}.",
            )

            # ── 2. Ingest non-text files into Qdrant for RAG ─────────────────
            if ext_data.source_type != "text" and len(ext_data.content) > 0:
                await manager.send_log(
                    session_id,
                    "RAG Ingestion",
                    f"Ingesting {ext_data.source_type} content into Qdrant...",
                )
                self.rag_service.ingest_document(ext_data.content, source=ext_data.source_type)

            # ── 3. Build the input message for the agent ─────────────────────
            # Trim content if it is too large for the request.
            content = ext_data.content
            if len(content) > self.MAX_AGENT_CONTENT_CHARS:
                content = content[: self.MAX_AGENT_CONTENT_CHARS].rstrip()
                content += "\n\n[Content truncated to reduce token usage and avoid rate limit issues.]"

            if raw_prompt and ext_data.content.startswith("[User Prompt:"):
                # Follow-up turn: user typed a prompt referencing prior doc context
                user_input = (
                    f"Document context ({ext_data.source_type}):\n"
                    f"{content}\n\n"
                    f"User instruction: {raw_prompt}"
                )
            elif raw_prompt:
                user_input = (
                    f"Content ({ext_data.source_type}):\n"
                    f"{content}\n\n"
                    f"User instruction: {raw_prompt}"
                )
            else:
                user_input = (
                    f"Content ({ext_data.source_type}):\n"
                    f"{content}\n\n"
                    "No explicit instruction was given. Decide the most useful action."
                )

            # ── 4. Cost estimation ───────────────────────────────────────────
            est_cost = CostEstimator.estimate_cost(ext_data.content)
            await manager.send_log(
                session_id,
                "Cost Estimation",
                f"Estimated input cost: ${est_cost:.6f}",
            )

            await manager.send_log(session_id, "Agent", "Starting agent run...")

            event_count = 0
            async for event in self.agent.astream_events(
                {"messages": [HumanMessage(content=user_input)]},
                version="v2",
            ):
                event_count += 1
                kind = event["event"]

                if event_count >= self.MAX_AGENT_EVENTS:
                    await manager.send_log(
                        session_id,
                        "Agent",
                        "Reached max agent event limit; stopping the reasoning loop.",
                    )
                    break

                # ── Tool invocation started ─────────────────────────────────
                if kind == "on_tool_start":
                    tool_name = event.get("name", "unknown_tool")
                    tool_input = event.get("data", {}).get("input", {})

                    # Detect clarification request early so we can flag it
                    if tool_name == "request_clarification":
                        needs_clarification = True
                        clarification_question = tool_input.get("question", "")

                    await manager.send_log(
                        session_id,
                        f"🔧 Action — {tool_name}",
                        f"Agent is calling `{tool_name}`.",
                    )

                # ── Tool execution finished ─────────────────────────────────
                elif kind == "on_tool_end":
                    tool_name = event.get("name", "unknown_tool")
                    await manager.send_log(
                        session_id,
                        f"👁 Observation — {tool_name}",
                        f"`{tool_name}` completed. Agent is processing result...",
                    )

                # ── Capture output tokens from the model stream ───────────────
                # Ignore chunks that are part of tool call bookkeeping.
                elif kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        # Exclude chunks that are part of a tool-call decision
                        if not getattr(chunk, "tool_calls", None):
                            await manager.send_token(session_id, chunk.content)
                            final_text += chunk.content
                            if len(final_text) >= self.MAX_AGENT_OUTPUT_CHARS:
                                await manager.send_log(
                                    session_id,
                                    "Agent",
                                    "Reached max output length; stopping the reasoning loop.",
                                )
                                break

                elif kind == "on_chat_model_end" and final_text.strip():
                    await manager.send_log(
                        session_id,
                        "Agent",
                        "Detected model end event; closing the reasoning loop.",
                    )
                    break

            # ── 6. Finalise ──────────────────────────────────────────────────
            if not final_text.strip():
                final_text = "I was unable to generate a response. Please try again."

            await manager.send_log(session_id, "Completion", "Processing completed.")
            await manager.send_result(
                session_id,
                cost=est_cost,
                extracted_text=ext_data.content,
                needs_clarification=needs_clarification,
                clarification_question=clarification_question,
            )

        except Exception as e:
            logger.exception(f"Agent execution failed: {e}")
            final_text = "An error occurred while processing your request."
            await manager.send_error(session_id, str(e))
            est_cost = 0.0

        return AgentResponse(
            final_output=final_text,
            needs_clarification=needs_clarification,
            clarification_question=clarification_question,
            execution_logs=[],
            estimated_cost=est_cost,
        )
