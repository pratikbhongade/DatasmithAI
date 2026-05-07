from typing import List
from app.models.schemas import ExtractionResult, AgentResponse, ExecutionStep
from app.agents.planner import PlannerAgent
from app.agents.executors import Executors
from app.agents.cost_estimator import CostEstimator
from app.services.rag_service import RAGService
from app.api.websocket import manager
from app.core.logger import get_logger
from app.core.config import settings

logger = get_logger(__name__)

class AgentOrchestrator:
    def __init__(self):
        self.planner = PlannerAgent()
        self.executors = Executors()
        self.rag_service = RAGService()

    async def process_request(self, ext_data: ExtractionResult, session_id: str, raw_prompt: str = "") -> AgentResponse:
        logs: List[ExecutionStep] = []
        res_text = "An error occurred while trying to process your request."
        est_cost = 0.0
        
        try:
            # print(f"extracted chars: {len(ext_data.content)}")
            await manager.send_log(session_id, "Extraction", f"Extracted {len(ext_data.content)} characters from {ext_data.source_type}.")

            if ext_data.source_type != "text" and len(ext_data.content) > 0:
                await manager.send_log(session_id, "RAG Ingestion", f"Ingesting {ext_data.source_type} content into Qdrant...")
                self.rag_service.ingest_document(ext_data.content, source=ext_data.source_type)

            await manager.send_log(session_id, "Planning", "Analyzing intent...")
            # if this is a follow-up reply with injected doc context, plan on just the user's instruction
            planning_content = raw_prompt if (raw_prompt and ext_data.content.startswith("[User Prompt:")) else ext_data.content
            # truncate - planner only needs a snippet to understand intent, not the full transcript
            plan = await self.planner.plan(planning_content[:800])
            
            # print("plan result:", plan.selected_task)
            # audio/video with no explicit instruction -> just summarize it
            if plan.is_ambiguous and ext_data.source_type in ("audio", "video"):
                await manager.send_log(session_id, "Auto-routing", "Audio detected with no specific task — defaulting to summarization.")
                plan.is_ambiguous = False
                plan.selected_task = "summarization"

            if plan.is_ambiguous:
                await manager.send_log(session_id, "Ambiguity Detected", "Intent is unclear. Asking follow-up question.")
                res_text = "I need a little more context to help you effectively."
                await manager.send_result(session_id, cost=0.0, extracted_text=ext_data.content, needs_clarification=True, clarification_question=plan.clarification_question)
                return AgentResponse(
                    final_output=res_text,
                    needs_clarification=True,
                    clarification_question=plan.clarification_question,
                    execution_logs=logs,
                    estimated_cost=0.0
                )

            await manager.send_log(session_id, "Task Selected", f"Planner selected task: {plan.selected_task}")

            est_cost = CostEstimator.estimate_cost(ext_data.content)
            await manager.send_log(session_id, "Cost Estimation", f"Estimated input cost: ${est_cost:.6f}")

            await manager.send_log(session_id, "Execution", "Running specific executor agent...")
            
            async def on_token(token: str):
                await manager.send_token(session_id, token)

            if plan.selected_task == "rag_qa":
                # TODO: Implement re-ranking for better retrieval accuracy
                await manager.send_log(session_id, "Hybrid Search", "Searching relevant document chunks...")
                context = self.rag_service.hybrid_search(raw_prompt)
                res_text = await self.executors.execute_rag_qa(query=raw_prompt, context=context, on_token=on_token)
            else:
                executor_method = getattr(self.executors, f"execute_{plan.selected_task}", self.executors.execute_conversational)
                res_text = await executor_method(ext_data.content, on_token=on_token)
                
            await manager.send_log(session_id, "Completion", "Task executed successfully.")
            await manager.send_result(session_id, cost=est_cost, extracted_text=ext_data.content)
                
        except Exception as e:
            logger.exception(f"Execution failed in Orchestrator: {e}")
            res_text = "An error occurred while trying to process your request."
            await manager.send_error(session_id, str(e))
            est_cost = 0.0

        return AgentResponse(
            final_output=res_text,
            needs_clarification=False,
            execution_logs=logs,
            estimated_cost=est_cost
        )
