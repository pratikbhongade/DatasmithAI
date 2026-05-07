from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

class UserInput(BaseModel):
    text_prompt: Optional[str] = None
    file_type: Optional[str] = None  # e.g., "image/jpeg", "application/pdf"

class ExtractionResult(BaseModel):
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_type: str  # text, image, pdf, audio, youtube

class ExecutionStep(BaseModel):
    step_name: str
    details: str

class AgentResponse(BaseModel):
    final_output: str
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    execution_logs: List[ExecutionStep] = Field(default_factory=list)
    estimated_cost: Optional[float] = None
