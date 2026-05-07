import pytest
from app.agents.planner import PlannerAgent, PlanOutput
from app.models.schemas import ExtractionResult
from unittest.mock import MagicMock

def test_planner_ambiguity(monkeypatch):
    # Mock LLM to return ambiguous plan
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = PlanOutput(
        is_ambiguous=True,
        clarification_question="Did you want me to summarize this code or explain it?",
        selected_task=""
    )
    
    planner = PlannerAgent()
    planner.llm = mock_llm
    
    plan = planner.plan("def foo():\n  pass")
    
    assert plan.is_ambiguous is True
    assert plan.clarification_question is not None
    assert plan.selected_task == ""

def test_planner_clear_intent(monkeypatch):
    # Mock LLM to return clear plan
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = PlanOutput(
        is_ambiguous=False,
        clarification_question=None,
        selected_task="summarization"
    )
    
    planner = PlannerAgent()
    planner.llm = mock_llm
    
    plan = planner.plan("Please summarize this text: The quick brown fox.")
    
    assert plan.is_ambiguous is False
    assert plan.selected_task == "summarization"
