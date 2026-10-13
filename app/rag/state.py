from typing import Any, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    """Routing classifier decision."""
    route: Literal["direct", "kb"] = Field(
        description="Select 'direct' for greetings/social chit-chat, or 'kb' for HR/policy/support questions."
    )
    reason: str = Field(description="Short rationale for the routing decision.")


class EvidenceGrade(BaseModel):
    """Evidence sufficiency grade."""
    grade: Literal["GOOD", "WEAK"] = Field(
        description="Return 'GOOD' if evidence directly answers the question; otherwise 'WEAK'."
    )
    reason: str = Field(description="Concise reason for the grade.")


class Citation(TypedDict):
    title: str
    url: str
    type: str  # "private_kb" | "web"
    page: int | None


class AgentState(TypedDict):
    """
    Typed LangGraph state schema for Agentic RAG.
    Preserves document metadata, grading outcomes, retry counters, and traces across graph transitions.
    """
    question: str
    search_query: str
    route: str
    kb_docs: list[dict[str, Any]]
    kb_grade: str | None
    web_results: list[dict[str, Any]]
    web_grade: str | None
    answer: str
    citations: list[Citation]
    source_used: str  # "private_kb" | "web_search" | "direct" | "insufficient_evidence"
    retry_count: int
    execution_trace: list[str]
    error_status: str | None
