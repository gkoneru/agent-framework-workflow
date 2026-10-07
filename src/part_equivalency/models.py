"""Typed messages that flow between workflow executors.

Two kinds of models:
* LLM-facing ``response_format`` models (EvaluationResult, ResearchResult, FinalSummaryDraft):
  every field is required and there are no dict fields, so they are compatible with
  OpenAI strict structured outputs.
* Internal workflow messages (PartRequest, CandidateSet, ResearchBatch, Findings, ...):
  free to use dicts/defaults because they never go to the model as a schema.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

Verdict = Literal["equivalent", "not_equivalent", "web_search_needed"]
ResearchVerdict = Literal["equivalent", "not_equivalent", "inconclusive"]


# ---------------------------------------------------------------- workflow input
def new_request_id() -> str:
    return f"REQ-{uuid.uuid4().hex[:8].upper()}"


class PartRequest(BaseModel):
    request_id: str = Field(default_factory=new_request_id)
    description: str
    specs: dict[str, str] = Field(default_factory=dict)


class SpecItem(BaseModel):
    name: str
    value: str


class IntakeSpec(BaseModel):
    """Structured output of the intake agent (free text -> normalized request)."""

    description: str
    specs: list[SpecItem]


# ---------------------------------------------------------------- stage 1: Cortex
class Candidate(BaseModel):
    part_number: str
    description: str
    specs: dict[str, str]
    score: float


class CandidateSet(BaseModel):
    request: PartRequest
    candidates: list[Candidate]


# ---------------------------------------------------------------- stage 2: evaluator (LLM, structured)
class Classification(BaseModel):
    part_number: str
    verdict: Verdict
    rationale: str
    missing_attributes: list[str]


class EvaluationResult(BaseModel):
    classifications: list[Classification]


# ---------------------------------------------------------------- stage 3: research (LLM, structured)
class ResearchBatch(BaseModel):
    """Routed to the research executor only when at least one candidate is ambiguous."""

    request: PartRequest
    candidates: list[Candidate]
    classifications: list[Classification]


class ResearchResult(BaseModel):
    part_number: str
    verdict: ResearchVerdict
    findings: str
    sources: list[str]


# ---------------------------------------------------------------- stage 4: summary
class Findings(BaseModel):
    request: PartRequest
    candidates: list[Candidate]
    classifications: list[Classification]
    research: list[ResearchResult] = Field(default_factory=list)


class FinalSummaryDraft(BaseModel):
    """What the summarizer LLM is allowed to decide: prose only."""

    summary: str
    next_question: str


class FinalSummary(BaseModel):
    """Workflow output. Equivalency facts are computed by code, prose by the LLM."""

    request_id: str
    recommendation: Literal["use_existing", "create_new"]
    recommended_part: str | None
    equivalent_parts: list[str]
    rejected_parts: list[str]
    unresolved_parts: list[str]
    summary: str
    next_question: str


# ---------------------------------------------------------------- streaming + HITL
class Progress(BaseModel):
    """Intermediate output streamed to the user while the workflow runs."""

    stage: Literal["cortex_search", "evaluate", "web_research"]
    status: Literal["started", "progress", "completed"]
    message: str
    data: dict[str, Any] | None = None

    def __str__(self) -> str:  # nice console rendering
        return f"[{self.stage}:{self.status}] {self.message}"


class DecisionRequest(BaseModel):
    """Sent with ctx.request_info(...) by the optional decision gate (workflow-hosted HITL)."""

    question: str
    options: list[str]
    summary: FinalSummary


class DecisionOutcome(BaseModel):
    request_id: str
    decision: Literal["use_existing", "create_new"]
    part_number: str | None
    message: str
