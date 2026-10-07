"""Deterministic part-equivalency workflow (graph API).

    intake ─▶ cortex_search ─▶ evaluate ─┬─(ResearchBatch)─▶ web_research ─▶ summarize ─▶ [user_decision]
                                         └─(Findings, default)─────────────▶ summarize

* CODE owns the control flow (edges + switch-case routing). LLM sub-agents are called *inside*
  steps and must return structured (Pydantic) output; code validates/reconciles it.
* Every step streams typed ``Progress`` via ``ctx.yield_output`` -> surfaced as
  ``type == "intermediate"`` events because of ``intermediate_output_from="all_other"``.
  The summarizer is the only ``output_from`` executor -> its ``FinalSummary`` is ``type == "output"``.
* Why the graph API (``WorkflowBuilder``) and not the functional ``@workflow`` API: in
  agent-framework-core 1.19 the functional API buffers events until the function returns.
  The graph runner streams events live, which is what the progress UX needs.
* A ``Workflow`` instance runs one run at a time -> call ``build_workflow()`` per run.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

from agent_framework import (
    Agent,
    Case,
    CheckpointStorage,
    Default,
    Executor,
    Message,
    Workflow,
    WorkflowBuilder,
    WorkflowContext,
    handler,
    response_handler,
)
from typing_extensions import Never

from .cortex import search_cortex
from .llm import make_chat_client, run_structured, web_search_tools, with_input
from .models import (
    Candidate,
    CandidateSet,
    Classification,
    DecisionOutcome,
    DecisionRequest,
    EvaluationResult,
    FinalSummary,
    FinalSummaryDraft,
    Findings,
    IntakeSpec,
    PartRequest,
    Progress,
    ResearchBatch,
    ResearchResult,
)

# =========================================================================== sub-agents
INTAKE_INSTRUCTIONS = (
    "Extract a normalized MRO part request from the user's text. Return the description and each "
    "technical attribute as name/value (names: voltage, size, current, connector, ...)."
)
EVALUATOR_INSTRUCTIONS = (
    "You are an MRO part-equivalency evaluator. For EVERY candidate, compare its catalog specs with the "
    "requested specs and classify it: 'equivalent' (all requested attributes match), 'not_equivalent' "
    "(any requested attribute conflicts), or 'web_search_needed' (no conflict but required attributes "
    "are missing from catalog data; list them in missing_attributes). Never guess missing values."
)
RESEARCH_INSTRUCTIONS = (
    "You research ONE candidate part on the web (manufacturer datasheets first). Decide whether it is "
    "equivalent to the request for the missing attributes. Cite sources. Use 'inconclusive' when no "
    "authoritative source exists."
)
SUMMARY_INSTRUCTIONS = (
    "Write a short summary for a maintenance planner using ONLY the provided facts (do not change "
    "verdicts), and a single next_question asking whether to use the recommended existing part or "
    "create a new part request."
)


@dataclass
class PartAgents:
    intake: Agent
    evaluator: Agent
    researcher: Agent
    summarizer: Agent

    @classmethod
    def create(cls) -> PartAgents:
        # One client per agent keeps it simple; they can be different models/providers.
        return cls(
            intake=Agent(name="intake", instructions=INTAKE_INSTRUCTIONS, client=make_chat_client()),
            evaluator=Agent(name="evaluator", instructions=EVALUATOR_INSTRUCTIONS, client=make_chat_client()),
            researcher=Agent(
                name="web_researcher",
                instructions=RESEARCH_INSTRUCTIONS,
                client=make_chat_client(),
                tools=web_search_tools(),  # hosted web search on OpenAI Responses; [] for stub
            ),
            summarizer=Agent(name="summarizer", instructions=SUMMARY_INSTRUCTIONS, client=make_chat_client()),
        )


# =========================================================================== executors
class Intake(Executor):
    """Accepts free text (AG-UI sends list[Message]) or an already-structured PartRequest."""

    def __init__(self, agent: Agent) -> None:
        super().__init__(id="intake")
        self.agent = agent

    @handler
    async def from_request(self, request: PartRequest, ctx: WorkflowContext[PartRequest]) -> None:
        await ctx.send_message(request)

    @handler
    async def from_text(self, text: str, ctx: WorkflowContext[PartRequest]) -> None:
        await ctx.send_message(await self._extract(text))

    @handler
    async def from_messages(self, messages: list[Message], ctx: WorkflowContext[PartRequest]) -> None:
        user_text = next((m.text for m in reversed(messages) if m.role == "user" and m.text), "")
        await ctx.send_message(await self._extract(user_text))

    async def _extract(self, text: str) -> PartRequest:
        spec = await run_structured(self.agent, with_input("Extract the part request.", {"text": text}), IntakeSpec)
        return PartRequest(
            description=spec.description,
            specs={s.name.lower(): s.value for s in spec.specs},
        )


class CortexSearch(Executor):
    """Stage 1 - pure code, no LLM."""

    def __init__(self, top_k: int = 5) -> None:
        super().__init__(id="cortex_search")
        self.top_k = top_k

    @handler
    async def search(self, request: PartRequest, ctx: WorkflowContext[CandidateSet, Progress]) -> None:
        await ctx.yield_output(
            Progress(stage="cortex_search", status="started", message=f"Searching Cortex for {request.specs}")
        )
        candidates = await search_cortex(request, self.top_k)
        await ctx.yield_output(
            Progress(
                stage="cortex_search",
                status="completed",
                message=f"Found {len(candidates)} candidate parts",
                data={"candidates": [c.model_dump() for c in candidates]},
            )
        )
        await ctx.send_message(CandidateSet(request=request, candidates=candidates))


class Evaluate(Executor):
    """Stage 2 - LLM sub-agent with structured output; code reconciles and routes."""

    def __init__(self, agent: Agent) -> None:
        super().__init__(id="evaluate")
        self.agent = agent

    @handler
    async def evaluate(self, cs: CandidateSet, ctx: WorkflowContext[ResearchBatch | Findings, Progress]) -> None:
        await ctx.yield_output(
            Progress(stage="evaluate", status="started", message=f"Evaluating {len(cs.candidates)} candidates")
        )
        result = EvaluationResult(classifications=[])
        if cs.candidates:
            payload = {"request": cs.request.model_dump(), "candidates": [c.model_dump() for c in cs.candidates]}
            result = await run_structured(
                self.agent, with_input("Classify every candidate.", payload), EvaluationResult
            )
        classifications = _reconcile(cs.candidates, result.classifications)

        counts = {
            v: sum(c.verdict == v for c in classifications)
            for v in ("equivalent", "not_equivalent", "web_search_needed")
        }
        await ctx.yield_output(
            Progress(
                stage="evaluate",
                status="completed",
                message=f"{counts['equivalent']} equivalent, {counts['not_equivalent']} not equivalent, "
                f"{counts['web_search_needed']} need web research",
                data={"classifications": [c.model_dump() for c in classifications]},
            )
        )
        # Deterministic routing decision. The switch-case edge group below dispatches on the TYPE.
        if counts["web_search_needed"]:
            await ctx.send_message(
                ResearchBatch(request=cs.request, candidates=cs.candidates, classifications=classifications)
            )
        else:
            await ctx.send_message(
                Findings(request=cs.request, candidates=cs.candidates, classifications=classifications)
            )


class WebResearch(Executor):
    """Stage 3 - one research sub-agent call per ambiguous candidate, bounded concurrency.

    The number of ambiguous parts is only known at runtime, so we parallelize *inside* the step
    (asyncio) and stream each result as soon as it lands (as_completed). Static fan-out edges are
    for a fixed set of branches known at build time.
    """

    def __init__(self, agent: Agent, max_concurrency: int = 4, background: bool = False) -> None:
        super().__init__(id="web_research")
        self.agent = agent
        self.max_concurrency = max_concurrency
        self.background = background

    @handler
    async def research(self, batch: ResearchBatch, ctx: WorkflowContext[Findings, Progress]) -> None:
        by_pn = {c.part_number: c for c in batch.candidates}
        todo = [c for c in batch.classifications if c.verdict == "web_search_needed"]
        for c in todo:
            await ctx.yield_output(
                Progress(
                    stage="web_research",
                    status="started",
                    message=f"Researching {c.part_number} (missing: {', '.join(c.missing_attributes) or 'n/a'})",
                    data={"part_number": c.part_number},
                )
            )

        sem = asyncio.Semaphore(self.max_concurrency)

        async def one(cls: Classification) -> ResearchResult:
            async with sem:
                payload = {
                    "request": batch.request.model_dump(),
                    "candidate": by_pn[cls.part_number].model_dump(),
                    "missing_attributes": cls.missing_attributes,
                }
                try:
                    return await run_structured(
                        self.agent,
                        with_input("Research this candidate.", payload),
                        ResearchResult,
                        background=self.background,
                    )
                except Exception as exc:  # one failed lookup must not fail the whole workflow
                    return ResearchResult(
                        part_number=cls.part_number,
                        verdict="inconclusive",
                        findings=f"Research failed: {exc}",
                        sources=[],
                    )

        results: list[ResearchResult] = []
        for next_done in asyncio.as_completed([one(c) for c in todo]):
            r = await next_done
            results.append(r)
            await ctx.yield_output(
                Progress(
                    stage="web_research",
                    status="progress",
                    message=f"{r.part_number}: {r.verdict} - {r.findings}",
                    data=r.model_dump(),
                )
            )
        await ctx.yield_output(
            Progress(stage="web_research", status="completed", message=f"Researched {len(results)} part(s)")
        )
        await ctx.send_message(
            Findings(
                request=batch.request,
                candidates=batch.candidates,
                classifications=batch.classifications,
                research=results,
            )
        )


class Summarize(Executor):
    """Stage 4 - code computes the verdicts; the LLM only writes prose (structured)."""

    def __init__(self, agent: Agent, forward_to_decision: bool) -> None:
        super().__init__(id="summarize")
        self.agent = agent
        self.forward_to_decision = forward_to_decision

    @handler
    async def summarize(self, f: Findings, ctx: WorkflowContext[FinalSummary, FinalSummary | str]) -> None:
        research = {r.part_number: r for r in f.research}
        final: dict[str, str] = {}
        for c in f.classifications:
            final[c.part_number] = (
                research[c.part_number].verdict
                if c.verdict == "web_search_needed" and c.part_number in research
                else c.verdict
            )
        score = {c.part_number: c.score for c in f.candidates}
        equivalent = sorted([p for p, v in final.items() if v == "equivalent"], key=lambda p: -score.get(p, 0))
        rejected = [p for p, v in final.items() if v == "not_equivalent"]
        unresolved = [p for p, v in final.items() if v not in ("equivalent", "not_equivalent")]
        recommended = equivalent[0] if equivalent else None

        facts = {
            "request": f.request.model_dump(),
            "equivalent_parts": equivalent,
            "rejected_parts": rejected,
            "unresolved_parts": unresolved,
            "recommended_part": recommended,
            "research": [r.model_dump() for r in f.research],
        }
        draft = await run_structured(self.agent, with_input("Summarize these facts.", facts), FinalSummaryDraft)

        summary = FinalSummary(
            request_id=f.request.request_id,
            recommendation="use_existing" if recommended else "create_new",
            recommended_part=recommended,
            equivalent_parts=equivalent,
            rejected_parts=rejected,
            unresolved_parts=unresolved,
            summary=draft.summary,
            next_question=draft.next_question,
        )
        await ctx.yield_output(summary)  # typed result (tool return value / UI card)
        await ctx.yield_output(f"{summary.summary}\n\n{summary.next_question}")  # chat text when hosted directly
        if self.forward_to_decision:
            await ctx.send_message(summary)


class UserDecision(Executor):
    """Optional human-in-the-loop gate, for hosting the workflow DIRECTLY (not as a tool).

    ``request_info`` pauses the run until a response arrives (``workflow.run(responses=...)``).
    Do NOT enable it when the workflow runs inside a tool call: a tool cannot pause the outer
    agent, so the question is lost. In the tool pattern the outer agent asks the question instead.
    """

    def __init__(self) -> None:
        super().__init__(id="user_decision")

    @handler
    async def ask(self, summary: FinalSummary, ctx: WorkflowContext[Never, DecisionOutcome]) -> None:
        options = ["use_existing", "create_new"] if summary.recommended_part else ["create_new"]
        await ctx.request_info(DecisionRequest(question=summary.next_question, options=options, summary=summary), str)

    @response_handler
    async def on_decision(
        self, original: DecisionRequest, answer: str, ctx: WorkflowContext[Never, DecisionOutcome]
    ) -> None:
        use_existing = "existing" in answer.lower() or answer.strip().lower() in ("yes", "y", "use")
        part = original.summary.recommended_part if use_existing else None
        decision = "use_existing" if part else "create_new"
        msg = (
            f"Linked request {original.summary.request_id} to existing part {part}."
            if part
            else f"Created new part request for {original.summary.request_id}."
        )
        await ctx.yield_output(
            DecisionOutcome(request_id=original.summary.request_id, decision=decision, part_number=part, message=msg)
        )


# =========================================================================== helpers + builder
def _reconcile(candidates: list[Candidate], llm: list[Classification]) -> list[Classification]:
    """Deterministic guard: exactly one classification per candidate, unknown -> web_search_needed."""
    by_pn = {c.part_number: c for c in llm}
    return [
        by_pn.get(c.part_number)
        or Classification(
            part_number=c.part_number,
            verdict="web_search_needed",
            rationale="Evaluator returned no verdict",
            missing_attributes=[],
        )
        for c in candidates
    ]


def build_workflow(
    *,
    agents: PartAgents | None = None,
    with_decision_gate: bool = False,
    checkpoint_storage: CheckpointStorage | None = None,
) -> Workflow:
    """Build a NEW workflow instance. Instances are single-run-at-a-time, so build one per run/thread."""
    agents = agents or PartAgents.create()
    intake = Intake(agents.intake)
    cortex = CortexSearch()
    evaluate = Evaluate(agents.evaluator)
    research = WebResearch(agents.researcher, background=os.getenv("RESEARCH_BACKGROUND") == "1")
    summarize = Summarize(agents.summarizer, forward_to_decision=with_decision_gate)
    gate = UserDecision() if with_decision_gate else None

    builder = (
        WorkflowBuilder(
            name="part_equivalency",
            description="Cortex search -> evaluate -> web research -> summarize",
            start_executor=intake,
            output_from=[summarize] + ([gate] if gate else []),
            intermediate_output_from="all_other",  # every other executor's yield_output streams as 'intermediate'
            checkpoint_storage=checkpoint_storage,
        )
        .add_edge(intake, cortex)
        .add_edge(cortex, evaluate)
        .add_switch_case_edge_group(
            evaluate,
            [
                Case(condition=lambda m: isinstance(m, ResearchBatch), target=research),
                Default(target=summarize),
            ],
        )
        .add_edge(research, summarize)
    )
    if gate:
        builder = builder.add_edge(summarize, gate)
    return builder.build()
