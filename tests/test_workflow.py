"""The deterministic workflow on its own (no outer agent)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import part_equivalency.workflow as wf
from part_equivalency.models import Candidate, Classification, DecisionOutcome, FinalSummary, PartRequest, Progress
from part_equivalency.workflow import build_workflow

FAN = PartRequest(
    description="cooling fan",
    specs={"voltage": "12VDC", "size": "120mm", "current": "0.25A", "connector": "3-pin"},
)


async def run(request: Any, **kwargs: Any) -> tuple[list[Progress], list[Any]]:
    progress, outputs = [], []
    async for event in build_workflow(**kwargs).run(request, stream=True):
        if event.type == "intermediate" and isinstance(event.data, Progress):
            progress.append(event.data)
        elif event.type == "output":
            outputs.append(event.data)
        elif event.type == "failed":
            pytest.fail(f"workflow failed: {event.details}")
    return progress, outputs


def summary_of(outputs: list[Any]) -> FinalSummary:
    return next(o for o in outputs if isinstance(o, FinalSummary))


async def test_full_run_classifies_and_recommends() -> None:
    _, outputs = await run(FAN)
    s = summary_of(outputs)
    assert s.request_id == FAN.request_id
    assert s.recommendation == "use_existing"
    assert s.recommended_part == "FAN-12V-120"
    assert s.equivalent_parts == ["FAN-12V-120", "FAN-12V-120-B"]
    assert sorted(s.rejected_parts) == ["FAN-12V-092", "FAN-12V-120-HS", "FAN-24V-120"]
    assert s.unresolved_parts == []
    assert any(isinstance(o, str) and s.next_question in o for o in outputs)  # chat text when hosted directly


async def test_progress_is_streamed_in_stage_order() -> None:
    progress, _ = await run(FAN)
    stages = [(p.stage, p.status) for p in progress]
    assert stages[:4] == [
        ("cortex_search", "started"),
        ("cortex_search", "completed"),
        ("evaluate", "started"),
        ("evaluate", "completed"),
    ]
    assert stages[-1] == ("web_research", "completed")


async def test_research_results_stream_as_they_complete() -> None:
    progress, _ = await run(FAN)
    done = [p.data["part_number"] for p in progress if p.stage == "web_research" and p.status == "progress"]
    assert done == ["FAN-12V-120", "FAN-12V-120-HS"]  # the stub makes FAN-12V-120-HS slower


async def test_free_text_input_goes_through_intake() -> None:
    _, outputs = await run("I need a replacement cooling fan: 12V DC, 120mm, 0.25A, 3-pin connector")
    assert summary_of(outputs).recommended_part == "FAN-12V-120"


async def test_no_ambiguity_skips_research() -> None:
    progress, outputs = await run(PartRequest(description="any 12V fan", specs={"voltage": "12VDC"}))
    assert not any(p.stage == "web_research" for p in progress)
    assert summary_of(outputs).recommendation == "use_existing"


async def test_research_failure_is_inconclusive_not_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    real = wf.run_structured

    async def flaky(agent: Any, prompt: str, schema: type, **kwargs: Any) -> Any:
        if schema is wf.ResearchResult:
            raise TimeoutError("search backend unavailable")
        return await real(agent, prompt, schema, **kwargs)

    monkeypatch.setattr(wf, "run_structured", flaky)
    progress, outputs = await run(FAN)
    s = summary_of(outputs)
    assert sorted(s.unresolved_parts) == ["FAN-12V-120", "FAN-12V-120-HS"]
    assert s.equivalent_parts == ["FAN-12V-120-B"]
    assert any("Research failed" in p.message for p in progress)


def test_reconcile_fills_missing_and_drops_unknown_verdicts() -> None:
    cands = [Candidate(part_number=p, description="", specs={}, score=1.0) for p in ("A", "B")]
    llm = [
        Classification(part_number="A", verdict="equivalent", rationale="ok", missing_attributes=[]),
        Classification(part_number="ZZZ", verdict="equivalent", rationale="hallucinated", missing_attributes=[]),
    ]
    out = wf._reconcile(cands, llm)
    assert [(c.part_number, c.verdict) for c in out] == [("A", "equivalent"), ("B", "web_search_needed")]


async def test_one_instance_cannot_serve_concurrent_runs() -> None:
    """Why the tool builds a NEW workflow per call (and why a shared ``workflow.as_agent()`` breaks)."""
    shared = build_workflow()

    async def drain(w: Any) -> None:
        async for _ in w.run(FAN.model_copy(), stream=True):
            pass

    results = await asyncio.gather(drain(shared), drain(shared), return_exceptions=True)
    assert any(isinstance(r, Exception) for r in results)

    fresh = await asyncio.gather(drain(build_workflow()), drain(build_workflow()), return_exceptions=True)
    assert fresh == [None, None]


async def test_decision_gate_when_workflow_is_hosted_directly() -> None:
    workflow = build_workflow(with_decision_gate=True)
    requests = [e async for e in workflow.run(FAN, stream=True) if e.type == "request_info"]
    assert len(requests) == 1
    assert requests[0].data.options == ["use_existing", "create_new"]

    outputs = [
        e.data
        async for e in workflow.run(stream=True, responses={requests[0].request_id: "use existing"})
        if e.type == "output"
    ]
    outcome = next(o for o in outputs if isinstance(o, DecisionOutcome))
    assert outcome.decision == "use_existing"
    assert outcome.part_number == "FAN-12V-120"
    assert outcome.request_id == FAN.request_id
