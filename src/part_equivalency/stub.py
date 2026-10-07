"""Deterministic offline chat client so the sample and the tests run with zero credentials.

It imitates the four structured sub-agents (intake, evaluator, web researcher, summarizer) by parsing
the JSON block that ``llm.with_input`` appends to every prompt, and it imitates the outer agent's
tool-calling decisions with a few regexes. Latency is simulated so streaming is visible; scale it
with ``STUB_LATENCY_SCALE`` (the tests use a small value; ``0`` disables sleeping).

This is NOT a model. Replace it by configuring a real provider (see llm.py).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from collections.abc import AsyncIterable, Awaitable, Sequence
from typing import Any

from agent_framework import (
    BaseChatClient,
    ChatMiddlewareLayer,
    ChatResponse,
    ChatResponseUpdate,
    Content,
    FunctionInvocationLayer,
    Message,
    ResponseStream,
)

from .llm import INPUT_END, INPUT_START
from .models import EvaluationResult, FinalSummaryDraft, IntakeSpec, ResearchResult


async def simulated_latency(seconds: float) -> None:
    scale = float(os.getenv("STUB_LATENCY_SCALE", "1"))
    if scale > 0:
        await asyncio.sleep(seconds * scale)


def _result_text(raw: Any) -> str:
    if isinstance(raw, list) and raw and getattr(raw[0], "text", None):
        return str(raw[0].text)
    return str(raw)


_STUB_DATASHEETS = {  # pretend web results for ambiguous parts
    "FAN-12V-120": (
        "equivalent",
        "Datasheet rev C lists 0.25A @ 12VDC, 3-pin.",
        "https://example.com/ds/FAN-12V-120.pdf",
    ),
    "FAN-12V-120-HS": (
        "not_equivalent",
        "Datasheet lists 0.45A and a 4-pin PWM connector.",
        "https://example.com/ds/FAN-12V-120-HS.pdf",
    ),
}


def _extract_input(messages: Sequence[Message]) -> Any:
    for m in reversed(messages):
        text = m.text or ""
        if INPUT_START in text:
            return json.loads(text.split(INPUT_START, 1)[1].split(INPUT_END, 1)[0])
    return None


def _last_summary(messages: Sequence[Message]) -> dict[str, Any] | None:
    """Most recent check_part_equivalency result in the conversation history (stub 'memory')."""
    for msg in reversed(messages):
        for c in msg.contents:
            if c.type == "function_result":
                raw = _result_text(c.result)
                try:
                    data = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                if isinstance(data, dict) and "request_id" in data:
                    return data
    return None


def _parse_specs(text: str) -> dict[str, str]:
    specs: dict[str, str] = {}
    if m := re.search(r"(\d+)\s*V(?:DC)?\b", text, re.I):
        specs["voltage"] = f"{m.group(1)}VDC"
    if m := re.search(r"(\d+)\s*mm", text, re.I):
        specs["size"] = f"{m.group(1)}mm"
    if m := re.search(r"(\d+(?:\.\d+)?)\s*A\b", text):
        specs["current"] = f"{m.group(1)}A"
    if m := re.search(r"(\d)\s*-?\s*pin", text, re.I):
        specs["connector"] = f"{m.group(1)}-pin"
    return specs


class StubChatClient(FunctionInvocationLayer[Any], ChatMiddlewareLayer[Any], BaseChatClient[Any]):
    """Deterministic fake LLM. Picks behaviour from options['response_format'] / tool results."""

    OTEL_PROVIDER_NAME = "stub"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(middleware=[], **kwargs)

    def _inner_get_response(  # type: ignore[override]
        self, *, messages: Sequence[Message], stream: bool, options: dict[str, Any], **kwargs: Any
    ) -> Awaitable[ChatResponse] | ResponseStream[ChatResponseUpdate, ChatResponse]:
        response_format = options.get("response_format")

        async def produce() -> list[Content]:
            delay = 0.5
            if response_format is ResearchResult:  # different latencies -> visible as_completed ordering
                delay = {"FAN-12V-120": 0.8, "FAN-12V-120-HS": 1.8}.get(
                    (_extract_input(messages) or {}).get("candidate", {}).get("part_number"), 1.0
                )
            await simulated_latency(delay)
            return self._decide(messages, options, response_format)

        if not stream:

            async def _get() -> ChatResponse:
                return ChatResponse(
                    messages=[Message(role="assistant", contents=await produce())],
                    response_format=response_format,
                )

            return _get()

        async def _stream() -> AsyncIterable[ChatResponseUpdate]:
            for content in await produce():
                if content.type == "text" and content.text:
                    for word in re.findall(r"\S+\s*", content.text):  # token-ish streaming
                        await simulated_latency(0.01)
                        yield ChatResponseUpdate(role="assistant", contents=[Content.from_text(text=word)])
                else:
                    yield ChatResponseUpdate(role="assistant", contents=[content])

        return ResponseStream(
            _stream(), finalizer=lambda ups: ChatResponse.from_updates(ups, output_format_type=response_format)
        )

    # ------------------------------------------------------------------ fake "reasoning"
    def _decide(self, messages: Sequence[Message], options: dict[str, Any], fmt: Any) -> list[Content]:
        data = _extract_input(messages)
        if fmt is IntakeSpec:
            text = data["text"]
            specs = [{"name": k, "value": v} for k, v in _parse_specs(text).items()]
            return [Content.from_text(text=json.dumps({"description": text, "specs": specs}))]
        if fmt is EvaluationResult:
            return [Content.from_text(text=self._evaluate(data).model_dump_json())]
        if fmt is ResearchResult:
            return [Content.from_text(text=self._research(data).model_dump_json())]
        if fmt is FinalSummaryDraft:
            return [Content.from_text(text=self._summarize(data).model_dump_json())]
        return self._master(messages, options)

    @staticmethod
    def _evaluate(data: dict[str, Any]) -> EvaluationResult:
        wanted: dict[str, str] = data["request"]["specs"]
        out = []
        for c in data["candidates"]:
            conflicts = [k for k, v in wanted.items() if k in c["specs"] and c["specs"][k] != v]
            missing = [k for k in wanted if k not in c["specs"]]
            if conflicts:
                verdict, why = (
                    "not_equivalent",
                    "Conflicting " + ", ".join(f"{k} ({c['specs'][k]} vs {wanted[k]})" for k in conflicts),
                )
            elif missing:
                verdict, why = "web_search_needed", "Catalog data lacks " + ", ".join(missing)
            else:
                verdict, why = "equivalent", "All requested attributes match"
            out.append(
                {
                    "part_number": c["part_number"],
                    "verdict": verdict,
                    "rationale": why,
                    "missing_attributes": missing if verdict == "web_search_needed" else [],
                }
            )
        return EvaluationResult.model_validate({"classifications": out})

    @staticmethod
    def _research(data: dict[str, Any]) -> ResearchResult:
        pn = data["candidate"]["part_number"]
        verdict, findings, url = _STUB_DATASHEETS.get(pn, ("inconclusive", "No authoritative source found.", ""))
        return ResearchResult(part_number=pn, verdict=verdict, findings=findings, sources=[url] if url else [])

    @staticmethod
    def _summarize(data: dict[str, Any]) -> FinalSummaryDraft:
        eq, rec = data["equivalent_parts"], data["recommended_part"]
        if rec:
            summary = (
                f"{len(eq)} existing part(s) are equivalent to your request: {', '.join(eq)}. "
                f"{rec} is the best match. Rejected: {', '.join(data['rejected_parts']) or 'none'}."
            )
            question = f"Do you want to use existing part {rec}, or create a new part request?"
        else:
            summary = "No existing part is a confirmed equivalent."
            question = "Do you want to create a new part request?"
        return FinalSummaryDraft(summary=summary, next_question=question)

    @staticmethod
    def _master(messages: Sequence[Message], options: dict[str, Any]) -> list[Content]:
        last = messages[-1]
        results = [c for c in last.contents if c.type == "function_result"]
        if results:
            raw = _result_text(results[0].result)
            try:
                summary = json.loads(raw)
                return [Content.from_text(text=f"{summary['summary']}\n\n{summary['next_question']}")]
            except (ValueError, KeyError, TypeError):
                return [Content.from_text(text=raw.strip().splitlines()[-1] if raw.strip() else "Done.")]
        user_text = last.text or ""
        # Follow-up turn: the user answers the workflow's next_question -> record the decision.
        prior = _last_summary(messages)
        names = {getattr(t, "name", None) for t in options.get("tools") or []}
        choice = re.search(r"\b(use|existing|link)\b|\b(new|create)\b", user_text, re.I)
        if prior and choice and "record_part_decision" in names:
            use_existing = bool(choice.group(1))
            return [
                Content.from_function_call(
                    call_id=f"call_{uuid.uuid4().hex[:8]}",
                    name="record_part_decision",
                    arguments={
                        "request_id": prior["request_id"],
                        "decision": "use_existing" if use_existing else "create_new",
                        "part_number": prior.get("recommended_part") if use_existing else None,
                    },
                )
            ]
        tool = next(
            (t for t in options.get("tools") or [] if getattr(t, "name", None) == "check_part_equivalency"), None
        )
        if tool is not None and re.search(r"part|fan|equivalent|replacement", user_text, re.I):
            props = tool.parameters().get("properties", {}) if hasattr(tool, "parameters") else {}
            args = (
                {"task": user_text}
                if "task" in props  # workflow.as_agent().as_tool() shape
                else {"description": user_text, "specs": _parse_specs(user_text)}
            )
            return [
                Content.from_function_call(
                    call_id=f"call_{uuid.uuid4().hex[:8]}",
                    name="check_part_equivalency",
                    arguments=args,
                )
            ]
        return [Content.from_text(text="Tell me which part you need and I'll check for equivalents.")]
