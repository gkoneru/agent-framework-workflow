"""ProgressStreamingAgent: tool progress becomes part of the agent's own response stream."""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

import pytest
from agent_framework import Agent, FunctionInvocationContext, tool

from part_equivalency import FinalSummary, ProgressStreamingAgent, create_agent
from part_equivalency.streaming import bind_tool_call_id, progress_message_id, progress_reporter
from part_equivalency.stub import StubChatClient

from .conftest import PART_REQUEST, collect


async def test_progress_is_streamed_between_tool_call_and_result() -> None:
    contents = await collect(create_agent(), PART_REQUEST)
    kinds = [c.type for c in contents]

    call_at = next(
        i for i, c in enumerate(contents) if c.type == "function_call" and c.name == "check_part_equivalency"
    )
    result_at = kinds.index("function_result")
    reasoning_at = [i for i, k in enumerate(kinds) if k == "text_reasoning"]
    assert len(reasoning_at) >= 8
    assert all(call_at < i < result_at for i in reasoning_at)

    call_id = contents[call_at].call_id
    reasoning = [contents[i] for i in reasoning_at]
    assert {r.id for r in reasoning} == {progress_message_id(call_id)}
    assert all(r.additional_properties["tool_call_id"] == call_id for r in reasoning)
    assert reasoning[0].additional_properties["progress"]["stage"] == "cortex_search"

    # The LLM receives only the compact typed summary, not the progress log.
    raw = contents[result_at].result
    summary = FinalSummary.model_validate_json(raw if isinstance(raw, str) else raw[0].text)
    assert summary.recommended_part == "FAN-12V-120"
    assert any(c.type == "text" for c in contents[result_at:])


async def test_final_response_is_available_from_the_stream() -> None:
    stream = create_agent().run(PART_REQUEST, stream=True)
    updates = [u async for u in stream]
    final = await stream.get_final_response()
    assert updates
    assert "FAN-12V-120" in final.text


async def test_non_streaming_run_passes_through() -> None:
    response = await create_agent().run(PART_REQUEST)
    assert "FAN-12V-120" in response.text
    assert not any(c.type == "text_reasoning" for m in response.messages for c in m.contents)


async def test_plain_chat_has_no_progress() -> None:
    contents = await collect(create_agent(), "Hi! What do you help with?")
    assert [c.type for c in contents if c.type != "text"] == []


async def test_unwrapped_agent_tools_still_work() -> None:
    from part_equivalency import create_master_agent

    contents = await collect(create_master_agent(), PART_REQUEST)
    assert not any(c.type == "text_reasoning" for c in contents)  # the reporter is a no-op without the wrapper
    assert any(c.type == "function_result" for c in contents)


def _agent_with(tool_fn: Any) -> ProgressStreamingAgent:
    # The stub calls a tool named check_part_equivalency, so reuse that name for generic tools.
    return ProgressStreamingAgent(
        Agent(name="t", client=StubChatClient(), tools=[tool_fn], middleware=[bind_tool_call_id])
    )


async def test_errors_inside_the_inner_stream_propagate() -> None:
    class Boom(Exception):
        pass

    class Failing(Agent):
        def run(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
            async def gen() -> Any:
                raise Boom("inner failure")
                yield  # pragma: no cover

            return gen()

    agent = ProgressStreamingAgent(Failing(name="f", client=StubChatClient()))
    with pytest.raises(Boom):
        await collect(agent, "hi")


async def test_cancelled_consumer_cancels_the_inner_run() -> None:
    """E.g. the browser disconnects: the server cancels the consuming task; the tool must not keep running."""
    first_progress = asyncio.Event()
    tool_cancelled = asyncio.Event()

    @tool(name="check_part_equivalency", approval_mode="never_require")
    async def slow(description: Annotated[str, "d"], specs: dict[str, str], ctx: FunctionInvocationContext) -> str:
        report = progress_reporter(ctx)
        try:
            for i in range(1000):
                await report(f"step {i}")
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            tool_cancelled.set()
            raise
        return "done"

    async def consume() -> None:
        async for update in _agent_with(slow).run(PART_REQUEST, stream=True):
            if any(c.type == "text_reasoning" for c in update.contents):
                first_progress.set()

    consumer = asyncio.create_task(consume())
    await asyncio.wait_for(first_progress.wait(), timeout=5)
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer
    await asyncio.wait_for(tool_cancelled.wait(), timeout=2)


async def test_caller_function_invocation_kwargs_are_preserved() -> None:
    seen: dict[str, Any] = {}

    @tool(name="check_part_equivalency", approval_mode="never_require")
    async def probe(description: Annotated[str, "d"], specs: dict[str, str], ctx: FunctionInvocationContext) -> str:
        seen.update(ctx.kwargs)
        return "{}"

    await collect(_agent_with(probe), PART_REQUEST, function_invocation_kwargs={"tenant": "contoso"})
    assert seen["tenant"] == "contoso"
    assert callable(seen["progress_sink"])
    assert seen["tool_call_id"]
