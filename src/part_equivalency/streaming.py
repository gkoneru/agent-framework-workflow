"""Stream progress emitted *inside a tool call* as part of the calling agent's own response.

Why this is needed (agent-framework-core 1.19 / 1.20):
  A tool call is opaque to the agent's response stream. ``workflow.as_agent().as_tool()`` runs the
  workflow to completion and returns only the final text; ``as_tool(stream_callback=...)`` sees the
  inner updates but only as a host-side callback - they never enter the parent agent's stream, so an
  AG-UI client sees nothing until the tool returns. Agent/chat middleware can transform updates but
  cannot inject new ones.

What this module provides (public framework APIs only, nothing patched):
  * ``ProgressStreamingAgent`` - wraps an ``Agent``. For each streaming run it creates a queue, hands
    tools a ``progress_sink`` through ``function_invocation_kwargs`` and yields the agent's own updates
    and the tools' progress in arrival order.
  * ``bind_tool_call_id`` - function middleware that gives the tool its provider tool-call id, so its
    progress can be attached to that call.
  * ``progress_reporter(ctx)`` - what a tool calls to report progress.

Progress is emitted as ``text_reasoning`` content with ``id = "progress_<tool_call_id>"``. The AG-UI
endpoint maps consecutive reasoning content with the same id to ONE reasoning message
(REASONING_START ... REASONING_MESSAGE_CONTENT* ... REASONING_END), which chat UIs render as a
collapsible "thinking"/details block between TOOL_CALL_START and TOOL_CALL_END.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, Awaitable, Callable
from typing import Any

from agent_framework import (
    Agent,
    AgentResponse,
    AgentResponseUpdate,
    AgentSession,
    BaseAgent,
    Content,
    FunctionInvocationContext,
    ResponseStream,
    function_middleware,
)
from pydantic import BaseModel

ProgressSink = Callable[[Any, "str | None"], Awaitable[None]]
Reporter = Callable[[Any], Awaitable[None]]

PROGRESS_SINK_KEY = "progress_sink"
TOOL_CALL_ID_KEY = "tool_call_id"


def progress_message_id(tool_call_id: str | None) -> str | None:
    """Reasoning id used for all progress of one tool call (AG-UI ``messageId``)."""
    return f"progress_{tool_call_id}" if tool_call_id else None


@function_middleware
async def bind_tool_call_id(ctx: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]) -> None:
    """Copy the provider tool-call id into the tool's kwargs.

    ``ctx.metadata["call_id"]`` is populated by the framework when a function-middleware pipeline
    exists; ``ctx.kwargs`` changes made here are visible to the tool.
    """
    ctx.kwargs[TOOL_CALL_ID_KEY] = ctx.metadata.get("call_id")
    await call_next()


def progress_reporter(ctx: FunctionInvocationContext) -> Reporter:
    """Return ``report(item)`` for use inside a tool. A no-op when the agent is not wrapped."""
    sink: ProgressSink | None = ctx.kwargs.get(PROGRESS_SINK_KEY)
    call_id: str | None = ctx.kwargs.get(TOOL_CALL_ID_KEY)

    async def report(item: Any) -> None:
        if sink is not None:
            await sink(item, call_id)

    return report


def _progress_update(item: Any, tool_call_id: str | None, author_name: str | None) -> AgentResponseUpdate:
    payload = item.model_dump(mode="json") if isinstance(item, BaseModel) else item
    return AgentResponseUpdate(
        role="assistant",
        author_name=author_name,
        contents=[
            Content.from_text_reasoning(
                id=progress_message_id(tool_call_id),
                text=f"{item}\n",
                additional_properties={TOOL_CALL_ID_KEY: tool_call_id, "progress": payload},
            )
        ],
    )


class ProgressStreamingAgent(BaseAgent):
    """Delegates to ``inner`` and merges tool progress into the streamed response.

    To the caller (and to AG-UI) this is still ONE agent: same name, same sessions, same final answer.
    Non-streaming runs are passed through unchanged (progress is simply not reported).
    """

    def __init__(self, inner: Agent, *, description: str | None = None, **kwargs: Any) -> None:
        super().__init__(
            id=inner.id,
            name=inner.name,
            description=description or inner.description,
            **kwargs,
        )
        self.inner = inner

    def create_session(self, *, session_id: str | None = None) -> AgentSession:
        return self.inner.create_session(session_id=session_id)

    def run(  # type: ignore[override]
        self,
        messages: Any = None,
        *,
        stream: bool = False,
        session: AgentSession | None = None,
        **kwargs: Any,
    ) -> Awaitable[AgentResponse] | ResponseStream[AgentResponseUpdate, AgentResponse]:
        if not stream:
            return self.inner.run(messages, session=session, **kwargs)
        return ResponseStream(
            self._stream(messages, session=session, **kwargs),
            finalizer=AgentResponse.from_updates,
        )

    async def _stream(
        self, messages: Any, *, session: AgentSession | None, **kwargs: Any
    ) -> AsyncIterable[AgentResponseUpdate]:
        queue: asyncio.Queue[Any] = asyncio.Queue()
        done = object()
        author = self.name

        async def sink(item: Any, tool_call_id: str | None) -> None:
            await queue.put(_progress_update(item, tool_call_id, author))

        # Keep whatever the caller (e.g. the AG-UI endpoint) already passes to tools.
        function_kwargs = {**(kwargs.pop("function_invocation_kwargs", None) or {}), PROGRESS_SINK_KEY: sink}

        async def pump() -> None:
            try:
                async for update in self.inner.run(
                    messages, stream=True, session=session, function_invocation_kwargs=function_kwargs, **kwargs
                ):
                    await queue.put(update)
            except Exception as exc:  # forwarded to the consumer below
                await queue.put(exc)
            finally:
                await queue.put(done)

        task = asyncio.create_task(pump())
        try:
            while (item := await queue.get()) is not done:
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            if not task.done():  # consumer stopped early (e.g. client disconnected)
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
