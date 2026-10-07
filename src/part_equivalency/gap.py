"""Side-by-side comparison: ``workflow.as_agent().as_tool()`` vs. ``ProgressStreamingAgent``.

python -m part_equivalency gap
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from agent_framework import Agent

from .agent import MASTER_INSTRUCTIONS, create_agent
from .llm import make_chat_client
from .workflow import build_workflow

PART_REQUEST = "I need a replacement cooling fan: 12V DC, 120mm, 0.25A, 3-pin connector"


@dataclass
class StreamTrace:
    """What arrived on an agent's response stream, in order, with timestamps (seconds)."""

    label: str
    events: list[tuple[float, str, str]] = field(default_factory=list)  # (t, content type, detail)
    callback_updates: int | None = None

    def count(self, content_type: str) -> int:
        return sum(1 for _, kind, _ in self.events if kind == content_type)

    def first(self, content_type: str) -> float | None:
        return next((t for t, kind, _ in self.events if kind == content_type), None)


async def trace(label: str, agent: Any, prompt: str = PART_REQUEST) -> StreamTrace:
    result = StreamTrace(label)
    seen_calls: set[str] = set()
    t0 = time.perf_counter()
    async for update in agent.run(prompt, stream=True):
        for c in update.contents:
            now = time.perf_counter() - t0
            if c.type == "function_call":
                if c.call_id and c.call_id not in seen_calls:  # real models stream a call in several chunks
                    seen_calls.add(c.call_id)
                    result.events.append((now, c.type, c.name or ""))
            elif c.type == "text_reasoning":
                result.events.append((now, c.type, (c.text or "").strip()))
            elif c.type == "function_result":
                text = c.result if isinstance(c.result, str) else str(c.result)
                result.events.append((now, c.type, f"{len(text)} chars"))
            elif c.type == "text" and c.text:
                result.events.append((now, c.type, c.text))
    return result


async def compare(prompt: str = PART_REQUEST) -> tuple[StreamTrace, StreamTrace]:
    """Run the same request through both approaches and return their stream traces."""
    received: list[Any] = []
    as_tool = (
        build_workflow()
        .as_agent(name="part_equivalency")
        .as_tool(name="check_part_equivalency", stream_callback=received.append)
    )
    plain = Agent(name="parts_assistant", instructions=MASTER_INSTRUCTIONS, client=make_chat_client(), tools=[as_tool])
    gap = await trace("GAP: workflow.as_agent().as_tool(stream_callback=...)", plain, prompt)
    gap.callback_updates = len(received)
    fix = await trace("FIX: ProgressStreamingAgent + @tool + progress_reporter", create_agent(), prompt)
    return gap, fix


def render(t: StreamTrace) -> str:
    lines = [f"\n=== {t.label}"]
    text_shown = False
    for at, kind, detail in t.events:
        if kind == "text":
            if not text_shown:
                lines.append(f"  {at:5.2f}s answer starts streaming")
                text_shown = True
            continue
        label = {"function_call": "tool call", "text_reasoning": "progress", "function_result": "tool result"}[kind]
        lines.append(f"  {at:5.2f}s {label:12} {detail[:90]}")
    if t.callback_updates is not None:
        lines.append(
            f"  -> stream_callback received {t.callback_updates} updates; "
            f"{t.count('text_reasoning')} reached the agent's response stream"
        )
    return "\n".join(lines)
