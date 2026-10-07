"""AG-UI endpoint: what a browser client actually receives over Server-Sent Events."""

from __future__ import annotations

import json
import uuid
from typing import Any

import httpx
import pytest

from part_equivalency.server import create_app

from .conftest import PART_REQUEST


async def sse_events(client: httpx.AsyncClient, messages: list[dict[str, Any]], thread_id: str) -> list[dict[str, Any]]:
    body = {
        "threadId": thread_id,
        "runId": f"run-{uuid.uuid4().hex[:8]}",
        "messages": messages,
        "state": {},
        "tools": [],
        "context": [],
        "forwardedProps": {},
    }
    events = []
    async with client.stream("POST", "/chat", json=body, headers={"Accept": "text/event-stream"}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        async for line in response.aiter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[5:]))
    return events


@pytest.fixture
async def client() -> Any:
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30) as c:
        yield c


async def test_healthz(client: httpx.AsyncClient) -> None:
    assert (await client.get("/healthz")).json() == {"status": "ok"}


async def test_workflow_progress_is_one_reasoning_block_inside_the_tool_call(client: httpx.AsyncClient) -> None:
    events = await sse_events(client, [{"id": "u1", "role": "user", "content": PART_REQUEST}], "t-1")
    types = [e["type"] for e in events]
    assert types[0] == "RUN_STARTED" and types[-1] == "RUN_FINISHED"

    start = next(i for i, e in enumerate(events) if e["type"] == "TOOL_CALL_START")
    assert events[start]["toolCallName"] == "check_part_equivalency"
    call_id = events[start]["toolCallId"]
    end = types.index("TOOL_CALL_END")
    result = types.index("TOOL_CALL_RESULT")

    content = [(i, e) for i, e in enumerate(events) if e["type"] == "REASONING_MESSAGE_CONTENT"]
    assert len(content) >= 8
    assert {e["messageId"] for _, e in content} == {f"progress_{call_id}"}  # one collapsible block per call
    assert all(start < i < end for i, _ in content)
    assert "cortex_search:started" in content[0][1]["delta"]
    for marker in ("REASONING_START", "REASONING_MESSAGE_START", "REASONING_MESSAGE_END", "REASONING_END"):
        assert types.count(marker) == 1, marker
        assert start < types.index(marker) < result

    answer = "".join(e["delta"] for e in events if e["type"] == "TEXT_MESSAGE_CONTENT")
    assert "FAN-12V-120" in answer and "create a new part request" in answer


async def test_plain_chat_has_no_reasoning_block(client: httpx.AsyncClient) -> None:
    events = await sse_events(client, [{"id": "u1", "role": "user", "content": "Hi!"}], "t-2")
    types = {e["type"] for e in events}
    assert "TEXT_MESSAGE_CONTENT" in types
    assert not any(t.startswith("REASONING") or t.startswith("TOOL_CALL") for t in types)
