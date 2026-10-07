"""Optional: the same flow against a real model. Skipped unless you run ``pytest -m live``.

Requires one provider configured (see .env.example); the stub is not allowed here.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from part_equivalency import create_agent
from part_equivalency.llm import provider

from .conftest import PART_REQUEST

pytestmark = pytest.mark.live


@pytest.fixture(autouse=True)
def _require_real_provider() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    if provider() == "stub":
        pytest.skip("no model provider configured (set OPENAI_*, AZURE_OPENAI_* or FOUNDRY_*)")


async def turn(agent: Any, prompt: str, session: Any) -> tuple[list[Any], list[Any], list[Any]]:
    """Streamed contents, plus function calls/results from the final response (real models stream calls in chunks)."""
    stream = agent.run(prompt, stream=True, session=session)
    streamed = [c async for update in stream for c in update.contents]
    final = await stream.get_final_response()
    contents = [c for m in final.messages for c in m.contents]
    return (
        streamed,
        [c for c in contents if c.type == "function_call"],
        [c for c in contents if c.type == "function_result"],
    )


async def test_live_conversation_streams_workflow_progress() -> None:
    agent = create_agent()
    session = agent.create_session()

    streamed, calls, results = await turn(agent, PART_REQUEST, session)
    assert [c.name for c in calls] == ["check_part_equivalency"], "the model should call the workflow exactly once"
    progress = [c for c in streamed if c.type == "text_reasoning"]
    assert len(progress) >= 6
    assert {c.id for c in progress} == {f"progress_{calls[0].call_id}"}
    raw = results[0].result
    summary = json.loads(raw if isinstance(raw, str) else raw[0].text)
    assert summary["request_id"].startswith("REQ-")

    _, followup_calls, _ = await turn(agent, "Use the recommended existing part.", session)
    decision = [c for c in followup_calls if c.name == "record_part_decision"]
    assert len(decision) == 1
    args = decision[0].parse_arguments()
    assert args["request_id"] == summary["request_id"]
    if summary["recommended_part"]:
        assert args["decision"] == "use_existing"
        assert args["part_number"] == summary["recommended_part"]
