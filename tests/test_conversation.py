"""One session, three turns: chat -> workflow (as a tool) -> follow-up decision."""

from __future__ import annotations

import json

import pytest

from part_equivalency import create_agent, record_part_decision

from .conftest import PART_REQUEST, collect


async def test_three_turn_conversation_on_one_session() -> None:
    agent = create_agent()
    session = agent.create_session()

    turn1 = await collect(agent, "Hi! What do you help with?", session=session)
    assert not any(c.type == "function_call" for c in turn1)

    turn2 = await collect(agent, PART_REQUEST, session=session)
    calls = [c for c in turn2 if c.type == "function_call" and c.name]
    assert [c.name for c in calls] == ["check_part_equivalency"]
    raw = next(c for c in turn2 if c.type == "function_result").result
    summary = json.loads(raw if isinstance(raw, str) else raw[0].text)
    assert any(c.type == "text_reasoning" for c in turn2)

    turn3 = await collect(agent, "Use the existing part please", session=session)
    decision = next(c for c in turn3 if c.type == "function_call" and c.name)
    assert decision.name == "record_part_decision"
    args = decision.parse_arguments()
    assert args["request_id"] == summary["request_id"]  # taken from the conversation history
    assert args["decision"] == "use_existing"
    assert args["part_number"] == "FAN-12V-120"
    assert not any(c.type == "text_reasoning" for c in turn3)  # no workflow -> no progress block
    text = "".join(c.text for c in turn3 if c.type == "text")
    assert f"Linked {summary['request_id']} to existing part FAN-12V-120" in text


async def test_create_new_decision() -> None:
    agent = create_agent()
    session = agent.create_session()
    await collect(agent, PART_REQUEST, session=session)
    turn = await collect(agent, "No, create a new one", session=session)
    call = next(c for c in turn if c.type == "function_call" and c.name)
    assert call.parse_arguments()["decision"] == "create_new"


async def test_record_decision_requires_part_for_use_existing() -> None:
    with pytest.raises(ValueError, match="part_number is required"):
        await record_part_decision.func(request_id="REQ-1", decision="use_existing", part_number=None)
    assert "FAN-1" in await record_part_decision.func(request_id="REQ-1", decision="use_existing", part_number="FAN-1")
