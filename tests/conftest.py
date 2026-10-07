from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

PART_REQUEST = "I need a replacement cooling fan: 12V DC, 120mm, 0.25A, 3-pin connector"


@pytest.fixture(autouse=True)
def _offline_stub(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every non-live test runs on the deterministic stub model with shortened latency."""
    if request.node.get_closest_marker("live"):
        return
    monkeypatch.setenv("PART_EQ_PROVIDER", "stub")
    monkeypatch.setenv("STUB_LATENCY_SCALE", "0.05")
    monkeypatch.delenv("RESEARCH_BACKGROUND", raising=False)


async def collect(agent: Any, prompt: str | Sequence[Any], **kwargs: Any) -> list[Any]:
    """All Content items of a streaming run, in arrival order."""
    return [c async for update in agent.run(prompt, stream=True, **kwargs) for c in update.contents]
