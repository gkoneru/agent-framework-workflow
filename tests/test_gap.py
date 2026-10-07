"""Documents the behaviour the customer hit with ``workflow.as_agent().as_tool()``.

If a future agent-framework release streams sub-agent updates into the parent stream, the first test
will fail - that is intentional: revisit whether the wrapper is still needed.
"""

from __future__ import annotations

from part_equivalency.gap import compare, render


async def test_as_tool_does_not_stream_progress_but_the_wrapper_does() -> None:
    gap, fix = await compare()

    # as_tool: progress only reaches the host-side callback, never the agent's response stream.
    assert gap.callback_updates and gap.callback_updates > 0
    assert gap.count("text_reasoning") == 0
    assert gap.count("function_result") == 1

    # Wrapper: progress arrives on the stream while the tool is still running.
    assert fix.count("text_reasoning") >= 8
    first_progress, result = fix.first("text_reasoning"), fix.first("function_result")
    assert first_progress is not None and result is not None and first_progress < result

    assert "0 reached the agent's response stream" in render(gap)
