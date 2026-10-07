"""The outer conversational agent and its tools.

* The LLM decides WHETHER to run the workflow (``check_part_equivalency`` is an ordinary tool).
* The workflow decides HOW the steps run (see workflow.py) - the LLM never orchestrates them.
* While the tool runs, the workflow's intermediate ``Progress`` is reported through
  ``progress_reporter(ctx)`` and streamed by ``ProgressStreamingAgent`` (see streaming.py).
* The tool returns a compact typed ``FinalSummary`` (JSON) to the LLM, not the progress log.
* The user's answer to the summary's question is a normal next turn -> ``record_part_decision``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from agent_framework import Agent, FunctionInvocationContext, tool
from pydantic import Field

from .llm import make_chat_client
from .models import FinalSummary, PartRequest, Progress
from .streaming import ProgressStreamingAgent, bind_tool_call_id, progress_reporter
from .workflow import PartAgents, build_workflow

MASTER_INSTRUCTIONS = (
    "You are the MRO parts assistant. You can chat normally about maintenance topics. "
    "When the user asks for a part, a replacement or equivalents, call check_part_equivalency ONCE with a "
    "description and normalized specs (voltage, size, current, connector, ...). Do not orchestrate the "
    "steps yourself - the tool runs the full deterministic workflow. Present the result briefly and ask "
    "the tool's next_question. When the user then chooses, call record_part_decision with the request_id "
    "from the tool result."
)


class WorkflowFailedError(RuntimeError):
    """The part-equivalency workflow did not produce a FinalSummary."""


@tool(name="check_part_equivalency", approval_mode="never_require")  # read-only: search + research
async def check_part_equivalency(
    description: Annotated[str, Field(description="Free-text description of the requested part")],
    specs: Annotated[dict[str, str], Field(description="Normalized attributes, e.g. {'voltage': '12VDC'}")],
    ctx: FunctionInvocationContext,  # injected by the framework, hidden from the LLM's tool schema
) -> str:
    """Run the deterministic part-equivalency workflow (catalog search, evaluation, web research, summary)."""
    report = progress_reporter(ctx)
    agents: PartAgents | None = ctx.kwargs.get("part_agents")  # optional injection (tests, custom models)
    request = PartRequest(description=description, specs={k.lower(): v for k, v in specs.items()})

    summary: FinalSummary | None = None
    async for event in build_workflow(agents=agents).run(request, stream=True):  # fresh instance per call
        if event.type == "intermediate" and isinstance(event.data, Progress):
            await report(event.data)
        elif event.type == "output" and isinstance(event.data, FinalSummary):
            summary = event.data
        elif event.type == "failed":
            raise WorkflowFailedError(f"Part-equivalency workflow failed: {event.details}")
    if summary is None:
        raise WorkflowFailedError("Part-equivalency workflow produced no summary")
    return summary.model_dump_json()


# This writes to the system of record. In production use approval_mode="always_require" so the user
# approves the write (AG-UI surfaces the approval request to the UI).
@tool(name="record_part_decision", approval_mode="never_require")
async def record_part_decision(
    request_id: Annotated[str, Field(description="request_id returned by check_part_equivalency")],
    decision: Annotated[Literal["use_existing", "create_new"], Field(description="The user's choice")],
    part_number: Annotated[str | None, Field(description="Existing part number when decision is use_existing")] = None,
) -> str:
    """Record the user's decision: link an existing part or create a new part request (stubbed)."""
    if decision == "use_existing":
        if not part_number:
            raise ValueError("part_number is required when decision is 'use_existing'")
        return f"Linked {request_id} to existing part {part_number}."
    return f"Created new part request for {request_id}."


def create_master_agent(*, client: Any | None = None) -> Agent:
    """The plain outer agent (no progress streaming). ``bind_tool_call_id`` is required for grouping."""
    return Agent(
        name="parts_assistant",
        description="MRO parts assistant",
        instructions=MASTER_INSTRUCTIONS,
        client=client or make_chat_client(),
        tools=[check_part_equivalency, record_part_decision],
        middleware=[bind_tool_call_id],
    )


def create_agent(*, client: Any | None = None) -> ProgressStreamingAgent:
    """The agent to expose to users: outer agent + workflow progress streamed inline."""
    return ProgressStreamingAgent(create_master_agent(client=client))
