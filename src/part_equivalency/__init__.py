"""Part equivalency: one conversational agent that calls a deterministic workflow and streams its progress."""

from .agent import check_part_equivalency, create_agent, create_master_agent, record_part_decision
from .models import FinalSummary, PartRequest, Progress
from .streaming import ProgressStreamingAgent, bind_tool_call_id, progress_message_id, progress_reporter
from .workflow import PartAgents, build_workflow

__version__ = "0.1.0"

__all__ = [
    "FinalSummary",
    "PartAgents",
    "PartRequest",
    "Progress",
    "ProgressStreamingAgent",
    "bind_tool_call_id",
    "build_workflow",
    "check_part_equivalency",
    "create_agent",
    "create_master_agent",
    "progress_message_id",
    "progress_reporter",
    "record_part_decision",
]
