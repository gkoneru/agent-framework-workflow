"""AG-UI (Server-Sent Events) endpoint for the parts assistant.

    uvicorn part_equivalency.server:app --port 8000      # or: python -m part_equivalency serve
    POST /chat   AG-UI RunAgentInput -> text/event-stream

Event mapping you will see for a part request:
    TOOL_CALL_START check_part_equivalency
      REASONING_START / REASONING_MESSAGE_START   messageId = "progress_<toolCallId>"
      REASONING_MESSAGE_CONTENT  (one per workflow progress update, streamed live)
      REASONING_MESSAGE_END / REASONING_END
    TOOL_CALL_END / TOOL_CALL_RESULT
    TEXT_MESSAGE_START ... TEXT_MESSAGE_CONTENT* ... TEXT_MESSAGE_END   (the agent's answer)
"""

from __future__ import annotations

from typing import Any

from agent_framework_ag_ui import add_agent_framework_fastapi_endpoint
from fastapi import FastAPI

from .agent import create_agent


def create_app(*, agent: Any | None = None, path: str = "/chat") -> FastAPI:
    app = FastAPI(title="Part equivalency assistant")
    add_agent_framework_fastapi_endpoint(app, agent or create_agent(), path)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


def __getattr__(name: str) -> Any:  # lazy ``app`` so importing this module has no side effects
    if name == "app":
        return create_app()
    raise AttributeError(name)
