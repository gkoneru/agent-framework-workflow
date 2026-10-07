"""Chat-client factory and a helper that runs a sub-agent with structured (Pydantic) output.

Provider selection - ``PART_EQ_PROVIDER`` forces one; otherwise the first match wins:

    OPENAI_API_KEY             -> "openai"        OpenAIChatClient (Responses API), model = OPENAI_MODEL
    AZURE_OPENAI_ENDPOINT      -> "azure_openai"  OpenAIChatClient(azure_endpoint=...), model = AZURE_OPENAI_MODEL,
                                                  Entra ID via AzureCliCredential (run ``az login``)
    FOUNDRY_PROJECT_ENDPOINT   -> "foundry"       FoundryChatClient, model = FOUNDRY_MODEL, AzureCliCredential
    (nothing set)              -> "stub"          deterministic offline model (see stub.py)
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, TypeVar, cast

from agent_framework import Agent
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

PROVIDERS = ("openai", "azure_openai", "foundry", "stub")
INPUT_START, INPUT_END = "<<INPUT>>", "<<END>>"


def with_input(instruction: str, payload: Any) -> str:
    """Prompt = instruction + a machine-readable JSON block (the stub parses it; real models read it too)."""
    return f"{instruction}\n{INPUT_START}{json.dumps(payload, default=str)}{INPUT_END}"


def provider() -> str:
    forced = os.getenv("PART_EQ_PROVIDER", "").strip().lower()
    if forced:
        if forced not in PROVIDERS:
            raise ValueError(f"PART_EQ_PROVIDER must be one of {PROVIDERS}, got {forced!r}")
        return forced
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        return "azure_openai"
    if os.getenv("FOUNDRY_PROJECT_ENDPOINT"):
        return "foundry"
    return "stub"


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is required for provider {provider()!r}")
    return value


def make_chat_client() -> Any:
    match provider():
        case "openai":
            from agent_framework.openai import OpenAIChatClient

            return OpenAIChatClient(model=_require("OPENAI_MODEL"))
        case "azure_openai":
            from agent_framework.openai import OpenAIChatClient
            from azure.identity import AzureCliCredential

            return OpenAIChatClient(
                azure_endpoint=_require("AZURE_OPENAI_ENDPOINT"),
                model=_require("AZURE_OPENAI_MODEL"),
                credential=AzureCliCredential(),
            )
        case "foundry":
            from agent_framework.foundry import FoundryChatClient
            from azure.identity import AzureCliCredential

            return FoundryChatClient(
                project_endpoint=_require("FOUNDRY_PROJECT_ENDPOINT"),
                model=_require("FOUNDRY_MODEL"),
                credential=AzureCliCredential(),
            )
        case _:
            from .stub import StubChatClient

            return StubChatClient()


def web_search_tools() -> list[Any]:
    """Hosted web-search tool for the research agent when the provider supports it ([] for the stub)."""
    match provider():
        case "openai" | "azure_openai":
            from agent_framework.openai import OpenAIChatClient

            return [OpenAIChatClient.get_web_search_tool()]
        case "foundry":
            from agent_framework.foundry import FoundryChatClient

            return [FoundryChatClient.get_web_search_tool()]
        case _:
            return []


async def run_structured(
    agent: Agent,
    prompt: str,
    schema: type[T],
    *,
    background: bool = False,
    poll_seconds: float = 2.0,
) -> T:
    """Run a sub-agent and return a validated instance of ``schema``.

    ``background=True`` (OpenAI / Azure OpenAI only) uses Responses *background mode*: the call returns
    immediately with a ``continuation_token`` and we poll until the long-running response completes.
    Useful for slow research models. It is a transport detail of ONE step, not orchestration.
    """
    if background and provider() in ("openai", "azure_openai"):
        from agent_framework.openai import OpenAIChatOptions, OpenAIContinuationToken

        session = agent.create_session()
        response = await agent.run(
            prompt, session=session, options=OpenAIChatOptions(background=True, response_format=schema)
        )
        while response.continuation_token is not None:
            await asyncio.sleep(poll_seconds)
            response = await agent.run(
                session=session,
                options=OpenAIChatOptions(
                    continuation_token=cast(OpenAIContinuationToken, response.continuation_token),
                    response_format=schema,
                ),
            )
    else:
        response = await agent.run(prompt, options={"response_format": schema})

    value = response.value
    if isinstance(value, schema):
        return value
    return schema.model_validate_json(response.text)
