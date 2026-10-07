"""LLM-facing response_format models must be compatible with OpenAI strict structured outputs."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from part_equivalency.models import (
    EvaluationResult,
    FinalSummaryDraft,
    IntakeSpec,
    PartRequest,
    Progress,
    ResearchResult,
)

LLM_FACING = [IntakeSpec, EvaluationResult, ResearchResult, FinalSummaryDraft]


def _objects(schema: dict[str, Any]) -> list[dict[str, Any]]:
    found = [schema, *schema.get("$defs", {}).values()]
    return [s for s in found if s.get("type") == "object"]


@pytest.mark.parametrize("model", LLM_FACING, ids=lambda m: m.__name__)
def test_llm_facing_models_are_strict_compatible(model: type[BaseModel]) -> None:
    for obj in _objects(model.model_json_schema()):
        props = obj.get("properties", {})
        assert set(obj.get("required", [])) == set(props), f"{obj.get('title')}: every field must be required"
        for name, prop in props.items():
            assert "additionalProperties" not in prop, f"{obj.get('title')}.{name}: free-form dicts are not strict"


def test_request_ids_are_unique_and_prefixed() -> None:
    ids = {PartRequest(description="x").request_id for _ in range(50)}
    assert len(ids) == 50
    assert all(i.startswith("REQ-") and len(i) == 12 for i in ids)


def test_progress_renders_for_humans() -> None:
    p = Progress(stage="evaluate", status="completed", message="1 equivalent")
    assert str(p) == "[evaluate:completed] 1 equivalent"
