"""Stage 1 is plain code: a catalog search. This is a STUB with a fixed catalog.

Replace ``search_cortex`` with your Snowflake Cortex Search call (keep the signature):
it receives the normalized ``PartRequest`` and returns ranked ``Candidate`` objects.
"""

from __future__ import annotations

from .models import Candidate, PartRequest
from .stub import simulated_latency

_CATALOG: list[Candidate] = [
    Candidate(
        part_number="FAN-12V-120",
        description="Axial fan 120x120x25mm, 12VDC, 3-pin",
        specs={"voltage": "12VDC", "size": "120mm", "connector": "3-pin"},
        score=0.94,
    ),
    Candidate(
        part_number="FAN-12V-120-B",
        description="Axial fan 120mm 12VDC 0.25A 3-pin ball bearing",
        specs={"voltage": "12VDC", "size": "120mm", "connector": "3-pin", "current": "0.25A"},
        score=0.91,
    ),
    Candidate(
        part_number="FAN-24V-120",
        description="Axial fan 120mm 24VDC 3-pin",
        specs={"voltage": "24VDC", "size": "120mm", "connector": "3-pin", "current": "0.12A"},
        score=0.83,
    ),
    Candidate(
        part_number="FAN-12V-092",
        description="Axial fan 92mm 12VDC 0.25A 3-pin",
        specs={"voltage": "12VDC", "size": "92mm", "connector": "3-pin", "current": "0.25A"},
        score=0.71,
    ),
    Candidate(
        part_number="FAN-12V-120-HS",
        description="High-speed axial fan 120mm 12VDC",
        specs={"voltage": "12VDC", "size": "120mm"},
        score=0.69,
    ),
]


async def search_cortex(request: PartRequest, top_k: int = 5) -> list[Candidate]:
    """Return the top-k catalog candidates for ``request`` (stub: fixed catalog, simulated latency)."""
    await simulated_latency(0.6)
    return [c.model_copy(deep=True) for c in _CATALOG[:top_k]]
