from __future__ import annotations

import pytest

from part_equivalency.__main__ import main


def test_demo_runs_end_to_end(capsys: pytest.CaptureFixture[str]) -> None:
    main(["demo"])
    out = capsys.readouterr().out
    assert "[model provider: stub]" in out
    assert "▸ [cortex_search:started]" in out
    assert "🔧 record_part_decision" in out
    assert "to existing part FAN-12V-120" in out


def test_gap_runs(capsys: pytest.CaptureFixture[str]) -> None:
    main(["gap"])
    out = capsys.readouterr().out
    assert "=== GAP" in out and "=== FIX" in out


def test_invalid_provider_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PART_EQ_PROVIDER", "nope")
    with pytest.raises(ValueError, match="PART_EQ_PROVIDER"):
        main(["demo"])
