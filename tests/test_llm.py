import sys
from types import SimpleNamespace
from typing import Any

import pytest

from sathealth.explain.llm import default_client, reword
from sathealth.explain.narrative import Sentence

FACTS = {"chan": "P-1", "onset": "1180", "rarity": "1 in 2,000,000"}
SENTENCES = [Sentence("Began at step 1180.", ("onset",))]


class _FakeClient:
    def __init__(self, text: str, stop_reason: str = "end_turn") -> None:
        self.calls: list[dict[str, Any]] = []
        self._response = SimpleNamespace(
            stop_reason=stop_reason,
            model="claude-opus-5-5",
            content=[SimpleNamespace(type="text", text=text)] if text else [],
        )
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._response


def test_grounded_rewording_is_kept() -> None:
    client = _FakeClient("On P-1 the deviation started at step 1180, rarer than 1 in 2000000.")
    brief = reword(SENTENCES, FACTS, client=client)
    assert brief is not None
    assert brief.grounded
    request = client.calls[0]
    assert request["model"] == "claude-opus-5-5"
    assert request["fallbacks"] == "default"
    assert "1180" in request["messages"][0]["content"]


def test_invented_numbers_are_flagged() -> None:
    brief = reword(SENTENCES, FACTS, client=_FakeClient("It began at step 1180 and lasted 42 s."))
    assert brief is not None
    assert not brief.grounded
    assert brief.invented == ("42",)


def test_refusal_or_empty_output_falls_back_to_the_template() -> None:
    assert reword(SENTENCES, FACTS, client=_FakeClient("x", stop_reason="refusal")) is None
    assert reword(SENTENCES, FACTS, client=_FakeClient("")) is None


def test_missing_sdk_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(RuntimeError, match="extra llm"):
        default_client()
