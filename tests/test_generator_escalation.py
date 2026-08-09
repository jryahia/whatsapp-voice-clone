"""Tests for the live escalation path in responder.generator.generate_reply."""

from __future__ import annotations

from typing import Any

import pytest

from responder import generator
from config import settings
from profiler.analyzer import VoiceProfile
from server.escalation import Escalator


def _make_stub_profile(name: str) -> object:
    """Return a minimal but valid VoiceProfile (all attributes present)."""
    return VoiceProfile(
        name=name,
        formality={"score": 0.5, "label": "informale", "lei_usage_pct": 0.0, "tu_usage_pct": 0.0},
        vocabulary={"top_words": [], "jargon": []},
        emoji={"preferred": [], "per_100_words": 0.0, "placement": {"start_pct": 0, "end_pct": 0, "mid_pct": 0}},
        greetings={"patterns": {}, "total_greetings": 0, "most_common": "Ciao"},
        response_style={
            "avg_length_chars": 0.0, "avg_length_words": 0.0,
            "question_pct": 0.0, "exclamation_pct": 0.0,
            "caps_pct": 0.0, "punctuation_habits": [],
        },
        time_patterns={"active_hours": [], "avg_response_time_s": 0.0, "busiest_hour": 0},
        raw_sample="",
    )


class _StubProfileStore:
    """Stand-in for ProfileStore that never touches ChromaDB."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def load(self, profile_name: str) -> object:
        return _make_stub_profile(profile_name)


@pytest.fixture
def escalate_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Patch ProfileStore + Escalator.escalate, recording every escalate call."""
    monkeypatch.setattr(generator, "ProfileStore", _StubProfileStore)

    calls: list[dict[str, Any]] = []

    def fake_escalate(self: Escalator, **kwargs: Any) -> bool:
        calls.append(kwargs)
        return True

    monkeypatch.setattr(Escalator, "escalate", fake_escalate)
    return calls


def test_price_query_invokes_escalator_once(
    escalate_calls: list[dict[str, Any]],
) -> None:
    """A price query must escalate to the owner exactly once."""
    result = generator.generate_reply(
        message="Ciao, quanto costa la pizza?",
        profile_name="test_profile",
        customer_name="Marco",
        customer_phone="whatsapp:+393451234567",
    )

    assert len(escalate_calls) == 1

    call = escalate_calls[0]
    assert call["message"] == "Ciao, quanto costa la pizza?"
    assert call["reason"] == "Price query — never hallucinate prices"
    assert call["confidence"] == 1.0
    assert call["customer_phone"] == "whatsapp:+393451234567"

    assert result["escalated"] is True
    assert result["confidence"] == 1.0
    assert result["model"] == "none"
    assert "titolare" in result["reply"]


def test_missing_phone_falls_back_to_unknown(
    escalate_calls: list[dict[str, Any]],
) -> None:
    """customer_phone defaults to None and is reported as 'unknown'."""
    generator.generate_reply(
        message="Quanto costa la pizza?",
        profile_name="test_profile",
    )

    assert len(escalate_calls) == 1
    assert escalate_calls[0]["customer_phone"] == "unknown"


def test_safe_message_does_not_escalate(
    escalate_calls: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A greeting-only message never reaches the Escalator."""
    monkeypatch.setattr(
        generator, "_call_openai", lambda *a, **kw: "Ciao! Come posso aiutarti?"
    )

    result = generator.generate_reply(
        message="Ciao",
        profile_name="test_profile",
        customer_phone="whatsapp:+393451234567",
    )

    assert escalate_calls == []
    assert result["escalated"] is False


def test_safe_message_no_api_key_returns_fallback(
    escalate_calls: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A safe message with an empty OPENAI_API_KEY must return the clean fallback,
    NOT raise an uncaught OpenAIError (regression: _call_openai must catch the
    base OpenAIError for missing credentials)."""
    monkeypatch.setattr(settings, "openai_api_key", "")

    result = generator.generate_reply(
        message="Ciao, siete aperti oggi?",
        profile_name="test_profile",
        customer_phone="whatsapp:+393451234567",
    )

    assert escalate_calls == []          # not escalated
    assert result["escalated"] is False
    assert result["reply"], "reply must not be empty"
    assert len(result["reply"]) > 10     # a real fallback sentence, not an error string

