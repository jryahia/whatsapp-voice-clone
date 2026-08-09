"""Tests for the PromptBuilder module — system prompt, user prompt, and summary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from profiler.analyzer import VoiceProfile
from responder.prompt_builder import PromptBuilder


# ---------------------------------------------------------------------------
# Factory helpers — avoids dependency on real ChromaDB
# ---------------------------------------------------------------------------


def make_voice_profile(
    name: str = "Mario Rossi",
    formality_label: str = "informale",
    formality_score: float = 0.4,
    lei_usage_pct: float = 10.0,
    tu_usage_pct: float = 90.0,
    top_words: list[str] | None = None,
    jargon: list[str] | None = None,
    preferred_emojis: list[str] | None = None,
    emoji_per_100: float = 2.5,
    most_common_greeting: str = "Ciao",
    greeting_patterns: dict[str, int] | None = None,
    avg_length_chars: float = 150.0,
    avg_length_words: float = 25.0,
    question_pct: float = 10.0,
    exclamation_pct: float = 5.0,
    caps_pct: float = 2.0,
    punctuation_habits: list[str] | None = None,
    busiest_hour: int = 19,
    avg_response_time_s: float = 120.0,
    raw_sample: str = "",
) -> VoiceProfile:
    """Create a test VoiceProfile with sensible defaults.

    Use keyword arguments to override specific fields.  No ChromaDB or
    real analysis is involved — pure dataclass construction.
    """
    if top_words is None:
        top_words = ["grazie", "pizza", "cliente", "oggi", "domani"]
    if jargon is None:
        jargon = ["pizza", "forno", "impasto"]
    if preferred_emojis is None:
        preferred_emojis = ["😊", "🍕"]
    if greeting_patterns is None:
        greeting_patterns = {"Ciao": 42, "Buongiorno": 10}
    if punctuation_habits is None:
        punctuation_habits = ["uses ellipsis...", "lots of exclamation marks!"]

    return VoiceProfile(
        name=name,
        formality={
            "score": formality_score,
            "label": formality_label,
            "lei_usage_pct": lei_usage_pct,
            "tu_usage_pct": tu_usage_pct,
        },
        vocabulary={
            "top_words": top_words,
            "jargon": jargon,
        },
        emoji={
            "preferred": preferred_emojis,
            "per_100_words": emoji_per_100,
            "placement": {
                "start_pct": 10.0,
                "end_pct": 60.0,
                "mid_pct": 30.0,
            },
        },
        greetings={
            "patterns": greeting_patterns,
            "total_greetings": sum(greeting_patterns.values()),
            "most_common": most_common_greeting,
        },
        response_style={
            "avg_length_chars": avg_length_chars,
            "avg_length_words": avg_length_words,
            "question_pct": question_pct,
            "exclamation_pct": exclamation_pct,
            "caps_pct": caps_pct,
            "punctuation_habits": punctuation_habits,
        },
        time_patterns={
            "active_hours": list(range(9, 22)),
            "avg_response_time_s": avg_response_time_s,
            "busiest_hour": busiest_hour,
        },
        raw_sample=raw_sample,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pizza_profile() -> VoiceProfile:
    """A VoiceProfile for a pizzeria owner named Mario Rossi."""
    return make_voice_profile(
        name="Mario Rossi",
        jargon=["pizza", "forno", "impasto", "mozzarella"],
        top_words=["grazie", "pizza", "cliente", "forno", "impasto"],
        preferred_emojis=["🍕", "😊", "🔥"],
        most_common_greeting="Ciao",
    )


@pytest.fixture
def builder(pizza_profile: VoiceProfile) -> PromptBuilder:
    """A PromptBuilder backed by the pizza_profile fixture."""
    return PromptBuilder(profile=pizza_profile)


# ---------------------------------------------------------------------------
# System prompt tests
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    """The system prompt must contain the three expected major sections."""

    def test_contains_voice_profile_section(self, builder: PromptBuilder) -> None:
        prompt: str = builder.build_system_prompt()
        assert "VOICE PROFILE" in prompt or "VOCE DEL TITOLARE" in prompt

    def test_contains_hard_rules_section(self, builder: PromptBuilder) -> None:
        prompt: str = builder.build_system_prompt()
        assert "HARD RULES" in prompt or "REGOLE ASSOLUTE" in prompt

    def test_contains_context_section(self, builder: PromptBuilder) -> None:
        prompt: str = builder.build_system_prompt()
        assert "CONTEXT" in prompt or "CONTESTO" in prompt

    def test_contains_profile_name(self, builder: PromptBuilder) -> None:
        prompt: str = builder.build_system_prompt()
        assert "Mario Rossi" in prompt

    def test_top_words_as_dicts_does_not_crash(self) -> None:
        """compute_vocabulary yields top_words as \{\{'word','count'\} dicts; the
        system prompt builder must handle that real shape without raising."""
        profile = make_voice_profile(
            top_words=[{"word": "grazie", "count": 8}, {"word": "pizza", "count": 5}],
            jargon=["pizza"],
        )
        b = PromptBuilder(profile=profile)
        prompt: str = b.build_system_prompt()
        assert "grazie" in prompt
        assert "pizza" in prompt

    def test_contains_greeting_instruction(self, builder: PromptBuilder) -> None:
        prompt: str = builder.build_system_prompt()
        # The greeting should appear both as a mention and in the hard rules
        assert "Ciao" in prompt

    def test_contains_formality_analysis(self, builder: PromptBuilder) -> None:
        prompt: str = builder.build_system_prompt()
        assert "Formalità" in prompt or "formale" in prompt.lower()

    def test_contains_emoji_section(self, builder: PromptBuilder) -> None:
        prompt: str = builder.build_system_prompt()
        assert "Emoji" in prompt or "emoji" in prompt.lower()

    def test_contains_jargon(self, builder: PromptBuilder) -> None:
        prompt: str = builder.build_system_prompt()
        assert "pizza" in prompt.lower()
        assert "forno" in prompt.lower()

    def test_contains_today_date(self, builder: PromptBuilder) -> None:
        from datetime import date

        prompt: str = builder.build_system_prompt()
        assert date.today().isoformat() in prompt

    def test_system_prompt_is_reasonably_long(
        self, builder: PromptBuilder
    ) -> None:
        prompt: str = builder.build_system_prompt()
        assert len(prompt) > 500

    def test_formal_profile_different_label(
        self, pizza_profile: VoiceProfile
    ) -> None:
        """A formal profile should produce different formality text."""
        formal_profile = make_voice_profile(
            name="Avv. Bianchi",
            formality_label="formale",
            formality_score=0.8,
            lei_usage_pct=85.0,
            tu_usage_pct=15.0,
            jargon=["consulenza", "contratto"],
        )
        formal_builder = PromptBuilder(profile=formal_profile)
        prompt: str = formal_builder.build_system_prompt()
        assert "Formale" in prompt or "formale" in prompt.lower()


# ---------------------------------------------------------------------------
# User prompt tests
# ---------------------------------------------------------------------------


class TestBuildUserPrompt:
    """The user prompt must wrap the incoming message correctly."""

    def test_with_customer_name(self, builder: PromptBuilder) -> None:
        prompt: str = builder.build_user_prompt(
            message="Vorrei una pizza margherita",
            customer_name="Luca",
        )
        expected = "Il cliente Luca ha scritto: Vorrei una pizza margherita"
        assert prompt == expected

    def test_without_customer_name(self, builder: PromptBuilder) -> None:
        prompt: str = builder.build_user_prompt(
            message="Buongiorno, avete il menu?",
        )
        expected = "Un cliente ha scritto: Buongiorno, avete il menu?"
        assert prompt == expected

    def test_with_none_name(self, builder: PromptBuilder) -> None:
        prompt: str = builder.build_user_prompt(
            message="Ciao!",
            customer_name=None,
        )
        expected = "Un cliente ha scritto: Ciao!"
        assert prompt == expected

    def test_empty_message_with_name(self, builder: PromptBuilder) -> None:
        prompt: str = builder.build_user_prompt(
            message="",
            customer_name="Maria",
        )
        expected = "Il cliente Maria ha scritto: "
        assert prompt == expected

    def test_empty_message_no_name(self, builder: PromptBuilder) -> None:
        prompt: str = builder.build_user_prompt(message="")
        expected = "Un cliente ha scritto: "
        assert prompt == expected


# ---------------------------------------------------------------------------
# Profile summary tests
# ---------------------------------------------------------------------------


class TestGetProfileSummary:
    """The profile summary must be a human-readable multi-line string."""

    def test_returns_multi_line_string(self, builder: PromptBuilder) -> None:
        summary: str = builder.get_profile_summary()
        lines = summary.split("\n")
        assert len(lines) >= 3

    def test_contains_profile_name(self, builder: PromptBuilder) -> None:
        summary: str = builder.get_profile_summary()
        assert "Mario Rossi" in summary

    def test_contains_formality_label(self, builder: PromptBuilder) -> None:
        summary: str = builder.get_profile_summary()
        assert "Informale" in summary or "informale" in summary.lower()

    def test_contains_greeting(self, builder: PromptBuilder) -> None:
        summary: str = builder.get_profile_summary()
        assert "Ciao" in summary

    def test_contains_emoji_info(self, builder: PromptBuilder) -> None:
        summary: str = builder.get_profile_summary()
        # Should mention at least one emoji or "nessuna emoji"
        assert any(e in summary for e in ["🍕", "😊", "🔥", "emoji", "nessuna"])

    def test_contains_jargon_info(self, builder: PromptBuilder) -> None:
        summary: str = builder.get_profile_summary()
        assert "Gergo" in summary

    def test_contains_busiest_hour(self, builder: PromptBuilder) -> None:
        summary: str = builder.get_profile_summary()
        assert "19:00" in summary or "19" in summary

    def test_summary_format(self, builder: PromptBuilder) -> None:
        summary: str = builder.get_profile_summary()
        assert summary.startswith("Profilo:")

    def test_no_jargon_profile(self) -> None:
        """A profile without jargon should mention 'nessun gergo'."""
        profile = make_voice_profile(jargon=[])
        b = PromptBuilder(profile=profile)
        summary: str = b.get_profile_summary()
        assert "nessun gergo" in summary.lower()


# ---------------------------------------------------------------------------
# Business type inference tests
# ---------------------------------------------------------------------------


class TestInferBusinessType:
    """The _infer_business_type heuristic should classify correctly."""

    @pytest.mark.parametrize(
        ("jargon", "expected_type"),
        [
            (["pizza", "pasta", "menu"], "Ristorazione / Food & Beverage"),
            (["parrucchiere", "taglio", "colore"], "Estetica / Benessere / Barbiere"),
            (["idraulico", "riparazione"], "Artigiano / Mestiere (idraulico, elettricista, ecc.)"),
            (["dottore", "visita"], "Sanità / Servizi medici"),
            (["consulenza", "consulente"], "Servizi / Consulenza"),
            (["programmazione", "software"], "Attività commerciale / Servizi"),
        ],
    )
    def test_business_type_classification(
        self, jargon: list[str], expected_type: str
    ) -> None:
        profile = make_voice_profile(jargon=jargon)
        b = PromptBuilder(profile=profile)
        prompt: str = b.build_system_prompt()
        assert expected_type in prompt
