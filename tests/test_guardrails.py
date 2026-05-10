"""Tests for the Guardrails module — safety check rules on incoming messages."""

from __future__ import annotations

import pytest

from responder.guardrails import Guardrails, GuardrailResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def guardrails() -> Guardrails:
    """Provide a fresh Guardrails instance for each test."""
    return Guardrails()


# ---------------------------------------------------------------------------
# Rule 1: Sensitive topics — complaint / refund / cancellation
# ---------------------------------------------------------------------------


class TestSensitiveTopicComplaint:
    """Messages containing complaint-related keywords must escalate."""

    @pytest.mark.parametrize(
        "message",
        [
            "Ho un reclamo per il servizio ricevuto",
            "Voglio fare una lamentela sull'ordine",
            "Chiedo un rimborso per il prodotto difettoso",
            "Vorrei parlare di una lamentela",
        ],
    )
    def test_complaint_keywords_escalate(
        self, guardrails: Guardrails, message: str
    ) -> None:
        result: GuardrailResult = guardrails.check(message)
        assert result.should_escalate is True
        assert result.reason == "Sensitive topic requiring owner"

    @pytest.mark.parametrize(
        "message",
        [
            "reclamo",
            "rimborso parziale",
        ],
    )
    def test_short_complaint_messages(
        self, guardrails: Guardrails, message: str
    ) -> None:
        result: GuardrailResult = guardrails.check(message)
        assert result.should_escalate is True


# ---------------------------------------------------------------------------
# Rule 1: Sensitive topics — allergy / intolerance
# ---------------------------------------------------------------------------


class TestSensitiveTopicAllergy:
    """Messages mentioning allergies or intolerances must escalate."""

    @pytest.mark.parametrize(
        "message",
        [
            "Ho una allergia alle arachidi",
            "Soffro di intolleranza al lattosio",
            "Avete ingredienti allergenici?",
            "Mio figlio ha un'allergia",
        ],
    )
    def test_allergy_keywords_escalate(
        self, guardrails: Guardrails, message: str
    ) -> None:
        result: GuardrailResult = guardrails.check(message)
        assert result.should_escalate is True
        assert result.reason == "Sensitive topic requiring owner"

    @pytest.mark.parametrize(
        "message",
        [
            "allergia",
            "intolleranza alimentare",
        ],
    )
    def test_short_allergy_messages(
        self, guardrails: Guardrails, message: str
    ) -> None:
        result: GuardrailResult = guardrails.check(message)
        assert result.should_escalate is True


# ---------------------------------------------------------------------------
# Rule 2: Price queries — must escalate, never hallucinate prices
# ---------------------------------------------------------------------------


class TestPriceQuery:
    """Price-related questions must always be escalated."""

    @pytest.mark.parametrize(
        "message",
        [
            "Qual è il prezzo della margherita?",
            "Quanto costa un caffè?",
            "Vorrei sapere il costo del taglio",
            "Quanto costano i prodotti?",
            "Che prezzo ha il menu?",
            "Avete un listino prezzi?",
        ],
    )
    def test_price_keywords_escalate(
        self, guardrails: Guardrails, message: str
    ) -> None:
        result: GuardrailResult = guardrails.check(message)
        assert result.should_escalate is True
        assert result.reason == "Price query — never hallucinate prices"

    @pytest.mark.parametrize(
        "message",
        [
            "prezzo",
            "costo della pizza",
        ],
    )
    def test_short_price_messages(
        self, guardrails: Guardrails, message: str
    ) -> None:
        result: GuardrailResult = guardrails.check(message)
        assert result.should_escalate is True


# ---------------------------------------------------------------------------
# Rule 3: Greeting-only messages — safe, high confidence
# ---------------------------------------------------------------------------


class TestGreetingOnly:
    """Greeting-only messages are safe to auto-reply with confidence 1.0."""

    @pytest.mark.parametrize(
        "message",
        [
            "Ciao",
            "Buongiorno",
            "Salve",
            "",
        ],
    )
    def test_greeting_only_does_not_escalate(
        self, guardrails: Guardrails, message: str
    ) -> None:
        result: GuardrailResult = guardrails.check(message)
        assert result.should_escalate is False
        assert result.confidence == 1.0

    def test_greeting_only_reason(self, guardrails: Guardrails) -> None:
        result: GuardrailResult = guardrails.check("Ciao")
        assert result.reason == "Greeting-only — safe to auto-reply"

    def test_empty_message(self, guardrails: Guardrails) -> None:
        result: GuardrailResult = guardrails.check("")
        assert result.should_escalate is False
        assert result.confidence == 1.0
        assert result.reason == "Empty message — no action needed"


# ---------------------------------------------------------------------------
# Rule 4: Opening hours queries — safe FAQ
# ---------------------------------------------------------------------------


class TestHoursQuery:
    """Hours/opening-time queries are safe FAQ items."""

    @pytest.mark.parametrize(
        "message",
        [
            "Quali sono i vostri orari?",
            "Siete aperti la domenica?",
            "Siete aperti domani?",
            "Fino a che ora siete aperti?",
            "Quando aprite oggi?",
        ],
    )
    def test_hours_queries_do_not_escalate(
        self, guardrails: Guardrails, message: str
    ) -> None:
        result: GuardrailResult = guardrails.check(message)
        assert result.should_escalate is False

    def test_hours_confidence(self, guardrails: Guardrails) -> None:
        result: GuardrailResult = guardrails.check("Siete aperti oggi?")
        assert result.confidence == 0.95
        assert result.reason == "Opening hours FAQ — safe to auto-reply"


# ---------------------------------------------------------------------------
# Rule 5: Booking queries — safe to collect info, no confirmation
# ---------------------------------------------------------------------------


class TestBooking:
    """Booking requests are safe to handle (collect info, never confirm)."""

    @pytest.mark.parametrize(
        "message",
        [
            "Vorrei fare una prenotazione per stasera",
            "Vorrei prenotare un tavolo per due",
            "Posso prenotare per sabato?",
            "Vorrei venire a cena domani",
        ],
    )
    def test_booking_does_not_escalate(
        self, guardrails: Guardrails, message: str
    ) -> None:
        result: GuardrailResult = guardrails.check(message)
        assert result.should_escalate is False
        assert result.reason == "Booking request — collect info only, no confirmation"

    def test_booking_confidence(self, guardrails: Guardrails) -> None:
        result: GuardrailResult = guardrails.check("Vorrei prenotare")
        assert result.confidence == 0.85


# ---------------------------------------------------------------------------
# Rule 6: Complex / long messages — reduced confidence
# ---------------------------------------------------------------------------


class TestComplexLanguage:
    """Long or complex messages should get a reduced confidence score."""

    def test_long_message_reduces_confidence(
        self, guardrails: Guardrails
    ) -> None:
        """Messages over 200 characters should have confidence < 1.0."""
        long_msg = (
            "Buongiorno, volevo chiedere informazioni sui vostri servioni. "
            "In particolare mi interessa sapere se fate anche consegne a domicilio "
            "e se avete un menu speciale per il weekend. Inoltre, vorrei sapere "
            "se è possibile ordinare online oppure devo passare direttamente in "
            "negozio. Grazie mille e buona giornata!"
        )
        assert len(long_msg) > 200
        result: GuardrailResult = guardrails.check(long_msg)
        assert result.should_escalate is False
        assert result.confidence < 1.0

    def test_very_long_sentence_reduces_confidence(
        self, guardrails: Guardrails
    ) -> None:
        """A single sentence over 200 chars should reduce confidence."""
        long_sentence = (
            "Volevo chiedere se per caso fate anche consegne a domicilio "
            "per la zona del centro storico durante il weekend e se avete "
            "un menu speciale per le occasioni come compleanni o feste "
            "perché vorrei organizzare una cena con i miei amici la prossima "
            "settimana e devo capire se siete la scelta giusta."
        )
        result: GuardrailResult = guardrails.check(long_sentence)
        assert result.confidence < 1.0

    @pytest.mark.parametrize(
        "message",
        [
            "Ciao, come stai?",
            "Avete il menu?",
        ],
    )
    def test_short_message_full_confidence(
        self, guardrails: Guardrails, message: str
    ) -> None:
        """Short, simple messages retain full confidence."""
        result: GuardrailResult = guardrails.check(message)
        assert result.confidence == 1.0


# ---------------------------------------------------------------------------
# General / edge cases
# ---------------------------------------------------------------------------


class TestGuardrailsEdgeCases:
    """Edge cases and integration checks."""

    def test_normal_message_no_escalation(
        self, guardrails: Guardrails
    ) -> None:
        """A normal informational message should not escalate."""
        result: GuardrailResult = guardrails.check("Grazie, ci vediamo domani!")
        assert result.should_escalate is False

    def test_whitespace_only(self, guardrails: Guardrails) -> None:
        """Whitespace-only messages are treated as empty."""
        result: GuardrailResult = guardrails.check("   ")
        assert result.should_escalate is False
        assert result.confidence == 1.0

    def test_mixture_price_and_greeting_still_escalates(
        self, guardrails: Guardrails
    ) -> None:
        """Price keywords in a greeting-like message still escalate (priority)."""
        result: GuardrailResult = guardrails.check("Ciao, quanto costa la pizza?")
        assert result.should_escalate is True
