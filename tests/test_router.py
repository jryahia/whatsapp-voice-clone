"""Tests for the MessageRouter module — intent detection and routing."""

from __future__ import annotations

from typing import Any

import pytest

from server.router import MessageRouter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def router() -> MessageRouter:
    """Provide a fresh MessageRouter instance for each test."""
    return MessageRouter()


# ---------------------------------------------------------------------------
# Intent: greeting
# ---------------------------------------------------------------------------


class TestIntentGreeting:
    """Short greeting messages should resolve to intent='greeting'."""

    @pytest.mark.parametrize(
        "message",
        [
            "Ciao!",
            "Buongiorno",
            "Salve",
            "Hey",
            "A presto",
            "Ciao a tutti",
        ],
    )
    def test_greeting_intent(self, router: MessageRouter, message: str) -> None:
        result: dict[str, Any] = router.route(message)
        assert result["intent"] == "greeting"
        assert result["should_escalate"] is False

    def test_greeting_short_message(self, router: MessageRouter) -> None:
        """Messages under 10 characters trigger greeting intent."""
        result: dict[str, Any] = router.route("Ok")
        assert result["intent"] == "greeting"

    def test_greeting_contains_question_flag(
        self, router: MessageRouter
    ) -> None:
        result: dict[str, Any] = router.route("Ciao?")
        assert result["info"]["contains_question"] is True

    def test_greeting_no_question(self, router: MessageRouter) -> None:
        result: dict[str, Any] = router.route("Ciao")
        assert result["info"]["contains_question"] is False


# ---------------------------------------------------------------------------
# Intent: hours
# ---------------------------------------------------------------------------


class TestIntentHours:
    """Messages about opening hours should resolve to intent='hours'."""

    @pytest.mark.parametrize(
        "message",
        [
            "Siete aperti domani?",
            "Quali sono i vostri orari?",
            "Quando aprite?",
            "Fino a che ora siete aperti?",
            "Siete aperti oggi?",
            "A che ora chiudete?",
        ],
    )
    def test_hours_intent(self, router: MessageRouter, message: str) -> None:
        result: dict[str, Any] = router.route(message)
        assert result["intent"] == "hours"
        assert result["should_escalate"] is False

    def test_hours_contains_keywords(self, router: MessageRouter) -> None:
        result: dict[str, Any] = router.route("Siete aperti?")
        keywords = result["info"]["keywords_found"]
        assert any("aperto" in kw or "aperti" in kw for kw in keywords)


# ---------------------------------------------------------------------------
# Intent: booking
# ---------------------------------------------------------------------------


class TestIntentBooking:
    """Messages about bookings should resolve to intent='booking'."""

    @pytest.mark.parametrize(
        "message",
        [
            "Vorrei prenotare un tavolo",
            "Vorrei fare una prenotazione",
            "Posso prenotare per sabato?",
            "Vorrei venire a cena stasera",
            "Fissiamo un appuntamento",
            "Prendere un appuntamento",
        ],
    )
    def test_booking_intent(self, router: MessageRouter, message: str) -> None:
        result: dict[str, Any] = router.route(message)
        assert result["intent"] == "booking"
        assert result["should_escalate"] is False

    def test_booking_keywords_found(self, router: MessageRouter) -> None:
        result: dict[str, Any] = router.route("Vorrei prenotare un tavolo")
        keywords = result["info"]["keywords_found"]
        assert "prenotare" in keywords or "prenot" in " ".join(keywords)


# ---------------------------------------------------------------------------
# Intent: price
# ---------------------------------------------------------------------------


class TestIntentPrice:
    """Messages about prices should resolve to intent='price'."""

    @pytest.mark.parametrize(
        "message",
        [
            "Quanto costa la margherita?",
            "Qual è il prezzo del menu?",
            "Quanto costano i vostri prodotti?",
            "Che prezzo ha la pizza?",
            "Avete una tariffa speciale?",
        ],
    )
    def test_price_intent(self, router: MessageRouter, message: str) -> None:
        result: dict[str, Any] = router.route(message)
        assert result["intent"] == "price"
        assert result["should_escalate"] is True

    def test_price_keywords_found(self, router: MessageRouter) -> None:
        result: dict[str, Any] = router.route("Quanto costa?")
        keywords = result["info"]["keywords_found"]
        assert "quanto costa" in keywords or "costo" in keywords


# ---------------------------------------------------------------------------
# Intent: complaint
# ---------------------------------------------------------------------------


class TestIntentComplaint:
    """Messages with complaints should resolve to intent='complaint'."""

    @pytest.mark.parametrize(
        "message",
        [
            "Ho un reclamo",
            "Voglio fare una lamentela",
            "Sono insoddisfatto del servizio",
            "Ho un problema con l'ordine",
        ],
    )
    def test_complaint_intent(self, router: MessageRouter, message: str) -> None:
        result: dict[str, Any] = router.route(message)
        assert result["intent"] == "complaint"
        assert result["should_escalate"] is True

    def test_complaint_keywords_found(self, router: MessageRouter) -> None:
        result: dict[str, Any] = router.route("lamentela")
        keywords = result["info"]["keywords_found"]
        assert "lamentela" in keywords


# ---------------------------------------------------------------------------
# Intent: unknown
# ---------------------------------------------------------------------------


class TestIntentUnknown:
    """Messages that don't match any known intent should resolve to intent='unknown'."""

    @pytest.mark.parametrize(
        "message",
        [
            "Scusa mi serve una mano",
            "Puoi aiutarmi con qualcosa?",
            "Non so cosa fare",
            "Ho visto il vostro annuncio",
        ],
    )
    def test_unknown_intent(self, router: MessageRouter, message: str) -> None:
        result: dict[str, Any] = router.route(message)
        assert result["intent"] == "unknown"
        assert result["should_escalate"] is False

    def test_unknown_confidence(self, router: MessageRouter) -> None:
        result: dict[str, Any] = router.route("Scusa mi serve una mano")
        assert result["info"]["confidence"] == 0.5


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestRouterEdgeCases:
    """Additional edge cases for the router."""

    def test_empty_message(self, router: MessageRouter) -> None:
        result: dict[str, Any] = router.route("")
        assert result["intent"] == "greeting"

    def test_price_higher_priority_than_booking(
        self, router: MessageRouter
    ) -> None:
        """Price keywords should take priority over booking keywords."""
        result: dict[str, Any] = router.route(
            "Quanto costa prenotare un tavolo?"
        )
        # 'quanto costa' triggers price first
        assert result["intent"] == "price"

    def test_complaint_highest_priority(self, router: MessageRouter) -> None:
        """Complaint keywords should have the highest priority."""
        result: dict[str, Any] = router.route(
            "Ho un reclamo sul prezzo della pizza"
        )
        # Even though 'prezzo' matches price, 'reclamo' matches complaint first
        assert result["intent"] == "complaint"

    def test_hours_vs_booking_priority(self, router: MessageRouter) -> None:
        """Hours intent should be detected correctly vs booking."""
        result: dict[str, Any] = router.route("Siete aperti domani?")
        assert result["intent"] == "hours"

    def test_menu_intent(self, router: MessageRouter) -> None:
        """Messages about the menu should resolve to intent='menu'."""
        result: dict[str, Any] = router.route("Che cosa avete nel menu?")
        assert result["intent"] == "menu"
        assert result["should_escalate"] is False

    def test_result_structure(self, router: MessageRouter) -> None:
        """The route result should have the correct keys."""
        result: dict[str, Any] = router.route("Ciao!")
        assert "intent" in result
        assert "should_escalate" in result
        assert "info" in result
        assert "keywords_found" in result["info"]
        assert "message_length" in result["info"]
        assert "contains_question" in result["info"]

    def test_message_length_in_info(self, router: MessageRouter) -> None:
        result: dict[str, Any] = router.route("Ciao!")
        assert result["info"]["message_length"] == 5

    def test_with_customer_name(self, router: MessageRouter) -> None:
        """Customer name should not affect routing, but should not crash."""
        result_with: dict[str, Any] = router.route(
            "Ciao!", customer_name="Luca"
        )
        result_without: dict[str, Any] = router.route("Ciao!")
        assert result_with["intent"] == result_without["intent"]
