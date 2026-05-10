"""Message router — intent detection and routing for incoming customer messages."""

from __future__ import annotations

import re
from typing import Any

import structlog

from responder.guardrails import Guardrails

logger = structlog.get_logger(__name__)

# Keyword lists for each intent (lowercase).
# Priority order (highest first): complaint > price > booking > menu > hours > greeting > unknown

_GREETING_KEYWORDS: set[str] = {
    "ciao", "buongiorno", "buonasera", "buon pomeriggio", "salve",
    "hey", "ehi", "ehilà", "pronto", "arrivederci", "a presto",
    "a dopo", "ci sentiamo", "buona giornata",
}

_HOURS_KEYWORDS: list[str] = [
    "orari", "aperto", "chiuso", "quando siete aperti", "domani",
    "aprite", "chiudete", "fino a che ora", "oggi siete aperti",
    "domani siete aperti",
]

_BOOKING_PATTERNS: list[str] = [
    "prenot", "fissare", "appuntamento", "vorrei venire", "posto",
    "vorrei prenotare", "posso prenotare", "prendere un appuntamento",
    "fissiamo", "fissare",
]

_MENU_KEYWORDS: set[str] = {
    "menù", "menu", "cosa avete", "listino", "prodotti",
    "cosa fate", "che cosa avete",
}

_PRICE_KEYWORDS: set[str] = {
    "prezzo", "prezzi", "costo", "costi", "quanto costa",
    "quanto costano", "tariffa", "tariffe", "che prezzo",
}

_COMPLAINT_KEYWORDS: set[str] = {
    "lamentela", "reclamo", "insoddisfatto", "problema",
    "lamentele", "insoddisfatta",
}


class MessageRouter:
    """Routes incoming messages to the appropriate handler based on intent detection.

    Uses simple keyword matching with a priority-ordered list of intents.
    """

    def __init__(self) -> None:
        self.guardrails = Guardrails()
        self.logger = logger.bind(component="message_router")

    def route(self, message: str, customer_name: str | None = None) -> dict[str, Any]:
        """Detect the intent of an incoming customer message.

        Parameters
        ----------
        message:
            The incoming customer message text.
        customer_name:
            Optional name of the sender for context.

        Returns
        -------
        dict with keys:
            intent (str):           The detected intent label.
            should_escalate (bool): Whether the message requires escalation.
            info (dict):            Extracted metadata about the message.
        """
        text = message.strip().lower()
        text_len = len(message)
        contains_question = "?" in message

        keywords_found: list[str] = []

        # ── Intent detection (priority order: complaint > price > booking > menu > hours > greeting > unknown) ──

        # 1. Complaint (highest priority)
        matched = self._match_keywords(text, _COMPLAINT_KEYWORDS, _COMPLAINT_KEYWORDS)
        if matched:
            keywords_found = matched
            self.logger.info("intent_detected", intent="complaint", keywords=matched)
            return self._result(
                intent="complaint",
                should_escalate=True,
                keywords=keywords_found,
                message_length=text_len,
                contains_question=contains_question,
            )

        # 2. Price
        matched = self._match_keywords(text, _PRICE_KEYWORDS, _PRICE_KEYWORDS)
        if matched:
            keywords_found = matched
            self.logger.info("intent_detected", intent="price", keywords=matched)
            return self._result(
                intent="price",
                should_escalate=True,
                keywords=keywords_found,
                message_length=text_len,
                contains_question=contains_question,
            )

        # 3. Booking
        matched = self._match_keywords(
            text,
            exact_keywords=set(),
            substring_patterns=_BOOKING_PATTERNS,
        )
        if matched:
            keywords_found = matched
            self.logger.info("intent_detected", intent="booking", keywords=matched)
            return self._result(
                intent="booking",
                should_escalate=False,
                keywords=keywords_found,
                message_length=text_len,
                contains_question=contains_question,
            )

        # 4. Menu
        matched = self._match_keywords(text, _MENU_KEYWORDS, _MENU_KEYWORDS)
        if matched:
            keywords_found = matched
            self.logger.info("intent_detected", intent="menu", keywords=matched)
            return self._result(
                intent="menu",
                should_escalate=False,
                keywords=keywords_found,
                message_length=text_len,
                contains_question=contains_question,
            )

        # 5. Hours
        matched = self._match_keywords(
            text,
            exact_keywords=set(),
            substring_patterns=_HOURS_KEYWORDS,
        )
        if matched:
            keywords_found = matched
            self.logger.info("intent_detected", intent="hours", keywords=matched)
            return self._result(
                intent="hours",
                should_escalate=False,
                keywords=keywords_found,
                message_length=text_len,
                contains_question=contains_question,
            )

        # 6. Greeting — short message or greeting-only words
        if text_len < 10 or self._is_greeting(text):
            self.logger.info("intent_detected", intent="greeting")
            return self._result(
                intent="greeting",
                should_escalate=False,
                keywords=keywords_found,
                message_length=text_len,
                contains_question=contains_question,
                confidence=1.0,
            )

        # 7. Unknown (fallback)
        self.logger.info("intent_detected", intent="unknown")
        return self._result(
            intent="unknown",
            should_escalate=False,
            keywords=keywords_found,
            message_length=text_len,
            contains_question=contains_question,
            confidence=0.5,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _match_keywords(
        text: str,
        exact_keywords: set[str],
        substring_patterns: list[str] | set[str] | None = None,
    ) -> list[str]:
        """Match keywords against the message text.

        Parameters
        ----------
        text:
            The lowercased message text.
        exact_keywords:
            Set of keywords to match exactly (word boundary).
        substring_patterns:
            Patterns to match via substring search.

        Returns
        -------
        list[str]: List of matched keywords/patterns.
        """
        found: list[str] = []

        # Exact word-boundary matching
        if exact_keywords:
            for kw in exact_keywords:
                if re.search(rf"\b{re.escape(kw)}\b", text):
                    found.append(kw)

        # Substring matching (for multi-word phrases or partial matches)
        if substring_patterns:
            for pat in substring_patterns:
                if pat in text:
                    found.append(pat)

        return found

    @staticmethod
    def _is_greeting(text: str) -> bool:
        """Check if the message is purely a greeting."""
        # Check against greeting keywords with word boundaries
        for greet in _GREETING_KEYWORDS:
            if re.search(rf"\b{re.escape(greet)}\b", text):
                return True
        return False

    @staticmethod
    def _result(
        intent: str,
        should_escalate: bool,
        keywords: list[str],
        message_length: int,
        contains_question: bool,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        """Build the standard route result dictionary."""
        result: dict[str, Any] = {
            "intent": intent,
            "should_escalate": should_escalate,
            "info": {
                "keywords_found": keywords,
                "message_length": message_length,
                "contains_question": contains_question,
            },
        }
        if confidence is not None:
            result["info"]["confidence"] = confidence
        return result
