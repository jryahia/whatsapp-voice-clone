"""Guardrails — safety checks that decide whether a message can be auto-replied
or must be escalated to the business owner."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)

# Greeting-only words (lowercase, stripped of punctuation)
_GREETING_WORDS: set[str] = {
    "ciao", "buongiorno", "buonasera", "buon pomeriggio", "salve",
    "hey", "ehilà", "ehi", "pronto", "sì", "no", "ok", "okay",
    "grazie", "grazie mille", "grazie tante", "prego", "arrivederci",
    "a presto", "a dopo", "ci sentiamo", "buona giornata",
}

# Regex combining all escalation triggers for _detect_sensitive_topics
_SENSITIVE_PATTERN = re.compile(
    r"(?i)\b("
    r"cancellazion[ei]|disdetta|rimbors[oi]|reclam[oi]|"
    r"allergi[ae]|allergenici?|intolleranz[ae]|"
    r"lamentela|lamentele"
    r")\b"
)

# Price queries
_PRICE_PATTERN = re.compile(
    r"(?i)\b(?:"
    r"prezzo|prezzi|costo|costi|quanto\s+costa|quanto\s+costano|"
    r"tariffa|tariffe|che\s+prezzo|listino"
    r")\b"
)

# Booking / appointment indicators — these are SAFE to auto-handle (collect info only)
_BOOKING_PATTERN = re.compile(
    r"(?i)\b(?:"
    r"prenotazion[ei]|prenotare|fissare\s+un\s+appuntamento|"
    r"vorrei\s+venir[ae]|vorrei\s+prenotare|posso\s+prenotare|"
    r"prendere\s+un\s+appuntamento|fissiamo|fissare"
    r")\b"
)

# Hours / opening time queries — safe FAQ
_HOURS_PATTERN = re.compile(
    r"(?i)\b(?:"
    r"aperto|chiuso|orari|orario|siete\s+apert[io]|"
    r"aprite|chiudete|fino\s+a\s+che\s+ora|quando\s+siete\s+apert[io]|"
    r"oggi\s+siete\s+apert[io]|domani\s+siete\s+apert[io]"
    r")\b"
)


@dataclass
class GuardrailResult:
    """Result of a guardrail check on an incoming message."""

    should_escalate: bool = False
    reason: str | None = None
    confidence: float = 1.0


class Guardrails:
    """Evaluates incoming messages against a set of safety rules.

    The rules are ordered by priority; the first matching rule determines
    the result unless later rules further modify the confidence score.
    """

    def __init__(self) -> None:
        self.logger = logger.bind(component="guardrails")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, message: str) -> GuardrailResult:
        """Run every guardrail rule against *message* and return the result.

        Returns
        -------
        GuardrailResult
            • should_escalate=True  → message must go to the owner
            • should_escalate=False → safe for auto-reply
            • confidence            → how sure we are about the classification
        """
        stripped = message.strip()
        if not stripped:
            return GuardrailResult(
                should_escalate=False,
                reason="Empty message — no action needed",
                confidence=1.0,
            )

        # 1) Sensitive topics → escalate immediately
        if self._detect_sensitive_topics(stripped):
            self.logger.info("guardrail_triggered_sensitive_topic", message=stripped[:100])
            return GuardrailResult(
                should_escalate=True,
                reason="Sensitive topic requiring owner",
                confidence=1.0,
            )

        # 2) Price queries → escalate immediately
        if _PRICE_PATTERN.search(stripped):
            self.logger.info("guardrail_triggered_price_query", message=stripped[:100])
            return GuardrailResult(
                should_escalate=True,
                reason="Price query — never hallucinate prices",
                confidence=1.0,
            )

        # 3) Greeting-only messages → safe, high confidence
        if self._is_greeting_only(stripped):
            self.logger.info("guardrail_greeting_only", message=stripped[:100])
            return GuardrailResult(
                should_escalate=False,
                reason="Greeting-only — safe to auto-reply",
                confidence=1.0,
            )

        # 4) Hours/opening time queries → safe FAQ (router handles it)
        if _HOURS_PATTERN.search(stripped):
            self.logger.info("guardrail_safe_faq_hours", message=stripped[:100])
            return GuardrailResult(
                should_escalate=False,
                reason="Opening hours FAQ — safe to auto-reply",
                confidence=0.95,
            )

        # 5) Booking indicators → safe to handle (collect info, don't confirm)
        if _BOOKING_PATTERN.search(stripped):
            self.logger.info("guardrail_booking_query", message=stripped[:100])
            return GuardrailResult(
                should_escalate=False,
                reason="Booking request — collect info only, no confirmation",
                confidence=0.85,
            )

        # 6) Complex Italian → reduce confidence
        confidence = self._compute_confidence(stripped)

        result = GuardrailResult(
            should_escalate=False,
            reason=None,
            confidence=confidence,
        )

        self.logger.debug(
            "guardrail_check_passed",
            message=stripped[:80],
            confidence=round(confidence, 3),
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_sensitive_topics(self, message: str) -> bool:
        """Regex-based detection of escalation triggers (rule 1)."""
        return bool(_SENSITIVE_PATTERN.search(message))

    @staticmethod
    def _is_greeting_only(text: str) -> bool:
        """Check if the message is purely a greeting / short acknowledgment.

        Rule 4: under 5 chars → automatically greeting.
        Otherwise, check every word against known greeting words.
        """
        if len(text) <= 5:
            return True

        # Strip punctuation and split
        cleaned = re.sub(r"[^\w\s'àèéìòù]", " ", text.lower())
        words = [w for w in cleaned.split() if w]

        if not words:
            return False

        # If all words are greeting words, it's greeting-only
        return all(w in _GREETING_WORDS for w in words)

    @staticmethod
    def _compute_confidence(text: str) -> float:
        """Compute a confidence score based on message complexity.

        Rule 5: long sentences, multiple punctuation, mixed languages → reduce confidence.
        Base is 1.0; each complexity factor multiplies a discount.
        """
        confidence = 1.0

        # Long messages (over 200 chars)
        if len(text) > 200:
            confidence *= 0.80

        # Multiple punctuation marks (!!!, ???, !?, etc.)
        multi_punct_count = len(re.findall(r"[!?]{2,}", text))
        if multi_punct_count > 0:
            confidence *= 0.85 ** multi_punct_count

        # Mixed languages detection: non-Italian characters or patterns
        # If the message contains many non-Italian, non-ASCII letters
        non_italian = re.findall(r"[^\w\sàèéìòù'!?.,:;()\-]", text, re.UNICODE)
        if len(non_italian) > 5:
            confidence *= 0.80

        # Very long individual sentences (over 200 chars between sentence breaks)
        sentences = re.split(r"[.!?]+", text)
        for sent in sentences:
            if len(sent.strip()) > 200:
                confidence *= 0.80
                break

        return max(0.1, min(1.0, round(confidence, 4)))
