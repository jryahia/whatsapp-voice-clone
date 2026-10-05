"""Escalation module — sends alerts to the business owner via Telegram bot API.

If Telegram credentials are not configured, escalations are logged locally
instead of sent to Telegram.
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog

from config import settings

logger = structlog.get_logger(__name__)

_ESCALATION_MESSAGE_TEMPLATE: str = (
    "ESCALATION - {profile_name}\n"
    "Cliente: {phone}\n"
    "Messaggio: {message}\n"
    "Motivo: {reason}\n"
    "Confidence: {confidence}"
)

_TELEGRAM_API_BASE: str = "https://api.telegram.org/bot{token}/sendMessage"

_MAX_MESSAGE_LENGTH: int = 200


class Escalator:
    """Sends escalation alerts to the business owner via Telegram.

    Usage::

        escalator = Escalator()
        success = escalator.escalate(
            message="Quanto costa la pizza margherita?",
            customer_phone="+393451234567",
            reason="Price query — never hallucinate prices",
            confidence=1.0,
        )
    """

    def __init__(self) -> None:
        self.bot_token: str = settings.escalation_telegram_bot_token
        self.chat_id: str = settings.escalation_telegram_chat_id
        self._configured: bool = bool(self.bot_token and self.chat_id)
        self.logger = logger.bind(
            component="escalator",
            telegram_configured=self._configured,
        )

        if not self._configured:
            self.logger.info(
                "escalation_telegram_not_configured",
                hint="Set ESCALATION_TELEGRAM_BOT_TOKEN and ESCALATION_TELEGRAM_CHAT_ID to enable Telegram alerts.",
            )

    def escalate(
        self,
        message: str,
        customer_phone: str,
        reason: str,
        confidence: float,
    ) -> bool:
        """Send an escalation alert via Telegram bot API.

        Parameters
        ----------
        message:
            The original customer message that triggered the escalation.
        customer_phone:
            The customer's phone number (as received from Twilio).
        reason:
            A human-readable reason for the escalation.
        confidence:
            The confidence score from the guardrail/message analysis.

        Returns
        -------
        bool:
            True if the escalation was handled (sent via Telegram or logged locally).
            False only if Telegram API call fails with a non-configuration error.
        """
        log = self.logger.bind(
            phone=customer_phone,
            reason=reason,
            confidence=round(confidence, 3),
        )

        # Truncate message to max length
        truncated_message: str = message[: _MAX_MESSAGE_LENGTH]
        if len(message) > _MAX_MESSAGE_LENGTH:
            truncated_message += "…"

        # If Telegram is not configured, log locally and return success
        if not self._configured:
            log.warning(
                "escalation_logged_locally",
                message_text=truncated_message,
                escalation_message=_ESCALATION_MESSAGE_TEMPLATE.format(
                    profile_name=settings.profile_name,
                    phone=customer_phone,
                    message=truncated_message,
                    reason=reason,
                    confidence=f"{confidence:.2f}",
                ),
            )
            return True

        # Build the Telegram message
        telegram_text: str = _ESCALATION_MESSAGE_TEMPLATE.format(
            profile_name=settings.profile_name,
            phone=customer_phone,
            message=truncated_message,
            reason=reason,
            confidence=f"{confidence:.2f}",
        )

        # Send via Telegram bot API
        url: str = _TELEGRAM_API_BASE.format(token=self.bot_token)
        payload: dict[str, Any] = {
            "chat_id": self.chat_id,
            "text": telegram_text,
            "parse_mode": "HTML",
        }

        try:
            with httpx.Client(timeout=10.0) as client:
                response: httpx.Response = client.post(url, json=payload)

            if response.status_code == 200:
                log.info("escalation_sent_via_telegram")
                return True

            # Log the error
            response_data: Any = response.json()
            error_description: str = response_data.get("description", "Unknown error")

            log.error(
                "telegram_api_error",
                status_code=response.status_code,
                description=error_description,
            )

            # If it's a configuration error (wrong token/chat_id), treat as handled
            if response.status_code in (401, 403, 404):
                log.warning(
                    "escalation_telegram_config_error",
                    error=error_description,
                    fallback="Escalation logged locally.",
                )
                return True

            # Non-configuration API error
            return False

        except httpx.TimeoutException:
            log.error("telegram_api_timeout")
            return False
        except httpx.RequestError as exc:
            log.error("telegram_api_request_error", error=str(exc))
            return False
