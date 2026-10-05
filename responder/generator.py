"""Reply generator — orchestrates guardrails → prompt building → OpenAI call."""

from __future__ import annotations

import time
from typing import Any

import structlog
from openai import OpenAI, RateLimitError, APIError, OpenAIError

from config import settings
from profiler.analyzer import VoiceProfile
from profiler.profile_store import ProfileStore
from .guardrails import Guardrails, GuardrailResult
from .prompt_builder import PromptBuilder

logger = structlog.get_logger(__name__)

# Max length for auto-generated replies (rule 6)
MAX_AUTO_REPLY_LENGTH: int = 500

# Escalation-safe fallback message
_ESCALATION_FALLBACK = (
    "Glielo chiedo al titolare e ti faccio sapere subito! "
    "Un attimo solo, per favore"
)

# Another fallback for complete API failure
_API_ERROR_FALLBACK = (
    "Mi dispiace, ho avuto un problema tecnico. "
    "Glielo chiedo al titolare e ti rispondo appena possibile!"
)


def generate_reply(
    message: str,
    profile_name: str,
    customer_name: str | None = None,
    customer_phone: str | None = None,
) -> dict[str, Any]:
    """Generate a WhatsApp reply using the owner's voice profile.

    Flow
    ----
    1. Load profile from ProfileStore
    2. Run guardrails — if escalation needed, notify the owner via Escalator
       and return the escalation message
    3. Build system prompt via PromptBuilder
    4. Call OpenAI chat.completions.create (gpt-4o-mini)
    5. Apply response length limit
    6. Log everything
    7. Handle rate limits (retry up to 3× with exponential backoff)

    Parameters
    ----------
    message:
        The incoming customer message.
    profile_name:
        Name of the VoiceProfile to use.
    customer_name:
        Optional name of the customer for personalised prompts.
    customer_phone:
        Optional phone number of the customer (as received from Twilio),
        included in the escalation alert sent to the owner.

    Returns
    -------
    dict with keys:
        reply (str)       — the generated reply or escalation message
        confidence (float) — confidence from guardrail check
        escalated (bool)   — whether this was escalated to the owner
        model (str)        — the model used
        prompt_used (str)  — the system prompt that was sent (truncated)
    """
    log = logger.bind(profile=profile_name)

    # ── 1. Load profile ──────────────────────────────────────────────────
    store = ProfileStore(persist_dir=settings.chroma_path)
    profile = store.load(profile_name)

    if profile is None:
        log.warning("profile_not_found", profile_name=profile_name)
        return _escalation_result(
            reason=f"Profilo '{profile_name}' non trovato.",
            confidence=1.0,
        )

    log.info("profile_loaded", name=profile.name)

    # ── 2. Guardrails ────────────────────────────────────────────────────
    guardrails = Guardrails()
    guardrail_result: GuardrailResult = guardrails.check(message)

    if guardrail_result.should_escalate:
        log.info(
            "message_escalated",
            reason=guardrail_result.reason,
            confidence=guardrail_result.confidence,
            message=message[:150],
        )
        # Notify the owner — `server` imports `responder.generator`, so this
        # import is function-local to avoid a circular import at module load.
        from server.escalation import Escalator

        Escalator().escalate(
            message=message,
            customer_phone=customer_phone or "unknown",
            reason=guardrail_result.reason or "",
            confidence=guardrail_result.confidence,
        )
        reply = _build_escalation_response(guardrail_result.reason)
        return {
            "reply": reply,
            "confidence": guardrail_result.confidence,
            "escalated": True,
            "model": "none",
            "prompt_used": "",
        }

    # ── 3. Build prompts ─────────────────────────────────────────────────
    builder = PromptBuilder(profile)
    system_prompt = builder.build_system_prompt()
    user_prompt = builder.build_user_prompt(message, customer_name)

    log.info(
        "generating_reply",
        system_prompt_truncated=system_prompt[:200],
        user_prompt=user_prompt[:150],
    )

    # ── 4. Call OpenAI (with retry) ──────────────────────────────────────
    reply_text = _call_openai(system_prompt, user_prompt, log)

    # ── 5. Apply length limit ────────────────────────────────────────────
    if len(reply_text) > MAX_AUTO_REPLY_LENGTH:
        reply_text = reply_text[:MAX_AUTO_REPLY_LENGTH].rsplit(" ", 1)[0] + "…"
        log.info("reply_truncated", original_length=len(reply_text))

    # ── 6. Log everything ────────────────────────────────────────────────
    log.info(
        "reply_generated",
        reply=reply_text[:200],
        reply_length=len(reply_text),
        confidence=guardrail_result.confidence,
        model=settings.openai_model,
        escalated=False,
    )

    return {
        "reply": reply_text,
        "confidence": guardrail_result.confidence,
        "escalated": False,
        "model": settings.openai_model,
        "prompt_used": system_prompt,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _call_openai(system_prompt: str, user_prompt: str, log: Any) -> str:
    """Call OpenAI chat.completions with retry on 429 rate limits.

    Retries: up to 3 attempts with exponential backoff (2s, 4s, 8s).
    Any OpenAIError (missing credentials, auth, connection, API) returns the
    clean _API_ERROR_FALLBACK instead of propagating an uncaught exception.
    """
    max_retries = 3
    base_delay = 2.0

    for attempt in range(1, max_retries + 1):
        try:
            client = OpenAI(api_key=settings.openai_api_key)
            response = client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=settings.openai_temperature,
                max_tokens=settings.openai_max_tokens,
            )

            content = response.choices[0].message.content
            if content is None:
                log.error("openai_empty_response")
                return _API_ERROR_FALLBACK
            return content.strip()

        except RateLimitError as exc:
            delay = base_delay * (2 ** (attempt - 1))
            log.warning(
                "openai_rate_limit",
                attempt=attempt,
                max_retries=max_retries,
                retry_after_seconds=delay,
                error=str(exc),
            )
            if attempt < max_retries:
                time.sleep(delay)
            else:
                log.error("openai_rate_limit_exhausted", max_retries=max_retries)
                return _API_ERROR_FALLBACK

        except APIError as exc:
            log.error(
                "openai_api_error",
                status_code=getattr(exc, "status_code", None),
                error=str(exc),
            )
            # Non-retryable — return fallback immediately
            return _API_ERROR_FALLBACK

        except OpenAIError as exc:
            # Base class: covers missing credentials, auth, connection errors.
            log.error(
                "openai_client_error",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return _API_ERROR_FALLBACK

    # Should not reach here, but safety net
    return _API_ERROR_FALLBACK


def _build_escalation_response(reason: str | None) -> str:
    """Build a natural Italian escalation response based on the reason."""
    if reason and "prezzo" in reason.lower():
        return (
            "Glielo chiedo al titolare e ti faccio sapere subito "
            "per quanto riguarda i prezzi! Un attimo"
        )
    if reason and "sensitiv" in reason.lower():
        return (
            "Questa cosa la gestisco meglio col titolare — "
            "glielo chiedo subito e ti faccio sapere!"
        )
    # Generic escalation
    return _ESCALATION_FALLBACK


def _escalation_result(
    reason: str,
    confidence: float,
) -> dict[str, Any]:
    """Build an escalation dict used when the profile can't even be loaded."""
    return {
        "reply": _build_escalation_response(reason),
        "confidence": confidence,
        "escalated": True,
        "model": "none",
        "prompt_used": "",
    }
