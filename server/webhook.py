"""FastAPI webhook — receives incoming WhatsApp messages from Twilio.

Exposes:
- POST /webhook/twilio — parses Twilio WhatsApp webhooks and returns TwiML
- GET  /health        — health check endpoint
- GET  /profiles      — lists all stored voice profiles
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import structlog
from fastapi import FastAPI, Form, HTTPException, Request
from twilio.twiml.messaging_response import MessagingResponse

from config import settings
from profiler.profile_store import ProfileStore
from responder.generator import generate_reply

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# In-memory rate limiter: phone_number -> [list of timestamps]
# ---------------------------------------------------------------------------
_rate_limit_buckets: dict[str, list[float]] = defaultdict(list)
_RATE_LIMIT_WINDOW: int = 60  # seconds
_RATE_LIMIT_MAX: int = 50     # max requests per window


def _check_rate_limit(phone: str) -> bool:
    """Check if *phone* has exceeded the rate limit.

    Returns True if the request is allowed, False if rate-limited.
    """
    now = time.time()
    bucket = _rate_limit_buckets[phone]

    # Prune timestamps older than the window
    cutoff = now - _RATE_LIMIT_WINDOW
    _rate_limit_buckets[phone] = [t for t in bucket if t > cutoff]

    if len(_rate_limit_buckets[phone]) >= _RATE_LIMIT_MAX:
        return False

    _rate_limit_buckets[phone].append(now)
    return True


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="WhatsApp Voice Clone Webhook")

logger.info("webhook_app_initialized", rate_limit_max=_RATE_LIMIT_MAX, rate_limit_window=_RATE_LIMIT_WINDOW)


@app.post("/webhook/twilio")
async def webhook_twilio(
    request: Request,
    From: str = Form(...),
    Body: str = Form(""),
    ProfileName: str | None = Form(None),
) -> str:
    """Handle incoming WhatsApp messages from Twilio.

    Twilio sends a POST with form-encoded fields:
    - From         — the sender's WhatsApp number (e.g., 'whatsapp:+393451234567')
    - Body         — the message text
    - ProfileName  — optional WhatsApp Profile Name

    Returns a TwiML XML string with the generated reply.
    """
    start_time = time.monotonic()
    log = logger.bind(phone=From, message_truncated=Body[:100])

    # ── Rate limiting ───────────────────────────────────────────────────
    if not _check_rate_limit(From):
        log.warning("rate_limit_exceeded", phone=From)
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
        )

    # ── Log incoming webhook ────────────────────────────────────────────
    log.info(
        "incoming_webhook",
        phone=From,
        message=Body[:100],
        profile_name=ProfileName,
    )

    # ── Generate reply ──────────────────────────────────────────────────
    try:
        result: dict[str, Any] = generate_reply(
            message=Body,
            profile_name=settings.profile_name,
            customer_name=ProfileName,
        )
        reply_text: str = result.get("reply", "")
    except Exception:
        log.exception("generate_reply_failed")
        reply_text = (
            "Mi dispiace, ho avuto un problema tecnico. "
            "Glielo chiedo al titolare e ti rispondo appena possibile! 🙏"
        )

    # ── Build TwiML response ────────────────────────────────────────────
    response = MessagingResponse()
    response.message(reply_text)

    twiml_result = str(response)
    latency_ms = (time.monotonic() - start_time) * 1000

    # ── Log completed webhook ───────────────────────────────────────────
    log.info(
        "webhook_response_sent",
        response=twiml_result[:200],
        latency_ms=round(latency_ms, 1),
        response_length=len(twiml_result),
    )

    return twiml_result


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "profile": settings.profile_name,
        "model": settings.openai_model,
    }


@app.get("/profiles")
async def list_profiles() -> list[dict]:
    """List all stored voice profiles."""
    store = ProfileStore(persist_dir=settings.chroma_path)
    profiles = store.list_profiles()
    logger.info("profiles_listed", count=len(profiles))
    return profiles
