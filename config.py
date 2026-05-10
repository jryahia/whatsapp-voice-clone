"""Pydantic config model with JSON + .env override support."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── OpenAI ──────────────────────────────────────────────────────────────
    openai_api_key: str = Field("", validation_alias="OPENAI_API_KEY")
    openai_model: str = "gpt-4o-mini"
    openai_max_tokens: int = 512
    openai_temperature: float = 0.7

    # ── Twilio ──────────────────────────────────────────────────────────────
    twilio_account_sid: str = Field("", validation_alias="TWILIO_ACCOUNT_SID")
    twilio_auth_token: str = Field("", validation_alias="TWILIO_AUTH_TOKEN")
    twilio_whatsapp_number: str = ""

    # ── App ─────────────────────────────────────────────────────────────────
    profile_name: str = "default"
    port: int = 8080
    host: str = "0.0.0.0"
    chroma_path: str = str(Path.home() / ".voice_clone_profiles")
    log_level: str = "INFO"
    max_messages_per_minute: int = 50
    confidence_threshold: float = 0.7
    escalation_telegram_bot_token: str = Field("", validation_alias="ESCALATION_TELEGRAM_BOT_TOKEN")
    escalation_telegram_chat_id: str = Field("", validation_alias="ESCALATION_TELEGRAM_CHAT_ID")

    @field_validator("openai_api_key", "twilio_account_sid", "twilio_auth_token")
    @classmethod
    def warn_if_empty(cls, v: str) -> str:
        return v

    @classmethod
    def from_json(cls, path: str | Path) -> "Settings":
        p = Path(path)
        if not p.exists():
            return cls()
        with open(p) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if v})


settings = Settings()
