"""Tests for the FastAPI webhook — TwiML media type, Twilio signature gate,
/profiles localhost host gate, and /health staying open."""
from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request
from twilio.request_validator import RequestValidator

from config import settings
from server import webhook

_TEST_TOKEN = "TEST_AUTH_TOKEN_123"
_WEBHOOK_URL = "http://testserver/webhook/twilio"
_BODY = {"From": "whatsapp:+393451234567", "Body": "Ciao", "ProfileName": "Marco"}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(settings, "twilio_auth_token", _TEST_TOKEN)
    return TestClient(webhook.app)


def _sig() -> str:
    return RequestValidator(_TEST_TOKEN).compute_signature(_WEBHOOK_URL, _BODY)


def _request_for(host: str) -> Request:
    """A synthetic GET request arriving from *host*."""
    scope = {
        "type": "http", "method": "GET", "path": "/profiles",
        "headers": [], "query_string": b"", "client": (host, 0),
        "server": ("testserver", 80), "scheme": "http", "root_path": "",
    }
    return Request(scope)


def _profiles_for(host: str):
    return asyncio.run(webhook.list_profiles(_request_for(host)))


class TestTwiMLMediaType:
    """Regression: /webhook/twilio returns RAW TwiML as text/xml, not JSON."""

    def test_returns_well_formed_xml(self, client: TestClient) -> None:
        resp = client.post("/webhook/twilio", data=_BODY,
                           headers={"X-Twilio-Signature": _sig()})

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/xml")
        assert resp.content.strip().startswith(b"<?xml ")
        root = ET.fromstring(resp.content)   # raises ParseError if malformed
        assert root.tag == "Response"
        assert len(list(root.iter("Message"))) >= 1


class TestTwilioSignatureGate:
    def test_valid_signature_allowed(self, client: TestClient) -> None:
        resp = client.post("/webhook/twilio", data=_BODY,
                           headers={"X-Twilio-Signature": _sig()})
        assert resp.status_code == 200

    @pytest.mark.parametrize("header", [{"X-Twilio-Signature": "tampered"}, {}])
    def test_bad_signature_rejected(self, client: TestClient, header: dict) -> None:
        resp = client.post("/webhook/twilio", data=_BODY, headers=header or None)
        assert resp.status_code == 403


class TestProfilesHostGate:
    def test_localhost_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import tempfile
        monkeypatch.setattr(settings, "chroma_path", tempfile.mkdtemp())
        assert isinstance(_profiles_for("127.0.0.1"), list)  # allowed, returns data

    def test_external_host_blocked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import tempfile
        monkeypatch.setattr(settings, "chroma_path", tempfile.mkdtemp())
        with pytest.raises(HTTPException) as excinfo:
            _profiles_for("203.0.113.50")
        assert excinfo.value.status_code == 403


class TestHealthStaysOpen:
    def test_health_open(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
