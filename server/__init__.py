"""Server module — FastAPI webhook, router, and escalation for WhatsApp Voice Clone."""

from __future__ import annotations

import structlog

from .webhook import app
from .escalation import Escalator
from .router import MessageRouter

__all__ = [
    "app",
    "Escalator",
    "MessageRouter",
]

logger = structlog.get_logger(__name__)


def start_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Start the FastAPI server using uvicorn.

    Parameters
    ----------
    host:
        The host interface to bind to (default: 0.0.0.0).
    port:
        The port to listen on (default: 8080).
    """
    import uvicorn

    logger.info("starting_server", host=host, port=port)
    uvicorn.run(
        "server.webhook:app",
        host=host,
        port=port,
        log_level="info",
    )
