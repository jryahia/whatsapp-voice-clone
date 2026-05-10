#!/usr/bin/env python3
"""WhatsApp Voice Clone — CLI entry point.

Train a voice profile from exported WhatsApp chat, serve the webhook,
test prompts, and list stored profiles.

Usage:
    python main.py train --export data/exports/owner_chat.txt --name "Mario Pizzeria"
    python main.py serve --profile "Mario Pizzeria" --port 8080
    python main.py test-prompt --profile "Mario Pizzeria" --message "Ciao Mario, siete aperti domani?"
    python main.py list-profiles
"""
from __future__ import annotations

import sys

from cli import app

if __name__ == "__main__":
    sys.exit(app())
