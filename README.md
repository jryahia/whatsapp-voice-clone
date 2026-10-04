# WhatsApp Voice Clone

**Learns a business owner's WhatsApp writing style from chat exports and auto-replies to customers in that voice, within business guardrails.**

![Python](https://img.shields.io/badge/Python-161b22?style=for-the-badge&labelColor=161b22&color=161b22) ![FastAPI](https://img.shields.io/badge/FastAPI-161b22?style=for-the-badge&labelColor=161b22&color=161b22) ![OpenAI GPT-4o-mini](https://img.shields.io/badge/OpenAI%20GPT--4o--mini-161b22?style=for-the-badge&labelColor=161b22&color=161b22) ![Twilio](https://img.shields.io/badge/Twilio-161b22?style=for-the-badge&labelColor=161b22&color=161b22)

```mermaid
flowchart LR
    S0["Exported WhatsApp chats"]
    S1["Style profiler (6 dimensions)"]
    S2["Responder with guardrails"]
    S3["Twilio webhook server"]
    S4["Reply in owner's voice"]
    S0 --> S1 --> S2 --> S3 --> S4
```

## Problem it solves

Small Italian businesses answer customers on WhatsApp all day and cannot afford enterprise bots, and generic bots sound nothing like the owner. This system builds a style profile from real chats and replies through a Twilio webhook.

> **AI that clones any small business owner's WhatsApp communication style — then auto-replies to customers in their exact voice.**

A Python system that analyzes exported WhatsApp chat history, builds a 6-dimensional "voice profile" (formality, vocabulary, emoji style, greetings, response patterns, time habits), and uses GPT-4o-mini to respond to customers as if it were the owner. Unknown/complex messages escalate to the real person.

**Target market**: Italian SMEs — pizzerias, barbers, mechanics, shops — who use WhatsApp daily but can't afford enterprise SaaS bots.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   WhatsApp Voice Clone                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │   PROFILER   │    │   RESPONDER  │    │   SERVER   │ │
│  │              │    │              │    │            │ │
│  │ • analyzer   │───▶│ • prompt_    │───▶│ • webhook  │ │
│  │ • extractor  │    │   builder    │    │ • router   │ │
│  │ • profile_   │    │ • generator  │    │ • escalation│ │
│  │   store      │    │ • guardrails │    │            │ │
│  │              │    │              │    │            │ │
│  └──────┬───────┘    └──────┬───────┘    └─────┬──────┘ │
│         │                   │                   │         │
│         ▼                   ▼                   ▼         │
│  ┌───────────┐     ┌──────────────┐      ┌──────────┐   │
│  │ ChromaDB  │     │  OpenAI GPT  │      │  Twilio  │   │
│  │ (profile  │     │  -4o-mini    │      │ WhatsApp │   │
│  │ storage)  │     │  (inference) │      │  API     │   │
│  └───────────┘     └──────────────┘      └──────────┘   │
│                                                          │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
                  ┌────────────────┐
                  │   Telegram     │
                  │   Escalation   │
                  │   (owner gets  │
                  │   notified)    │
                  └────────────────┘
```

---

## Features

| Feature | Details |
|---------|---------|
| **Voice Profiling** | 6-dimensional analysis of exported WhatsApp chat |
| **GPT-4o-mini** | Low-cost, high-speed inference ($0.15/1M input tokens) |
| **ChromaDB Storage** | Multiple profiles in one database, metadata-filtered queries |
| **Twilio Webhook** | Real WhatsApp Business API integration |
| **Italian Native** | Full Italian language support, Lei/tu formality detection |
| **Business Guardrails** | 7 hard rules: no price hallucination, no auto-booking, complaint escalation |
| **Escalation** | Unknown/high-risk messages forwarded to owner via Telegram |
| **CLI** | 4 commands: `train`, `serve`, `test-prompt`, `list-profiles` |
| **Structured Logging** | JSON-structured logs via structlog for audit trail |
| **Async** | FastAPI async webhooks for low-latency responses |

---

## Voice Profiling (6 Dimensions)

The profiler extracts these dimensions from your exported WhatsApp chat:

### 1. Formality
Analyses formality level on a 0.0–1.0 scale:
- Lei vs tu pronoun usage
- Formal greetings (Buongiorno) vs casual (Ciao, Ehilà)
- Sentence complexity (avg words per sentence)
- Formal closings (Cordiali saluti, Distinti saluti)

### 2. Vocabulary
- Top 50 most-used words (Italian stopwords filtered out)
- Jargon/key terms detected (ALL CAPS or quoted words)
- Industry-specific vocabulary (e.g. pizza types, menu items)

### 3. Emoji Style
- Count and frequency (per 100 words)
- Preferred emojis ranked by usage
- Placement patterns (start, middle, or end of sentences)

### 4. Greetings
Detects and counts 11 Italian greeting patterns:
- Ciao, Buongiorno, Buonasera, Salve, Pronto, Ehilà, Ehi, Ohi, Senti, Allora, Guarda
- Distribution percentages and most common pattern

### 5. Response Style
- Average message length (chars and words)
- Question frequency (% of messages ending with ?)
- Exclamation frequency (% ending with !)
- Caps-lock tendency (ALL CAPS words)
- Punctuation habits (..., !!, ??, multi-punctuation)

### 6. Time Patterns
- Active hours (which hours of the day the owner responds)
- Busiest hour of the day
- Average response time between received and sent messages

---

## Guardrails & Safety

The system implements **7 hard business rules** to prevent mistakes:

1. **No price hallucination** — never generates a price the owner didn't set
2. **No auto-booking** — collects info but never confirms without owner
3. **Escalate sensitive topics** — cancellations, refunds, complaints, allergies
4. **Complex message detection** — long/complex messages reduce confidence
5. **Rate limiting** — max 50 messages/minute per phone number
6. **Audit trail** — every single auto-reply is logged with full context
7. **Stateless by default** — each message handled independently (5-min thread context only)

---

## Pricing Model

```
 One-time setup:  €400–€500
 Monthly:         €40–€50
```

Clients get:
- One voice profile trained on their actual chat history
- A configured Twilio WhatsApp Business number
- 24/7 automated customer responses in their voice
- Owner escalation via Telegram for anything complex
- 30-minute setup from chat export to live

---

## Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key
- [Twilio account](https://www.twilio.com) with WhatsApp Business API enabled
- A WhatsApp chat export (.txt or .json)

### Install

```bash
git clone git@github.com:jryahia/whatsapp-voice-clone.git
cd whatsapp-voice-clone

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env with your keys:
# OPENAI_API_KEY=sk-...
# TWILIO_ACCOUNT_SID=AC...
# TWILIO_AUTH_TOKEN=...
# TWILIO_WHATSAPP_NUMBER=+14155238886
```

### Train a Voice Profile

Export your WhatsApp chat (from WhatsApp → Settings → Chats → Export chat → Without media) and run:

```bash
python main.py train \
  --export data/exports/mario_pizzeria.txt \
  --name "Mario Pizzeria"
```

Output:
```
 Training voice profile: Mario Pizzeria
    Export: /home/user/whatsapp-voice-clone/data/exports/mario_pizzeria.txt

Yes Profile analyzed!
  Formality         informale (0.34)
  Top Words         42 unique
  Preferred Emojis  12 found
  Greetings         184 total — most common: Ciao
  Avg Message       67 chars
  Questions         23.1% of messages
  Busiest Hour      19:00
```

### Test a Prompt

```bash
python main.py test-prompt \
  --profile "Mario Pizzeria" \
  --message "Ciao Mario, siete aperti domani?"
```

### Serve the Webhook

```bash
python main.py serve --name "Mario Pizzeria" --port 8080
```

Then set your Twilio WhatsApp webhook to:
```
https://your-server.com:8080/webhook/twilio
```

### List All Profiles

```bash
python main.py list-profiles
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `train --export PATH --name NAME` | Train a voice profile from chat export |
| `serve --name NAME --port PORT` | Start FastAPI webhook server |
| `test-prompt --profile NAME --message MSG` | Test a message against a profile |
| `list-profiles` | List all stored profiles |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/webhook/twilio` | Twilio WhatsApp webhook receiver |
| GET | `/health` | Health check + active profile info |
| GET | `/profiles` | List stored voice profiles |

---

## Project Structure

```
whatsapp-voice-clone/
├── main.py               # Entry point
├── cli.py                # Typer CLI (train / serve / test-prompt / list-profiles)
├── config.py             # Pydantic config with JSON + .env overrides
├── logger.py             # Structured JSON logging (structlog)
│
├── profiler/             # Voice profiling module
│   ├── __init__.py
│   ├── analyzer.py       # Orchestrates analysis → VoiceProfile dataclass
│   ├── extractor.py      # 6 linguistic analysis functions
│   └── profile_store.py  # ChromaDB profile persistence
│
├── responder/            # AI response generation
│   ├── __init__.py
│   ├── prompt_builder.py # Dynamic GPT system prompt from voice profile
│   ├── generator.py      # OpenAI API calls with retry + rate limit handling
│   └── guardrails.py     # 7 hard business rules
│
├── server/               # Webhook server
│   ├── __init__.py
│   ├── webhook.py        # FastAPI + Twilio TwiML response
│   ├── router.py         # Intent classification (7 categories)
│   └── escalation.py     # Telegram bot escalation
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_guardrails.py
│   ├── test_prompt_builder.py
│   └── test_router.py
│
├── data/
│   ├── exports/          # Raw chat exports (gitignored)
│   └── profiles/         # Voice profile storage
│
├── config.json           # Default configuration
├── .env.example          # Environment variable template
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Deployment Options

| Option | Setup Time | Monthly Cost | Notes |
|--------|-----------|-------------|-------|
| VPS (DigitalOcean, Hetzner) | 15 min | ~€6/mo | Full control, ngrok for webhook |
| Railway | 5 min | ~€5/mo | Easy deploy, built-in HTTPS |
| Docker self-hosted | 20 min | ~€6/mo | Containerized, portable |

### Quick VPS Deploy

```bash
# On your VPS
git clone git@github.com:jryahia/whatsapp-voice-clone.git
cd whatsapp-voice-clone
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install uvicorn

# For HTTPS webhook, use ngrok or Caddy
python main.py serve --name "Mario Pizzeria" --port 8080
```

---

## Screenshots

| CLI Train Output | Webhook Dashboard |
|---|---|
| *(add screenshot)* | *(add screenshot)* |

---

## Requirements

- Python 3.10+
- [OpenAI SDK](https://pypi.org/project/openai/) ≥ 1.0
- [ChromaDB](https://www.trychroma.com/) ≥ 0.4
- [FastAPI](https://fastapi.tiangolo.com/) ≥ 0.100
- [Twilio](https://www.twilio.com/docs/libraries/python) ≥ 8.0
- [structlog](https://www.structlog.org/) ≥ 24.0
- [httpx](https://www.python-httpx.org/) ≥ 0.25
- [typer](https://typer.tiangolo.com/) ≥ 0.9
- [rich](https://rich.readthedocs.io/) ≥ 13.7

---

## License

MIT — free to use, modify, and distribute.

---

<div align="center">
Built with  by <a href="https://github.com/jryahia">Yahia</a>
</div>
