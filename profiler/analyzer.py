"""High-level profile analysis for WhatsApp chat exports.

Provides the VoiceProfile dataclass and analyze_chat function that
orchestrates all extraction utilities.
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import structlog

from .extractor import (
    compute_formality_score,
    compute_response_style,
    compute_time_patterns,
    compute_vocabulary,
    extract_emoji_stats,
    extract_greeting_patterns,
)

logger = structlog.get_logger(__name__)

# Italian stopwords base for language detection
_DETECTION_STOPWORDS: set[str] = {
    "il", "lo", "la", "i", "gli", "le",
    "di", "a", "da", "in", "con", "su",
    "che", "e", "o", "ma", "se", "per",
    "non", "si", "ci", "ne",
    "ha", "ho", "hai", "hanno", "sono",
    "è", "un", "una", "uno",
}

TIMESTAMP_PATTERN = re.compile(
    r"^\[?\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}:\d{2}(?::\d{2})?\]?"
)


# ---------------------------------------------------------------------------
# Domain dataclass
# ---------------------------------------------------------------------------


@dataclass
class VoiceProfile:
    """Complete voice profile extracted from a WhatsApp chat export."""

    name: str
    formality: dict
    vocabulary: dict
    emoji: dict
    greetings: dict
    response_style: dict
    time_patterns: dict
    raw_sample: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> VoiceProfile:
        data = json.loads(json_str)
        return cls(**data)


# ---------------------------------------------------------------------------
# Detection & parsing helpers
# ---------------------------------------------------------------------------


def _detect_is_italian(text: str, threshold: float = 0.02) -> bool:
    """Rough language detection: check if Italian stopword ratio is above threshold."""
    if not text or not text.strip():
        return True  # assume Italian on empty
    words = re.findall(r"\w+(?:'\w+)?", text.lower())
    if not words:
        return True
    stopword_hits = sum(1 for w in words if w in _DETECTION_STOPWORDS)
    ratio = stopword_hits / len(words)
    return ratio >= threshold


def _parse_json_export(path: Path) -> tuple[str, list[dict], bool]:
    """Parse a JSON WhatsApp export file.

    Returns (full_text, messages_list, has_messages).
    Handles two formats:
      1. A list of dicts with 'text', 'timestamp', 'direction' keys.
      2. Any other JSON structure — try to extract text fields.
    """
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)

    if isinstance(data, list):
        messages: list[dict[str, Any]] = []
        text_parts: list[str] = []
        for item in data:
            if isinstance(item, dict):
                text = item.get("text") or item.get("message") or item.get("content", "")
                if isinstance(text, str):
                    text_parts.append(text)
                ts = item.get("timestamp") or item.get("date") or item.get("time", "")
                direction = item.get("direction") or item.get("type", "sent")
                # Normalize direction
                if direction in ("incoming", "received", "in"):
                    direction = "received"
                elif direction in ("outgoing", "sent", "out"):
                    direction = "sent"
                messages.append({
                    "text": str(text),
                    "timestamp": str(ts) if ts else "",
                    "direction": direction,
                })
        return "\n".join(text_parts), messages, len(text_parts) > 0

    # Single dict — flatten relevant string fields
    text_parts = []
    for val in data.values():
        if isinstance(val, str) and len(val) > 20:
            text_parts.append(val)

    return "\n".join(text_parts), [], len(text_parts) > 0


def _parse_txt_export(path: Path) -> tuple[str, list[dict], bool]:
    """Parse a plain-text WhatsApp export.

    Lines may optionally start with a timestamp like:
        [14/03/25, 18:47] Nome Cognome: Message
        [14/03/25, 18:47] Nome: Message
        14/03/25, 18:47 - Nome: Message

    Returns (full_text, messages_list, has_messages).
    """
    raw = path.read_text(encoding="utf-8")
    lines = raw.strip().splitlines()

    # Pattern to extract timestamp + name from line prefix
    # Matches: [DD/MM/YY, HH:MM] Name: or [DD/MM/YY, HH:MM:SS] Name:
    msg_pattern = re.compile(
        r"^\[?\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}:\d{2}(?::\d{2})?\]?\s+(.*?):\s+(.+)"
    )
    # Also match with dash separator: 14/03/25, 18:47 - Name: Message
    msg_pattern2 = re.compile(
        r"^\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}:\d{2}(?::\d{2})?\s*-\s+(.*?):\s+(.+)"
    )

    texts: list[str] = []
    messages: list[dict] = []
    direction = "sent"  # default, since we can't distinguish direction from plain text

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Try structured parse
        match = msg_pattern.match(line) or msg_pattern2.match(line)
        if match:
            # name = match.group(1)
            text = match.group(2).strip()
            texts.append(text)
            messages.append({
                "text": text,
                "timestamp": "",
                "direction": direction,
            })
        else:
            # Plain line without timestamp prefix
            texts.append(line)
            messages.append({
                "text": line,
                "timestamp": "",
                "direction": direction,
            })

    return "\n".join(texts), messages, len(texts) > 0


# ---------------------------------------------------------------------------
# Formality label helpers
# ---------------------------------------------------------------------------


def _formality_label(score: float) -> str:
    if score >= 0.65:
        return "formale"
    elif score >= 0.35:
        return "informale"
    return "molto informale"


def _count_pronoun(text: str, pronoun: str) -> float:
    """Count occurrences of a pronoun (case-sensitive for Lei, lowercase for tu)."""
    lower = text.lower()
    # Lei (formal): count capitalized versions
    lei_count = 0
    if pronoun == "Lei":
        lei_count += len(re.findall(r"\bLei\b", text))
        lei_count += len(re.findall(r"(?:^|[.!?]\s)lei\b", lower))
        return lei_count
    # tu: always lowercase
    return len(re.findall(r"\btu\b", lower)) + len(re.findall(r"\bti\b", lower))


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------


def analyze_chat(export_path: str | Path) -> VoiceProfile:
    """Analyze a WhatsApp chat export and produce a VoiceProfile.

    Supports:
    - Plain .txt exports (one message per line, optional [DD/MM/YY, HH:MM] prefix)
    - .json exports (list of dicts with 'text', 'timestamp', 'direction' keys)

    Parameters
    ----------
    export_path:
        Path to the export file.

    Returns
    -------
    VoiceProfile with all extracted features.
    """
    path = Path(export_path).expanduser().resolve()
    logger.info("analyzing_chat_export", path=str(path))

    if not path.exists():
        logger.warning("export_file_not_found", path=str(path))
        return VoiceProfile(
            name=path.stem,
            formality={"score": 0.5, "label": "informale", "lei_usage_pct": 0.0, "tu_usage_pct": 0.0},
            vocabulary={"top_words": [], "jargon": []},
            emoji={"preferred": [], "per_100_words": 0.0, "placement": {"start_pct": 0.0, "end_pct": 0.0, "mid_pct": 0.0}},
            greetings={"patterns": {}, "total_greetings": 0, "most_common": ""},
            response_style={
                "avg_length_chars": 0.0, "avg_length_words": 0.0,
                "question_pct": 0.0, "exclamation_pct": 0.0,
                "caps_pct": 0.0, "punctuation_habits": [],
            },
            time_patterns={"active_hours": [], "avg_response_time_s": 0.0, "busiest_hour": 0},
            raw_sample="",
        )

    # Read raw content
    raw_bytes = path.read_bytes()
    raw_text = raw_bytes.decode("utf-8", errors="replace")

    # Save raw sample (first 2000 chars)
    raw_sample = raw_text[:2000]

    # Check for empty file
    if not raw_text.strip():
        logger.info("empty_export_file", path=str(path))
        return VoiceProfile(
            name=path.stem,
            formality={"score": 0.5, "label": "informale", "lei_usage_pct": 0.0, "tu_usage_pct": 0.0},
            vocabulary={"top_words": [], "jargon": []},
            emoji={"preferred": [], "per_100_words": 0.0, "placement": {"start_pct": 0.0, "end_pct": 0.0, "mid_pct": 0.0}},
            greetings={"patterns": {}, "total_greetings": 0, "most_common": ""},
            response_style={
                "avg_length_chars": 0.0, "avg_length_words": 0.0,
                "question_pct": 0.0, "exclamation_pct": 0.0,
                "caps_pct": 0.0, "punctuation_habits": [],
            },
            time_patterns={"active_hours": [], "avg_response_time_s": 0.0, "busiest_hour": 0},
            raw_sample=raw_sample,
        )

    # Detect language
    is_italian = _detect_is_italian(raw_text)
    if not is_italian:
        logger.warning("non_italian_text_detected", path=str(path))

    # Parse based on file extension
    suffix = path.suffix.lower()
    if suffix == ".json":
        full_text, messages, has_content = _parse_json_export(path)
    else:
        full_text, messages, has_content = _parse_txt_export(path)

    if not has_content:
        logger.warning("no_content_found_in_export", path=str(path))
        return VoiceProfile(
            name=path.stem,
            formality={"score": 0.5, "label": "informale", "lei_usage_pct": 0.0, "tu_usage_pct": 0.0},
            vocabulary={"top_words": [], "jargon": []},
            emoji={"preferred": [], "per_100_words": 0.0, "placement": {"start_pct": 0.0, "end_pct": 0.0, "mid_pct": 0.0}},
            greetings={"patterns": {}, "total_greetings": 0, "most_common": ""},
            response_style={
                "avg_length_chars": 0.0, "avg_length_words": 0.0,
                "question_pct": 0.0, "exclamation_pct": 0.0,
                "caps_pct": 0.0, "punctuation_habits": [],
            },
            time_patterns={"active_hours": [], "avg_response_time_s": 0.0, "busiest_hour": 0},
            raw_sample=raw_sample,
        )

    # --- Run all extractors ---

    # Greetings
    greeting_patterns = extract_greeting_patterns(full_text)
    total_greetings = sum(greeting_patterns.values())
    most_common_greeting = (
        max(greeting_patterns, key=greeting_patterns.get)
        if greeting_patterns
        else ""
    )

    # Formality
    formality_score = compute_formality_score(full_text)
    lei_usage = _count_pronoun(full_text, "Lei")
    tu_usage = _count_pronoun(full_text, "tu")
    pronoun_total = lei_usage + tu_usage or 1
    lei_pct = round(lei_usage / pronoun_total * 100, 1)
    tu_pct = round(tu_usage / pronoun_total * 100, 1)

    # Emoji
    emoji_stats = extract_emoji_stats(full_text)

    # Vocabulary
    vocab = compute_vocabulary(full_text)

    # Response style
    style = compute_response_style(full_text)

    # Time patterns
    time_pats = compute_time_patterns(messages)

    # Assemble profile
    profile = VoiceProfile(
        name=path.stem,
        formality={
            "score": formality_score,
            "label": _formality_label(formality_score),
            "lei_usage_pct": lei_pct,
            "tu_usage_pct": tu_pct,
        },
        vocabulary=vocab,
        emoji={
            "preferred": emoji_stats["preferred"],
            "per_100_words": emoji_stats["per_100_words"],
            "placement": emoji_stats["placement"],
        },
        greetings={
            "patterns": greeting_patterns,
            "total_greetings": total_greetings,
            "most_common": most_common_greeting,
        },
        response_style=style,
        time_patterns=time_pats,
        raw_sample=raw_sample,
    )

    logger.info(
        "profile_created",
        name=profile.name,
        formality_label=profile.formality["label"],
        total_greetings=total_greetings,
        busiest_hour=time_pats.get("busiest_hour"),
    )

    return profile
