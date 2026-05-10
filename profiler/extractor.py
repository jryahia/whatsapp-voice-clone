"""Token-level analysis utilities for Italian WhatsApp chat exports."""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone
from itertools import pairwise
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ITALIAN_GREETINGS = [
    "Ciao",
    "Buongiorno",
    "Buonasera",
    "Salve",
    "Pronto",
    "Ehilà",
    "Ehi",
    "Ohi",
    "Senti",
    "Allora",
    "Guarda",
]

ITALIAN_STOPWORDS: set[str] = {
    "il", "lo", "la", "i", "gli", "le",
    "un", "uno", "una",
    "di", "a", "da", "in", "con", "su", "per", "tra", "fra",
    "che", "e", "o", "ma", "se",
    "come", "perché", "anche",
    "non", "si", "ci", "ne",
    "mi", "ti", "li",
    "lo", "la", "gli", "le",
    "c'è", "ci sono",
    "ha", "ho", "hai", "hanno", "abbiamo",
    "sono", "sei", "è", "siamo", "siete",
    "questo", "quella", "quello", "questi", "quelle", "quelli",
    "del", "della", "dei", "delle",
    "nel", "nella", "negli",
    "sulle", "sul", "sullo",
}

FORMAL_GREETINGS = frozenset({"Buongiorno", "Buonasera", "Salve"})
INFORMAL_GREETINGS = frozenset({"Ciao", "Ehila", "Ehilà", "Ehi", "Ohi", "Senti", "Pronto"})

FORMAL_CLOSINGS = frozenset({
    "cordiali saluti",
    "distinti saluti",
    "cordialmente",
    "distintamente",
    "saluti cordiali",
    "saluti distinti",
})

# Comprehensive Unicode emoji pattern — covers most common emoji ranges
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Misc Symbols and Pictographs
    "\U0001F680-\U0001F6FF"  # Transport and Map
    "\U0001F1E0-\U0001F1FF"  # Flags (regional indicators)
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"  # Enclosed / supplemental
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols Extended-A
    "\U00002600-\U000026FF"  # Misc symbols
    "\U0000FE00-\U0000FE0F"  # Variation selectors
    "\U0000200D"             # Zero-width joiner
    "\U00002B50"             # Star
    "\U0000231A-\U0000231B"  # Watch / Hourglass
    "\U000023E9-\U000023F3"  # Arrows / Buttons
    "\U000025AA-\U000025AB"  # Squares
    "\U000025B6"             # Play button
    "\U000025C0"             # Reverse button
    "\U000025FB-\U000025FE"  # Medium squares
    "\U00002614"             # Umbrella
    "\U00002645"             # Woman / Man
    "\U00002650-\U00002651"  # Signs
    "\U0000267E-\U0000267F"  # Signs
    "\U00002693"             # Anchor
    "\U000026A1"             # High voltage
    "\U000026AA-\U000026AB"  # Circles
    "\U000026BD-\U000026BE"  # Soccer / Baseball
    "\U000026C4-\U000026C5"  # Snowman / Sun
    "\U000026D4"             # No entry
    "\U000026EA"             # Church
    "\U000026F2-\U000026F3"  # Fountain / Golf
    "\U000026F5"             # Sailboat
    "\U000026FA"             # Tent
    "\U000026FD"             # Fuel pump
    "\U00002702"             # Scissors
    "\U00002708"             # Airplane
    "\U0000270C-\U0000270D"  # Victory / Writing
    "\U0000270F"             # Pencil
    "\U00002812-\U00002814"  # Pen tips
    "\U00002934-\U00002935"  # Arrows
    "\U00002B05-\U00002B07"  # Arrows
    "\U00003030"             # Wavy dash
    "\U0000303D"             # Part alternation mark
    "\U00003297"             # Congratulation
    "\U00003299"             # Secret
    "]",
    flags=re.UNICODE,
)

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
WORD_SPLIT = re.compile(r"\w+(?:'\w+)?", re.UNICODE)
QUOTED_WORD = re.compile(r"""(?P<word>"[A-Z][A-Za-z0-9]+"|'[A-Z][A-Za-z0-9]+')""", re.UNICODE)
ALL_CAPS_WORD = re.compile(r"\b[A-Z]{2,}\b")
MULTI_PUNCT_RE = re.compile(r"\.{3,}|!{2,}|\?{2,}|[!?]{2,}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_lines(text: str) -> list[str]:
    """Split text into individual message lines.

    Handles both plain text and timestamp-prefixed formats like:
    [14/03/25, 18:47] Nome: Messaggio
    """
    lines = text.strip().splitlines()
    # Strip timestamps if present
    cleaned: list[str] = []
    ts_pattern = re.compile(r"^\[?\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}:\d{2}(?::\d{2})?\]?\s*.*?:\s*")
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Try to remove leading timestamp + name prefix
        cleaned_line = ts_pattern.sub("", stripped, count=1)
        if not cleaned_line:
            cleaned_line = stripped
        cleaned.append(cleaned_line)
    return cleaned


def _count_words(text: str) -> int:
    return len(WORD_SPLIT.findall(text))


def _lower(text: str) -> str:
    return text.lower()


# ---------------------------------------------------------------------------
# Public extraction functions
# ---------------------------------------------------------------------------


def extract_greeting_patterns(text: str) -> dict:
    """Find opening lines starting with Italian greetings.

    Returns a dict mapping greeting -> count for each detected greeting.
    Only counts the first word(s) of each line when they match a greeting.
    """
    lines = _split_lines(text)
    greeting_counts: dict[str, int] = {}

    for line in lines:
        trimmed = line.strip()
        if not trimmed:
            continue
        # Check multi-word greetings first, then single-word
        lower_line = trimmed.lower()
        for greeting in ITALIAN_GREETINGS:
            if lower_line.startswith(greeting.lower()) and (
                len(trimmed) == len(greeting)
                or not trimmed[len(greeting)].isalpha()
            ):
                greeting_counts[greeting] = greeting_counts.get(greeting, 0) + 1
                break

    return greeting_counts


def extract_emoji_stats(text: str) -> dict:
    """Count emoji frequency, total count, per-100-words, and placement patterns."""
    emojis = EMOJI_PATTERN.findall(text)
    total_emoji = len(emojis)
    word_count = _count_words(text)
    per_100_words = round((total_emoji / word_count) * 100, 2) if word_count > 0 else 0.0

    # Emoji frequency
    freq = Counter(emojis)

    # Placement: examine sentences for emoji position
    sentences = SENTENCE_SPLIT.split(text) if SENTENCE_SPLIT.split(text) else [text]
    start_count = 0
    end_count = 0
    mid_count = 0
    emoji_sentences = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        emoji_positions = [m.start() for m in EMOJI_PATTERN.finditer(sentence)]
        if not emoji_positions:
            continue
        emoji_sentences += 1
        # First emoji at start
        first_pos = min(emoji_positions)
        last_pos = max(emoji_positions)
        sen_len = len(sentence)
        # Start: first 20% of sentence
        if first_pos / sen_len < 0.2:
            start_count += 1
        # End: last 20% of sentence
        if last_pos / sen_len > 0.8:
            end_count += 1
        if first_pos / sen_len >= 0.2 and last_pos / sen_len <= 0.8:
            mid_count += 1

    total_placed = start_count + end_count + mid_count or 1
    placement = {
        "start_pct": round(start_count / total_placed * 100, 1),
        "end_pct": round(end_count / total_placed * 100, 1),
        "mid_pct": round(mid_count / total_placed * 100, 1),
    }

    # Preferred: top 5 most frequent emoji
    preferred = [e for e, _ in freq.most_common(5)]

    return {
        "preferred": preferred,
        "total_emoji": total_emoji,
        "per_100_words": per_100_words,
        "frequency": dict(freq.most_common(20)),
        "placement": placement,
    }


def compute_formality_score(text: str) -> float:
    """Compute a formality score from 0.0 (very informal) to 1.0 (very formal).

    Analyzes:
    - Use of 'Lei' (formal you) vs 'tu' (informal you)
    - Formal vs informal greetings
    - Sentence complexity (avg words per sentence)
    - Presence of formal closings
    """
    if not text or not text.strip():
        return 0.5  # neutral default

    lower = _lower(text)
    words_lower = set(WORD_SPLIT.findall(lower))

    # 1. Pronoun analysis: Lei (formal) vs tu (informal)
    lei_count = len(re.findall(r"\bLei\b", text)) + len(re.findall(r"\bLe\b", text))
    # "lei" (lowercase) could also be the formal "Lei" in non-capitalized messages
    # We count it conservatively — only after start/period/exclamation/question
    lei_count += len(re.findall(r"(?:^|[.!?]\s)lei\b", lower))
    tu_count = len(re.findall(r"\btu\b", lower))
    # Count "ti" as informal too
    ti_count = len(re.findall(r"\bti\b", lower))

    total_pronoun = lei_count + tu_count + ti_count

    # 2. Greeting analysis
    greetings = extract_greeting_patterns(text)
    formal_greeting_count = sum(
        count for g, count in greetings.items() if g in FORMAL_GREETINGS
    )
    informal_greeting_count = sum(
        count for g, count in greetings.items() if g in INFORMAL_GREETINGS
    )

    # 3. Sentence complexity
    sentences = [
        s.strip()
        for s in SENTENCE_SPLIT.split(text)
        if s.strip()
    ]
    total_words_in_sentences = sum(len(WORD_SPLIT.findall(s)) for s in sentences)
    avg_words_per_sentence = (
        total_words_in_sentences / len(sentences) if sentences else 0
    )

    # 4. Formal closings
    formal_closing_count = sum(
        count for pattern in FORMAL_CLOSINGS
        for count in [len(re.findall(re.escape(pattern), lower))]
        if count > 0
    )

    # --- Scoring: each dimension contributes 0-1 ---

    # A) Pronoun formality (weight: 0.35)
    if total_pronoun > 0:
        pronoun_score = lei_count / (lei_count + tu_count + ti_count)
    else:
        pronoun_score = 0.5  # neutral

    # B) Greeting formality (weight: 0.25)
    total_greetings = formal_greeting_count + informal_greeting_count
    if total_greetings > 0:
        greeting_score = formal_greeting_count / total_greetings
    else:
        greeting_score = 0.5

    # C) Sentence complexity (weight: 0.25)
    # Longer sentences = more formal. Typical range: 8-25 words/sentence.
    complexity_norm = min(avg_words_per_sentence / 25.0, 1.0)
    complexity_score = complexity_norm
    # Short sentences can also be formal, so we penalize extreme brevity
    if avg_words_per_sentence < 5:
        complexity_score = 0.1

    # D) Formal closings (weight: 0.15)
    total_sentences = max(len(sentences), 1)
    closing_score = min(formal_closing_count / max(total_sentences * 0.05, 1), 1.0)

    score = (
        pronoun_score * 0.35
        + greeting_score * 0.25
        + complexity_score * 0.25
        + closing_score * 0.15
    )

    return round(max(0.0, min(1.0, score)), 3)


def compute_vocabulary(text: str) -> dict:
    """Return top 50 filtered words and extract jargon/custom words."""
    if not text or not text.strip():
        return {"top_words": [], "jargon": []}

    words = WORD_SPLIT.findall(text.lower())
    # Filter stopwords and short tokens
    filtered = [w for w in words if w not in ITALIAN_STOPWORDS and len(w) > 1]
    word_counts = Counter(filtered)
    top_50 = word_counts.most_common(50)
    top_words = [{"word": w, "count": c} for w, c in top_50]

    # Jargon: ALL CAPS words or quoted words appearing >1 time
    raw_words = WORD_SPLIT.findall(text)  # preserve original case
    cap_words = [w for w in raw_words if len(w) >= 2 and w.isupper() and not w.isdigit()]
    quoted_words_raw = QUOTED_WORD.findall(text)
    # Strip quote marks
    quoted_words = [w.strip("\"'") for w in quoted_words_raw]

    all_special = cap_words + quoted_words
    special_counts = Counter(all_special)
    jargon = [w for w, c in special_counts.most_common() if c > 1 and w.lower() not in ITALIAN_STOPWORDS]

    return {"top_words": top_words, "jargon": jargon}


def compute_response_style(text: str) -> dict:
    """Analyze message-level response style.

    Returns:
        avg_message_length_chars, avg_message_length_words,
        question_frequency, exclamation_frequency,
        caps_tendency, punctuation_habits
    """
    lines = _split_lines(text)
    if not lines:
        return {
            "avg_length_chars": 0.0,
            "avg_length_words": 0.0,
            "question_pct": 0.0,
            "exclamation_pct": 0.0,
            "caps_pct": 0.0,
            "punctuation_habits": [],
        }

    total_chars = sum(len(l) for l in lines)
    total_words_total = sum(_count_words(l) for l in lines)
    num_lines = len(lines)

    questions = sum(1 for l in lines if l.strip().endswith("?"))
    exclamations = sum(1 for l in lines if l.strip().endswith("!"))

    # Caps tendency: % of all-capital words across all lines
    all_words = WORD_SPLIT.findall(text)
    all_caps_words = ALL_CAPS_WORD.findall(text)
    caps_pct = round(len(all_caps_words) / max(len(all_words), 1) * 100, 1)

    # Punctuation habits: detect repeated/multiple punctuation
    habits: list[str] = []
    all_text = " ".join(lines)
    multi_matches = MULTI_PUNCT_RE.findall(all_text)
    punct_counts = Counter(multi_matches)
    for punct, count in punct_counts.items():
        if punct == "..." and count >= 3:
            habits.append("ellipsis")
        elif "!" in punct and len(punct) >= 2 and count >= 2:
            habits.append("multi_exclamation")
        elif "?" in punct and len(punct) >= 2 and count >= 2:
            habits.append("multi_question")
    if "??" in all_text and "???" not in all_text:
        if "multi_question" not in habits:
            habits.append("multi_question")
    if "!!" in all_text and "!!!" not in all_text:
        if "multi_exclamation" not in habits:
            habits.append("multi_exclamation")

    return {
        "avg_length_chars": round(total_chars / num_lines, 1),
        "avg_length_words": round(total_words_total / num_lines, 1),
        "question_pct": round(questions / num_lines * 100, 1),
        "exclamation_pct": round(exclamations / num_lines * 100, 1),
        "caps_pct": caps_pct,
        "punctuation_habits": habits,
    }


def compute_time_patterns(messages: list[dict]) -> dict:
    """Analyze temporal patterns from a list of message dicts.

    Each message dict must have:
        - 'timestamp': ISO-format datetime string
        - 'direction': 'sent' or 'received'

    Returns:
        - active_hours: list of hours (0-23) sorted by activity (most active first)
        - avg_response_time_seconds: average time between received -> sent messages
        - busiest_hour: the hour with the most messages
    """
    if not messages:
        return {
            "active_hours": [],
            "avg_response_time_s": 0.0,
            "busiest_hour": 0,
        }

    # Parse timestamps
    parsed: list[tuple[datetime, str]] = []
    hour_counts: Counter[int] = Counter()

    for msg in messages:
        ts = msg.get("timestamp")
        direction = msg.get("direction", "sent")
        if not ts:
            continue
        if isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts)
            except (ValueError, TypeError):
                logger.warning("unparseable_timestamp", timestamp=ts)
                continue
        elif isinstance(ts, (int, float)):
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        else:
            continue
        parsed.append((dt, direction))
        hour_counts[dt.hour] += 1

    # Active hours sorted by activity (descending)
    active_hours = [h for h, _ in hour_counts.most_common()]

    # Busiest hour
    busiest_hour = hour_counts.most_common(1)[0][0] if hour_counts else 0

    # Average response time: for each received message, find the next sent message
    if len(parsed) < 2:
        return {
            "active_hours": active_hours,
            "avg_response_time_s": 0.0,
            "busiest_hour": busiest_hour,
        }

    parsed_sorted = sorted(parsed, key=lambda x: x[0])
    response_times: list[float] = []

    for i, (dt, direction) in enumerate(parsed_sorted):
        if direction != "received":
            continue
        # Look for next sent message
        for j in range(i + 1, len(parsed_sorted)):
            if parsed_sorted[j][1] == "sent":
                diff = (parsed_sorted[j][0] - dt).total_seconds()
                if 0 < diff < 86400:  # cap at 24h to filter out overnight gaps
                    response_times.append(diff)
                break

    avg_response = round(sum(response_times) / max(len(response_times), 1), 1)

    return {
        "active_hours": active_hours,
        "avg_response_time_s": avg_response,
        "busiest_hour": busiest_hour,
    }
