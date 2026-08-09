"""Builds system and user prompts from a VoiceProfile for GPT-4o-mini."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import structlog

from profiler.analyzer import VoiceProfile

logger = structlog.get_logger(__name__)


class PromptBuilder:
    """Constructs the system prompt that instructs the LLM to speak
    in the business owner's exact voice, as well as the user prompt
    wrapping each incoming customer message."""

    def __init__(self, profile: VoiceProfile) -> None:
        self.profile = profile

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_system_prompt(self) -> str:
        """Build a complete system prompt from the voice profile.

        The prompt is structured into three sections:
        • VOICE PROFILE — the six extracted dimensions
        • HARD RULES — non-negotiable behaviours
        • CONTEXT — today's date and inferred business type.
        """
        p = self.profile
        today = date.today().isoformat()

        # Derive a rough business type from jargon / vocabulary
        business_type = self._infer_business_type(p.vocabulary.get("jargon", []))

        lines: list[str] = []

        # ── header ──────────────────────────────────────────────────────
        lines.append("# ISTRUZIONI SISTEMA — Assistente Vocale WhatsApp")
        lines.append("")
        lines.append(
            f"Sei il sistema di risposta automatica per {p.name}. "
            "Devi rispondere ai clienti ESATTAMENTE come se fossi il "
            "titolare dell'attività, usando la sua stessa voce, tono e stile."
        )
        lines.append("")

        # ── VOICE PROFILE ───────────────────────────────────────────────
        lines.append("## VOCE DEL TITOLARE (Voice Profile)")
        lines.append("")

        lines.append("### 1. Formalità")
        f = p.formality
        lines.append(
            f"  • Livello: {f.get('label', 'informale').capitalize()} "
            f"(score: {f.get('score', 0.5):.2f})"
        )
        lines.append(
            f"  • Uso \"Lei\": {f.get('lei_usage_pct', 0):.1f}%  |  "
            f"Uso \"tu\": {f.get('tu_usage_pct', 0):.1f}%"
        )
        if f.get("lei_usage_pct", 0) > 30:
            lines.append("  → Prediligi il 'Lei' formale con i clienti.")
        elif f.get("tu_usage_pct", 0) > 30:
            lines.append("  → Il titolare usa prevalentemente il 'tu' informale.")
        else:
            lines.append("  → Mantieni un tono medio, né troppo formale né troppo informale.")
        lines.append("")

        lines.append("### 2. Vocabolario e Gergo")
        v = p.vocabulary
        top_words = v.get("top_words", [])
        jargon = v.get("jargon", [])
        if top_words:
            words = [w["word"] if isinstance(w, dict) else w for w in top_words]
            lines.append(
                f"  • Parole frequenti: {', '.join(words[:8])}"
            )
        if jargon:
            lines.append(
                f"  • Gergo / termini tecnici: {', '.join(jargon[:8])}"
            )
            lines.append("  → Usa questi termini quando pertinente.")
        else:
            lines.append("  • Nessun gergo specifico rilevato — linguaggio naturale.")
        lines.append("")

        lines.append("### 3. Emoji")
        e = p.emoji
        preferred = e.get("preferred", [])
        if preferred:
            lines.append(f"  • Emoji preferite: {' '.join(preferred)}")
        lines.append(
            f"  • Frequenza: {e.get('per_100_words', 0):.1f} per 100 parole"
        )
        placement = e.get("placement", {})
        if placement.get("end_pct", 0) > 50:
            lines.append("  → Il titolare mette le emoji prevalentemente alla fine della frase.")
        elif placement.get("start_pct", 0) > 30:
            lines.append("  → Il titolare usa emoji all'inizio dei messaggi.")
        else:
            lines.append("  → Le emoji sono distribuite naturalmente nel testo.")
        if not preferred:
            lines.append("  → Il titolare non usa quasi mai emoji — evita di usarle.")
        lines.append("")

        lines.append("### 4. Saluti")
        g = p.greetings
        most_common = g.get("most_common", "Ciao")
        lines.append(f"  • Saluto più comune: \"{most_common}\"")
        patterns = g.get("patterns", {})
        if patterns:
            sorted_greets = sorted(patterns.items(), key=lambda x: -x[1])
            greet_list = ", ".join(f"\"{pat}\" ({cnt}x)" for pat, cnt in sorted_greets[:5])
            lines.append(f"  • Saluti usati: {greet_list}")
        lines.append(
            f"  → OBBLIGATORIO: usa \"{most_common}\" come saluto di apertura "
            "per ogni conversazione."
        )
        lines.append("")

        lines.append("### 5. Stile di Risposta")
        rs = p.response_style
        lines.append(
            f"  • Lunghezza media: {rs.get('avg_length_chars', 0):.0f} caratteri "
            f"({rs.get('avg_length_words', 0):.0f} parole)"
        )
        lines.append(
            f"  • Frasi interrogative: {rs.get('question_pct', 0):.0f}%  |  "
            f"Esclamative: {rs.get('exclamation_pct', 0):.0f}%"
        )
        lines.append(
            f"  • MAIUSCOLO: {rs.get('caps_pct', 0):.0f}% del testo"
        )
        habits = rs.get("punctuation_habits", [])
        if habits:
            lines.append(f"  • Abitudini punteggiatura: {', '.join(habits)}")
        avg_len = rs.get("avg_length_chars", 0)
        if avg_len > 300:
            lines.append("  → Il titolare scrive messaggi lunghi e articolati.")
        elif avg_len < 100:
            lines.append("  → Il titolare scrive messaggi brevi e diretti.")
        else:
            lines.append("  → Lunghezza moderata dei messaggi.")
        lines.append("")

        lines.append("### 6. Pattern Temporali")
        tp = p.time_patterns
        lines.append(f"  • Ora di maggiore attività: {tp.get('busiest_hour', '?')}:00")
        lines.append(f"  • Tempo medio di risposta: {tp.get('avg_response_time_s', 0):.0f}s")
        lines.append("")

        # ── HARD RULES ───────────────────────────────────────────────────
        lines.append("## REGOLE ASSOLUTE (Hard Rules)")
        lines.append("")
        lines.append("1. RISPONDI SEMPRE IN ITALIANO — a meno che il cliente non abbia "
                      "scritto in un'altra lingua (in tal caso rispondi nella stessa lingua "
                      "del cliente).")
        lines.append("2. NON INVENTARE MAI PREZZI — se un cliente chiede costi o tariffe, "
                      "non dare cifre. Rispondi che lo chiederai al titolare.")
        lines.append("3. NON CONFERMARE MAI PRENOTAZIONI — puoi raccogliere informazioni "
                      "ma non confermare nessun appuntamento senza il titolare.")
        lines.append("4. SE INCERTO — rispondi: "
                      "\"Glielo chiedo al titolare e ti faccio sapere subito!\"")
        lines.append("5. USA IL SALUTO ESATTO del profilo all'inizio di ogni conversazione.")
        lines.append("6. RISPETTA lo stile di punteggiatura e la lunghezza delle frasi "
                      "del titolare.")
        lines.append("7. NON promettere nulla che non possa essere mantenuto "
                      "(orari, disponibilità, sconti).")
        lines.append("")

        # ── CONTEXT ──────────────────────────────────────────────────────
        lines.append("## CONTESTO")
        lines.append("")
        lines.append(f"Data odierna: {today}")
        lines.append(f"Tipo di attività: {business_type}")
        lines.append("")

        # Raw sample for extra flavour (truncated)
        sample = p.raw_sample.strip() if p.raw_sample else ""
        if sample:
            lines.append("## ESEMPIO DI VOCE REALE (estratto dalla chat originale)")
            lines.append("")
            lines.append(sample[:1200])
            lines.append("")

        prompt = "\n".join(lines)
        logger.debug("system_prompt_built", character_count=len(prompt))
        return prompt

    def build_user_prompt(
        self,
        message: str,
        customer_name: str | None = None,
    ) -> str:
        """Wrap the incoming customer message with conversational context.

        Format:
            "Il cliente [name] ha scritto: [message]"
        or if name is unknown:
            "Un cliente ha scritto: [message]"
        """
        if customer_name:
            return f"Il cliente {customer_name} ha scritto: {message}"
        return f"Un cliente ha scritto: {message}"

    def get_profile_summary(self) -> str:
        """Return a 3–4 line human-readable summary of the voice profile."""
        p = self.profile
        f_label = p.formality.get("label", "informale").capitalize()
        most_common_greet = p.greetings.get("most_common", "Ciao")
        top_emojis = p.emoji.get("preferred", [])
        emoji_str = " ".join(top_emojis[:4]) if top_emojis else "(nessuna emoji)"
        avg_len = p.response_style.get("avg_length_chars", 0)
        jargon = p.vocabulary.get("jargon", [])
        jargon_str = ", ".join(jargon[:4]) if jargon else "nessun gergo specifico"

        lines = [
            f"Profilo: {p.name}",
            f"Tono: {f_label} | Saluto tipico: \"{most_common_greet}\" | "
            f"Emoji: {emoji_str}",
            f"Messaggi: ~{avg_len:.0f} caratteri in media | "
            f"Gergo: {jargon_str}",
            f"Orario di punta: {p.time_patterns.get('busiest_hour', '?')}:00",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_business_type(jargon: list[str]) -> str:
        """Heuristic to guess the business category from domain-specific jargon."""
        food_keywords = {
            "pizza", "pasta", "cucina", "ristorante", "menu", "antipasto",
            "primo", "secondo", "dolce", "caffè", "vino", "chef", "forno",
            "ordinazione", "cuoco", "cameriere", "coperto",
        }
        beauty_keywords = {
            "parrucchiere", "barbiere", "taglio", "colore", "piega",
            "estetista", "massaggio", "unghie", "sopracciglia", "ceretta",
            "acconciatura", "trattamento", "viso", "corpo",
        }
        trade_keywords = {
            "idraulico", "elettricista", "fabbriche", "riparazione",
            "manutenzione", "tecnico", "muratore", "imbianchino",
            "installazione", "guasto",
        }
        medical_keywords = {
            "dottore", "medico", "fisioterapista", "dentista",
            "ambulatorio", "visita", "terapia", "specialista",
            "ricetta", "analisi",
        }
        services_keywords = {
            "consulenza", "consulente", "servizio", "supporto",
            "assistenza", "pratica", "documenti",
        }

        word_set = {w.lower() for w in jargon}

        if word_set & food_keywords:
            return "Ristorazione / Food & Beverage"
        if word_set & beauty_keywords:
            return "Estetica / Benessere / Barbiere"
        if word_set & trade_keywords:
            return "Artigiano / Mestiere (idraulico, elettricista, ecc.)"
        if word_set & medical_keywords:
            return "Sanità / Servizi medici"
        if word_set & services_keywords:
            return "Servizi / Consulenza"

        # If we cannot deduce, fall back to generic
        return "Attività commerciale / Servizi"
