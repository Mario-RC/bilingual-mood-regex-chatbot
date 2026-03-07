"""Core engine for a bilingual regex-based mood chatbot.

This module encapsulates state, language detection, response selection, and
template processing so the command-line entrypoint stays small and maintainable.
"""

from __future__ import annotations

import datetime
import locale
import random
import re
from pathlib import Path
from urllib.request import urlretrieve
from dataclasses import dataclass
from datetime import date
from typing import Pattern

import fasttext
from dialogue_rules.pairs_reflections import pairs_en, pairs_es, reflections_en, reflections_es


USER_TEMPLATE = "USER > "
BOT_TEMPLATE = "BOT  > {0}"
MOOD_TEMPLATE = "MOOD > {0}"

DEFAULT_USER_NAME = "desconocido"
DEFAULT_USER_CITY = "desconocido"
DEFAULT_USER_COUNTRY = "desconocido"

MOOD_LABELS = ["happy", "sad", "confident", "funny", "fear", "angry"]
DEFAULT_RANDOM_SEED = 42
FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
FASTTEXT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "lid.176.ftz"
MOOD_TOKENS = {
    "happy": "\033[0;32;m:) \033[0;30;m",
    "sad": "\033[0;34;m:( \033[0;30;m",
    "confident": "\033[0;35;m;) \033[0;30;m",
    "funny": "\033[0;33;m:p \033[0;30;m",
    "fear": "\033[0;37;m:s \033[0;30;m",
    "angry": "\033[0;31;m>:( \033[0;30;m",
}


# Best-effort locale setup; failure should never break chatbot startup.
try:
    locale.setlocale(locale.LC_ALL, "es_ES.utf8")
except locale.Error:
    pass


@dataclass
class ChatbotState:
    """Mutable conversation state shared across responses."""

    mood: str = "normal"
    user_name: str = DEFAULT_USER_NAME
    user_city: str = DEFAULT_USER_CITY
    user_country: str = DEFAULT_USER_COUNTRY
    language: str = "es"


class RegexBilingualChatbot:
    """Bilingual ELIZA-style chatbot with conditional and templated responses."""

    def __init__(
        self,
        pairs_es,
        pairs_en,
        reflections_es: dict[str, str] | None = None,
        reflections_en: dict[str, str] | None = None,
    ) -> None:
        self._pairs_es = [(re.compile(pattern, re.IGNORECASE), responses) for pattern, responses in pairs_es]
        self._pairs_en = [(re.compile(pattern, re.IGNORECASE), responses) for pattern, responses in pairs_en]
        self._reflections_es = reflections_es or {}
        self._reflections_en = reflections_en or {}
        self._regex_es = self._compile_reflections(self._reflections_es)
        self._regex_en = self._compile_reflections(self._reflections_en)
        self.state = ChatbotState()
        self._random = random.Random(DEFAULT_RANDOM_SEED)
        self._lang_model = self._load_fasttext_model()

    @staticmethod
    def _compile_reflections(reflections: dict[str, str]) -> Pattern[str]:
        """Compile a regex from reflection keys, longest-first to preserve phrases."""
        if not reflections:
            # Regex that can never match, used as a safe default.
            return re.compile(r"a^", re.IGNORECASE)

        sorted_keys = sorted(reflections.keys(), key=len, reverse=True)
        return re.compile(r"\b({0})\b".format("|".join(map(re.escape, sorted_keys))), re.IGNORECASE)

    @staticmethod
    def _load_fasttext_model():
        """Load fastText language-id model, downloading it once if missing."""
        FASTTEXT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        if not FASTTEXT_MODEL_PATH.exists():
            urlretrieve(FASTTEXT_MODEL_URL, FASTTEXT_MODEL_PATH)

        return fasttext.load_model(str(FASTTEXT_MODEL_PATH))

    def _detect_language(self, message: str) -> str:
        """Detect message language using fastText; fallback to Spanish on errors."""
        if not message.strip():
            return "es"

        clean_message = " ".join(message.split())
        try:
            labels, _ = self._lang_model.predict(clean_message, k=1)
            detected = labels[0].replace("__label__", "") if labels else "es"
        except Exception:
            return "es"

        return "en" if detected == "en" else "es"

    def _current_date_time(self) -> tuple[str, str, str, str, str, str, str, str]:
        """Return formatted date/time tokens used by response templates."""
        now = datetime.datetime.now()
        return (
            f"{now.day:01d}",
            now.strftime("%A").lower(),
            f"{now.month:02d}",
            now.strftime("%B").lower(),
            f"{now.year:02d}",
            f"{now.hour:02d}",
            f"{now.minute:02d}",
            f"{now.second:02d}",
        )

    @staticmethod
    def _season_es() -> str:
        """Return season name in Spanish based on day-of-year."""
        day_of_year = date.today().timetuple().tm_yday
        spring = range(80, 172)
        summer = range(172, 264)
        fall = range(264, 355)

        if day_of_year in spring:
            return "primavera"
        if day_of_year in summer:
            return "verano"
        if day_of_year in fall:
            return "otoño"
        return "invierno"

    @staticmethod
    def _day_to_es(day_name_en: str) -> str:
        """Translate weekday names from English to Spanish."""
        mapping = {
            "Monday": "lunes",
            "Tuesday": "martes",
            "Wednesday": "miercoles",
            "Thursday": "jueves",
            "Friday": "viernes",
            "Saturday": "sabado",
            "Sunday": "domingo",
        }
        return mapping.get(day_name_en, day_name_en)

    def _set_state_value(self, variable: str, value: str) -> None:
        """Update mutable user state variables."""
        if variable == "user_name":
            self.state.user_name = value
        elif variable == "user_city":
            self.state.user_city = value
        elif variable == "user_country":
            self.state.user_country = value

    def _get_state_value(self, variable: str) -> str:
        """Resolve dynamic placeholders such as date/time and user properties."""
        day, day_name, month, month_name, year, hour, minute, second = self._current_date_time()

        if variable == "day":
            return day
        if variable == "day_name":
            return self._day_to_es(day_name)
        if variable == "month":
            return month
        if variable == "month_name":
            return month_name
        if variable == "season":
            return self._season_es()
        if variable == "year":
            return year
        if variable == "hour":
            return hour
        if variable == "minute":
            return minute
        if variable == "second":
            return second
        if variable == "user_name":
            return self.state.user_name.capitalize()
        if variable == "user_city":
            return self.state.user_city.capitalize()
        if variable == "user_country":
            return self.state.user_country.capitalize()
        if variable == "bot_mood":
            return self.state.mood
        return ""

    def _substitute_reflections(self, text: str) -> str:
        """Apply language-specific pronoun reflections to wildcard captures."""
        if self.state.language == "en":
            regex = self._regex_en
            mapping = self._reflections_en
        else:
            regex = self._regex_es
            mapping = self._reflections_es

        if not mapping:
            return text

        def replace(match: re.Match[str]) -> str:
            token = match.group(0).lower()
            return mapping.get(token, token)

        return regex.sub(replace, text.lower())

    def _apply_wildcards(self, response: str, match: re.Match[str]) -> str:
        """Replace %1/%2 placeholders using regex match groups."""
        pos = response.find("%")
        while pos >= 0 and pos + 1 < len(response):
            group_char = response[pos + 1]
            if not group_char.isdigit():
                pos = response.find("%", pos + 1)
                continue

            group_idx = int(group_char)
            group_value = self._substitute_reflections(match.group(group_idx)) if group_idx <= match.lastindex else ""
            response = response[:pos] + group_value + response[pos + 2 :]
            pos = response.find("%")
        return response

    def _process_set_get_tags(self, response: str) -> str:
        """Handle <set ...> and <get ...> directives embedded in responses."""
        tag_pattern = re.compile(r"<\s*(set|get)\s+([^>]+)>")

        while True:
            tag_match = tag_pattern.search(response)
            if not tag_match:
                break

            action = tag_match.group(1)
            payload = tag_match.group(2).strip()

            if action == "set":
                if "=" in payload:
                    variable, value = payload.split("=", 1)
                    self._set_state_value(variable.strip(), value.strip())
                replacement = ""
            else:
                replacement = self._get_state_value(payload)

            response = response[: tag_match.start()] + replacement + response[tag_match.end() :]

        # Normalize extra spaces introduced by removed <set ...> tags.
        return re.sub(r"\s{2,}", " ", response).strip()

    def _pick_conditional_response(self, responses) -> str:
        """Choose a response honoring optional conditional branches."""
        conditional_candidates = []
        regular_candidates = []

        for response in responses:
            if isinstance(response, str) and response.startswith("*"):
                match = re.match(
                    r"^\*\s*(<\s*get\s+[^>]+>)\s*(==|!=)\s*(.*?)\s*=>\s*(.+)$",
                    response,
                )
                if not match:
                    continue

                lhs_expr, operator, rhs_raw, candidate = match.groups()
                lhs_value = self._process_set_get_tags(lhs_expr).lower().strip()
                rhs_value = rhs_raw.lower().strip()

                if (operator == "==" and lhs_value == rhs_value) or (
                    operator == "!=" and lhs_value != rhs_value
                ):
                    conditional_candidates.append(candidate)
            else:
                regular_candidates.append(response)

        if conditional_candidates:
            return self._random.choice(conditional_candidates)
        if regular_candidates:
            return self._random.choice(regular_candidates)
        return ""

    def _decorate_with_mood(self, response: str) -> str:
        """Set and prepend a random chatbot mood token to the response text."""
        self.state.mood = self._random.choice(MOOD_LABELS)
        prefix = MOOD_TOKENS.get(self.state.mood, "")
        return f"{prefix}{response}"

    @staticmethod
    def _cleanup_trailing_punctuation(response: str) -> str:
        """Fix punctuation combinations produced by template composition."""
        pairs = {"..": ".", "?.": ".", "??": "?", ".?": "?"}
        if len(response) >= 2:
            ending = response[-2:]
            if ending in pairs:
                return response[:-2] + pairs[ending]
        return response

    def respond(self, message: str) -> str:
        """Generate a response for an incoming user message."""
        self.state.language = self._detect_language(message)
        pairs = self._pairs_en if self.state.language == "en" else self._pairs_es

        for pattern, responses in pairs:
            match = pattern.match(message)
            if not match:
                continue

            response = self._pick_conditional_response(responses)
            response = self._decorate_with_mood(response)
            response = self._apply_wildcards(response, match)
            response = self._process_set_get_tags(response)
            response = self._cleanup_trailing_punctuation(response)
            return response

        return "I could not understand that." if self.state.language == "en" else "No he entendido eso."

    def converse(self, quit_commands: tuple[str, ...] = ("salir", "exit")) -> None:
        """Run an interactive conversation loop until a quit command is entered."""
        intro = (
            "Hola, soy Nuria y me gusta hablar mucho ;)\n"
            "Please write to start a conversation.\n"
            "Type salir or exit to finish.\n"
        )
        print(intro + "=" * 50 + "\n")

        while True:
            try:
                user_input = input(USER_TEMPLATE)
            except EOFError:
                print("\nEnd of input detected. Bye.")
                break
            except KeyboardInterrupt:
                print("\nInterrupted by user. Bye.")
                break

            if not user_input:
                continue

            normalized = user_input.strip()
            if normalized.lower() in quit_commands:
                print(BOT_TEMPLATE.format("Hasta luego, nos vemos pronto."))
                break

            while normalized and normalized[-1] in "!.":
                normalized = normalized[:-1]

            response = self.respond(normalized)
            print(MOOD_TEMPLATE.format(self.state.mood))
            print(BOT_TEMPLATE.format(response))
            print("")


def create_default_chatbot() -> RegexBilingualChatbot:
    """Factory that builds the chatbot using bundled dialog patterns."""

    return RegexBilingualChatbot(
        pairs_es=pairs_es,
        pairs_en=pairs_en,
        reflections_es=reflections_es,
        reflections_en=reflections_en,
    )
