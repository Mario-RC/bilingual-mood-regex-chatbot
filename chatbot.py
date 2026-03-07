"""Command-line entrypoint for the bilingual mood regex chatbot."""

from __future__ import annotations

from chatbot_engine import create_default_chatbot


def main() -> None:
    """Start the interactive conversation loop."""
    chatbot = create_default_chatbot()
    chatbot.converse()


if __name__ == "__main__":
    main()
