
# Bilingual Mood Regex Chatbot

Rule-based bilingual (Spanish/English) chatbot inspired by
[ELIZA](https://en.wikipedia.org/wiki/ELIZA), with regex dialogue rules,
offline fastText language detection, user memory slots, and random mood decoration.

## Features

- Spanish and English response rules.
- Regex wildcard captures (`%1`, `%2`, ...).
- Reflection dictionaries for pronoun substitution.
- Dynamic placeholders: date/time, season, and user profile fields.
- Conditional responses using chatbot/user state.
- Colored mood prefix for each response.

## Project Structure

- `chatbot.py`: CLI entrypoint.
- `chatbot_engine.py`: core chatbot engine and state management.
- `dialogue_rules/pairs_reflections.py`: bilingual pattern/response rules.
- `requirements.txt`: runtime dependencies.

Legacy note: `bot.py` was removed and replaced by `chatbot.py` as the main executable.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first execution downloads the fastText language-id model (`lid.176.ftz`) to
`models/` automatically (one-time step).

## Usage

Run the chatbot interactively:

```bash
python3 chatbot.py
```

Exit commands in chat:

- `salir`
- `exit`

## Rule Format Notes

The response engine supports custom directives inside response templates:

- `<set user_name=...>` to store user context.
- `<get user_name>` to retrieve stored values.
- Conditional entries:
	`* <get bot_mood> == angry => ...`

This allows responses to adapt to both user memory and chatbot mood state.
