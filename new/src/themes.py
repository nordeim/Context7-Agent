# File: src/themes.py
from __future__ import annotations
from rich.theme import Theme

CYBERPUNK = Theme(
    {
        "banner": "bold magenta",
        "border": "magenta",
        "user": "bold bright_white",
        "assistant": "bright_magenta",
        "sys": "bright_cyan",
    }
)

OCEAN = Theme(
    {
        "banner": "bold blue",
        "border": "blue",
        "user": "bright_white",
        "assistant": "bright_cyan",
        "sys": "cyan",
    }
)

FOREST = Theme(
    {
        "banner": "bold green",
        "border": "green",
        "user": "bright_white",
        "assistant": "green",
        "sys": "yellow",
    }
)

SUNSET = Theme(
    {
        "banner": "bold red",
        "border": "red",
        "user": "bright_white",
        "assistant": "bright_red",
        "sys": "orange1",
    }
)

THEMES = {
    "cyberpunk": CYBERPUNK,
    "ocean": OCEAN,
    "forest": FOREST,
    "sunset": SUNSET,
}
