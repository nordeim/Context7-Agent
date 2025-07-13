# File: src/themes.py
from __future__ import annotations
from rich.theme import Theme

CYBERPUNK = Theme({
    "primary": "bold magenta",
    "secondary": "bright_cyan",
    "user": "bold bright_white",
    "assistant": "bright_magenta",
    "border": "magenta",
})

OCEAN = Theme({
    "primary": "bold blue",
    "secondary": "cyan",
    "user": "bright_white",
    "assistant": "bright_cyan",
    "border": "blue",
})

FOREST = Theme({
    "primary": "bold green",
    "secondary": "yellow",
    "user": "bright_white",
    "assistant": "green",
    "border": "green",
})

SUNSET = Theme({
    "primary": "bold red",
    "secondary": "orange3",
    "user": "bright_white",
    "assistant": "bright_red",
    "border": "red",
})

THEMES = {
    "cyberpunk": CYBERPUNK,
    "ocean": OCEAN,
    "forest": FOREST,
    "sunset": SUNSET,
}
