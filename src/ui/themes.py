# File: src/ui/themes.py
"""
Theme definitions for the Context7 Document Explorer.
"""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class Theme:
    """Theme configuration for the UI."""
    
    name: str
    primary: str
    secondary: str
    accent: str
    background: str
    surface: str
    text: str
    text_dim: str
    success: str
    warning: str
    error: str
    info: str
    
    # Gradient colors for backgrounds
    gradient_start: str
    gradient_end: str
    
    # Special effects
    glow_color: str
    shadow_color: str
    
    # ASCII art style
    border_style: str
    ascii_art_style: str


THEMES: Dict[str, Theme] = {
    "cyberpunk": Theme(
        name="Cyberpunk",
        primary="#ff006e",
        secondary="#3a0ca3",
        accent="#8338ec",
        background="#0a0a0a",
        surface="#1a1a2e",
        text="#f72585",
        text_dim="#7209b7",
        success="#06ffa5",
        warning="#ffbe0b",
        error="#ff006e",
        info="#4cc9f0",
        gradient_start="#3a0ca3",
        gradient_end="#f72585",
        glow_color="#ff006e",
        shadow_color="#3a0ca3",
        border_style="double",
        ascii_art_style="cyberpunk"
    ),
    
    "ocean": Theme(
        name="Ocean Breeze",
        primary="#006ba6",
        secondary="#0496ff",
        accent="#1d7874",
        background="#001524",
        surface="#15616d",
        text="#78c3fb",
        text_dim="#4a7c7e",
        success="#06ffa5",
        warning="#ffd60a",
        error="#d62828",
        info="#89d2dc",
        gradient_start="#001524",
        gradient_end="#0496ff",
        glow_color="#0496ff",
        shadow_color="#001524",
        border_style="rounded",
        ascii_art_style="waves"
    ),
    
    "forest": Theme(
        name="Forest Deep",
        primary="#2d6a4f",
        secondary="#40916c",
        accent="#52b788",
        background="#081c15",
        surface="#1b4332",
        text="#95d5b2",
        text_dim="#74c69d",
        success="#b7e4c7",
        warning="#f4a261",
        error="#e76f51",
        info="#90e0ef",
        gradient_start="#081c15",
        gradient_end="#52b788",
        glow_color="#52b788",
        shadow_color="#081c15",
        border_style="heavy",
        ascii_art_style="trees"
    ),
    
    "sunset": Theme(
        name="Sunset Glow",
        primary="#f77f00",
        secondary="#fcbf49",
        accent="#eae2b7",
        background="#003049",
        surface="#d62828",
        text="#fcbf49",
        text_dim="#f77f00",
        success="#06ffa5",
        warning="#fcbf49",
        error="#d62828",
        info="#eae2b7",
        gradient_start="#003049",
        gradient_end="#f77f00",
        glow_color="#f77f00",
        shadow_color="#003049",
        border_style="ascii",
        ascii_art_style="sunset"
    )
}


def get_theme(name: str) -> Theme:
    """Get theme by name, default to cyberpunk if not found."""
    return THEMES.get(name, THEMES["cyberpunk"])
