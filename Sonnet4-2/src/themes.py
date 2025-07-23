"""
Themes and visual styling for the Context7 Agent TUI.

Provides beautiful themes with gradients, animations, and ASCII art.
"""

import time
import random
from typing import Dict, List, Tuple
from rich.style import Style
from rich.text import Text
from rich.panel import Panel
#from rich.gradient import Gradient
from rich_gradient import Gradient
from rich.console import Console

class Theme:
    """Base theme class with colors, styles, and animations."""
    
    def __init__(
        self,
        name: str,
        primary: str,
        secondary: str,
        accent: str,
        background: str,
        text: str,
        success: str,
        warning: str,
        error: str,
        gradient_colors: List[str],
        ascii_art: str
    ):
        self.name = name
        self.primary = primary
        self.secondary = secondary
        self.accent = accent
        self.background = background
        self.text = text
        self.success = success
        self.warning = warning
        self.error = error
        self.gradient_colors = gradient_colors
        self.ascii_art = ascii_art
    
    def get_gradient(self, text: str) -> Text:
        """Create gradient text."""
        return Text.from_markup(f"[{self.gradient_colors[0]}]{text}[/]")
    
    def get_panel_style(self) -> Style:
        """Get panel border style."""
        return Style(color=self.primary, bold=True)
    
    def get_header_style(self) -> Style:
        """Get header text style."""
        return Style(color=self.accent, bold=True)
    
    def get_success_style(self) -> Style:
        """Get success message style."""
        return Style(color=self.success, bold=True)
    
    def get_error_style(self) -> Style:
        """Get error message style."""
        return Style(color=self.error, bold=True)

# Define beautiful themes
THEMES = {
    "cyberpunk": Theme(
        name="Cyberpunk",
        primary="#ff00ff",      # Magenta
        secondary="#00ffff",    # Cyan
        accent="#ffff00",       # Yellow
        background="#1a0033",   # Dark purple
        text="#ffffff",         # White
        success="#00ff00",      # Green
        warning="#ff8800",      # Orange
        error="#ff0000",        # Red
        gradient_colors=["#ff00ff", "#00ffff", "#ffff00"],
        ascii_art="""
╔═══════════════════════════════════════════════════════════╗
║  ██████╗██╗   ██╗██████╗ ███████╗██████╗ ██████╗ ██╗   ██╗║
║ ██╔════╝╚██╗ ██╔╝██╔══██╗██╔════╝██╔══██╗██╔══██╗██║   ██║║
║ ██║      ╚████╔╝ ██████╔╝█████╗  ██████╔╝██████╔╝██║   ██║║
║ ██║       ╚██╔╝  ██╔══██╗██╔══╝  ██╔══██╗██╔═══╝ ██║   ██║║
║ ╚██████╗   ██║   ██████╔╝███████╗██║  ██║██║     ╚██████╔╝║
║  ╚═════╝   ╚═╝   ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝      ╚═════╝ ║
╚═══════════════════════════════════════════════════════════╝
        """
    ),
    
    "ocean": Theme(
        name="Ocean",
        primary="#0077be",      # Ocean blue
        secondary="#40e0d0",    # Turquoise
        accent="#87ceeb",       # Sky blue
        background="#001122",   # Deep ocean
        text="#f0f8ff",         # Alice blue
        success="#20b2aa",      # Light sea green
        warning="#ffd700",      # Gold
        error="#ff6347",        # Tomato
        gradient_colors=["#0077be", "#40e0d0", "#87ceeb"],
        ascii_art="""
╔═══════════════════════════════════════════════════════════╗
║   ██████╗  ██████╗███████╗ █████╗ ███╗   ██╗              ║
║  ██╔═══██╗██╔════╝██╔════╝██╔══██╗████╗  ██║              ║
║  ██║   ██║██║     █████╗  ███████║██╔██╗ ██║              ║
║  ██║   ██║██║     ██╔══╝  ██╔══██║██║╚██╗██║              ║
║  ╚██████╔╝╚██████╗███████╗██║  ██║██║ ╚████║              ║
║   ╚═════╝  ╚═════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝              ║
╚═══════════════════════════════════════════════════════════╝
        """
    ),
    
    "forest": Theme(
        name="Forest",
        primary="#228b22",      # Forest green
        secondary="#32cd32",    # Lime green
        accent="#90ee90",       # Light green
        background="#013220",   # Dark forest
        text="#f5fffa",         # Mint cream
        success="#00ff7f",      # Spring green
        warning="#ffa500",      # Orange
        error="#dc143c",        # Crimson
        gradient_colors=["#228b22", "#32cd32", "#90ee90"],
        ascii_art="""
╔═══════════════════════════════════════════════════════════╗
║ ███████╗ ██████╗ ██████╗ ███████╗███████╗████████╗        ║
║ ██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔════╝╚══██╔══╝        ║
║ █████╗  ██║   ██║██████╔╝█████╗  ███████╗   ██║           ║
║ ██╔══╝  ██║   ██║██╔══██╗██╔══╝  ╚════██║   ██║           ║
║ ██║     ╚██████╔╝██║  ██║███████╗███████║   ██║           ║
║ ╚═╝      ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝   ╚═╝           ║
╚═══════════════════════════════════════════════════════════╝
        """
    ),
    
    "sunset": Theme(
        name="Sunset",
        primary="#ff6347",      # Tomato
        secondary="#ffa500",    # Orange
        accent="#ffd700",       # Gold
        background="#2f1b14",   # Dark brown
        text="#fff8dc",         # Cornsilk
        success="#32cd32",      # Lime green
        warning="#ff8c00",      # Dark orange
        error="#b22222",        # Fire brick
        gradient_colors=["#ff6347", "#ffa500", "#ffd700"],
        ascii_art="""
╔═══════════════════════════════════════════════════════════╗
║ ███████╗██╗   ██╗███╗   ██╗███████╗███████╗████████╗      ║
║ ██╔════╝██║   ██║████╗  ██║██╔════╝██╔════╝╚══██╔══╝      ║
║ ███████╗██║   ██║██╔██╗ ██║███████╗█████╗     ██║         ║
║ ╚════██║██║   ██║██║╚██╗██║╚════██║██╔══╝     ██║         ║
║ ███████║╚██████╔╝██║ ╚████║███████║███████╗   ██║         ║
║ ╚══════╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝╚══════╝   ╚═╝         ║
╚═══════════════════════════════════════════════════════════╝
        """
    )
}

class AnimationEffects:
    """Animation effects for the TUI."""
    
    @staticmethod
    def typing_effect(console: Console, text: str, speed: float = 0.05, style: str = ""):
        """Simulate typing effect."""
        for i in range(len(text) + 1):
            console.clear()
            if style:
                console.print(f"[{style}]{text[:i]}[/]", end="")
            else:
                console.print(text[:i], end="")
            if i < len(text):
                time.sleep(speed)
    
    @staticmethod
    def particle_loader(console: Console, theme: Theme, duration: float = 2.0):
        """Animated particle loader effect."""
        particles = ["⋆", "✦", "✧", "⋅", "·"]
        start_time = time.time()
        
        while time.time() - start_time < duration:
            console.clear()
            loader_text = ""
            for i in range(20):
                if random.random() < 0.3:
                    particle = random.choice(particles)
                    loader_text += f"[{theme.accent}]{particle}[/]"
                else:
                    loader_text += " "
            
            console.print(f"\n{loader_text}\n", justify="center")
            time.sleep(0.1)
    
    @staticmethod
    def pulse_text(text: str, theme: Theme) -> Text:
        """Create pulsing text effect."""
        pulse_text = Text()
        for i, char in enumerate(text):
            intensity = abs(time.time() * 3 + i) % 2
            if intensity > 1:
                intensity = 2 - intensity
            
            alpha = int(255 * intensity)
            color = f"#{alpha:02x}{alpha:02x}{alpha:02x}"
            pulse_text.append(char, style=Style(color=color))
        
        return pulse_text

def get_theme(name: str) -> Theme:
    """Get theme by name."""
    return THEMES.get(name.lower(), THEMES["cyberpunk"])

def list_themes() -> List[str]:
    """List available theme names."""
    return list(THEMES.keys())
