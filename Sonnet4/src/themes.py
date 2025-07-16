"""
Themes and visual effects for the Context7 Agent TUI.

Provides stunning visual themes with gradients, animations, and ASCII art.
"""

import time
import random
from typing import Dict, List, Tuple
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
# from rich.gradient import Gradient
from rich.style import Style

class ThemeManager:
    """Manages visual themes and animations for the TUI."""
    
    THEMES = {
        "cyberpunk": {
            "primary": "#ff00ff",
            "secondary": "#00ffff",
            "accent": "#ffff00",
            "background": "#1a0033",
            "text": "#ffffff",
            "error": "#ff0040",
            "success": "#00ff40",
            "gradient": ["#ff00ff", "#00ffff", "#ffff00"],
            "ascii_art": """
╔═══════════════════════════════════════╗
║  ▄████▄ ▓██   ██▓ ▄▄▄▄   ▓█████  ██▀███  ║
║ ▒██▀ ▀█  ▒██  ██▒▓█████▄ ▓█   ▀ ▓██ ▒ ██▒║
║ ▒▓█    ▄  ▒██ ██░▒██▒ ▄██▒███   ▓██ ░▄█ ▒║
║ ▒▓▓▄ ▄██▒ ░ ▐██▓░▒██░█▀  ▒▓█  ▄ ▒██▀▀█▄  ║
║ ▒ ▓███▀ ░ ░ ██▒▓░░▓█  ▀█▓░▒████▒░██▓ ▒██▒║
║ ░ ░▒ ▒  ░  ██▒▒▒ ░▒▓███▀▒░░ ▒░ ░░ ▒▓ ░▒▓░║
║   ░  ▒   ▓██ ░▒░ ▒░▒   ░  ░ ░  ░  ░▒ ░ ▒░║
╚═══════════════════════════════════════╝
            """
        },
        "ocean": {
            "primary": "#0066cc",
            "secondary": "#00ccff",
            "accent": "#66ffcc",
            "background": "#001133",
            "text": "#e6f3ff",
            "error": "#ff6666",
            "success": "#66ffaa",
            "gradient": ["#0066cc", "#00ccff", "#66ffcc"],
            "ascii_art": """
╔═══════════════════════════════════════╗
║     🌊  O C E A N   A G E N T  🌊      ║
║                                       ║
║    ~~~   ~~~   ~~~   ~~~   ~~~       ║
║  ~~~   ~~~   ~~~   ~~~   ~~~   ~~~   ║
║    ~~~   ~~~   ~~~   ~~~   ~~~       ║
║  🐋   ~~~   ~~~   ~~~   ~~~   🐠     ║
║    ~~~   ~~~   ~~~   ~~~   ~~~       ║
╚═══════════════════════════════════════╝
            """
        },
        "forest": {
            "primary": "#228b22",
            "secondary": "#32cd32",
            "accent": "#90ee90",
            "background": "#0d2818",
            "text": "#f0fff0",
            "error": "#ff4444",
            "success": "#44ff44",
            "gradient": ["#228b22", "#32cd32", "#90ee90"],
            "ascii_art": """
╔═══════════════════════════════════════╗
║     🌲  F O R E S T   A G E N T  🌲    ║
║                                       ║
║    🌳    🌲    🌳    🌲    🌳        ║
║  🌲  🦋  🌳  🐿️  🌲  🦌  🌳  🐦    ║
║    🌳    🌲    🌳    🌲    🌳        ║
║  🌲    🌳    🌲    🌳    🌲          ║
║    🌳    🌲    🌳    🌲    🌳        ║
╚═══════════════════════════════════════╝
            """
        },
        "sunset": {
            "primary": "#ff6600",
            "secondary": "#ff9933",
            "accent": "#ffcc66",
            "background": "#331100",
            "text": "#fff5e6",
            "error": "#ff3333",
            "success": "#ffaa33",
            "gradient": ["#ff6600", "#ff9933", "#ffcc66"],
            "ascii_art": """
╔═══════════════════════════════════════╗
║     🌅  S U N S E T   A G E N T  🌅    ║
║                                       ║
║           ☀️                          ║
║        🌅🌅🌅🌅🌅                      ║
║      🌅🌅🌅🌅🌅🌅🌅                    ║
║    🌆🌆🌆🌆🌆🌆🌆🌆🌆                  ║
║  🏙️🏙️🏙️🏙️🏙️🏙️🏙️🏙️🏙️🏙️              ║
╚═══════════════════════════════════════╝
            """
        }
    }
    
    def __init__(self, console: Console):
        """Initialize the theme manager."""
        self.console = console
        self.current_theme = "cyberpunk"
    
    def set_theme(self, theme_name: str) -> bool:
        """Set the current theme."""
        if theme_name in self.THEMES:
            self.current_theme = theme_name
            return True
        return False
    
    def get_theme(self) -> Dict:
        """Get the current theme configuration."""
        return self.THEMES[self.current_theme]
    
    def create_gradient_text(self, text: str, style_name: str = "gradient") -> Text:
        """Create gradient text with the current theme colors."""
        theme = self.get_theme()
        colors = theme["gradient"]
        
        gradient_text = Text()
        text_len = len(text)
        
        for i, char in enumerate(text):
            # Calculate color position in gradient
            position = i / max(1, text_len - 1)
            
            # Interpolate between colors
            if position <= 0.5:
                # First half: primary to secondary
                ratio = position * 2
                color = self._interpolate_color(colors[0], colors[1], ratio)
            else:
                # Second half: secondary to accent
                ratio = (position - 0.5) * 2
                color = self._interpolate_color(colors[1], colors[2], ratio)
            
            gradient_text.append(char, style=color)
        
        return gradient_text
    
    def _interpolate_color(self, color1: str, color2: str, ratio: float) -> str:
        """Interpolate between two hex colors."""
        # Convert hex to RGB
        r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
        r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
        
        # Interpolate
        r = int(r1 + (r2 - r1) * ratio)
        g = int(g1 + (g2 - g1) * ratio)
        b = int(b1 + (b2 - b1) * ratio)
        
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def show_welcome_screen(self):
        """Display the themed welcome screen with ASCII art."""
        theme = self.get_theme()
        
        # Clear screen
        self.console.clear()
        
        # Create welcome panel
        welcome_text = self.create_gradient_text("CONTEXT7 AI AGENT", "gradient")
        ascii_art = Text(theme["ascii_art"], style=theme["primary"])
        
        panel = Panel(
            ascii_art + "\n\n" + welcome_text,
            title="🚀 Welcome to the Future of AI",
            border_style=theme["primary"],
            padding=(1, 2)
        )
        
        self.console.print(panel, justify="center")
        
        # Animated loading
        self.animated_loading("Initializing quantum processors", 2.0)
    
    def animated_loading(self, message: str, duration: float):
        """Show animated loading with particles."""
        theme = self.get_theme()
        
        with Progress(
            SpinnerColumn("dots12", style=theme["primary"]),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True
        ) as progress:
            task = progress.add_task(message, total=100)
            
            for i in range(100):
                progress.update(task, advance=1)
                time.sleep(duration / 100)
    
    def create_typing_effect(self, text: str, delay: float = 0.05):
        """Create a typing animation effect."""
        theme = self.get_theme()
        
        for char in text:
            self.console.print(char, style=theme["text"], end="")
            time.sleep(delay)
        self.console.print()  # New line at the end
    
    def create_panel(self, content: str, title: str, style: str = "primary") -> Panel:
        """Create a themed panel."""
        theme = self.get_theme()
        return Panel(
            content,
            title=title,
            border_style=theme[style],
            padding=(1, 2)
        )
    
    def create_table(self, title: str) -> Table:
        """Create a themed table."""
        theme = self.get_theme()
        table = Table(title=title, style=theme["text"])
        table.title_style = theme["primary"]
        return table
    
    def show_error(self, message: str):
        """Display an error message with theme styling."""
        theme = self.get_theme()
        error_panel = Panel(
            f"❌ {message}",
            title="Error",
            border_style=theme["error"],
            padding=(1, 2)
        )
        self.console.print(error_panel)
    
    def show_success(self, message: str):
        """Display a success message with theme styling."""
        theme = self.get_theme()
        success_panel = Panel(
            f"✅ {message}",
            title="Success",
            border_style=theme["success"],
            padding=(1, 2)
        )
        self.console.print(success_panel)
    
    def particle_animation(self, duration: float = 3.0):
        """Show a particle animation effect."""
        theme = self.get_theme()
        particles = ["✨", "⭐", "💫", "🌟", "✴️", "💥"]
        
        start_time = time.time()
        while time.time() - start_time < duration:
            # Clear previous particles
            self.console.print("\n" * 3, end="")
            
            # Generate random particles
            line = ""
            for _ in range(40):
                if random.random() < 0.3:
                    particle = random.choice(particles)
                    line += particle
                else:
                    line += " "
            
            self.console.print(line, style=theme["accent"])
            time.sleep(0.1)
    
    def get_available_themes(self) -> List[str]:
        """Get list of available theme names."""
        return list(self.THEMES.keys())
