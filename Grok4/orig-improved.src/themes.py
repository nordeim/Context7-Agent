# File: src/themes.py
"""
Themes module for the Context7 Agent.

Defines themes, styles, and ASCII art.
"""

THEMES = ["cyberpunk", "ocean", "forest", "sunset"]

ASCII_ART = {
    "cyberpunk": """
   _____           _            _   ___      
  / ____|         | |          | | |__ \\     
 | |    |__   ___ | |_ ___  ___| |_   ) |    
 | |    / _ \\ / _ \\| __/ _ \\/ __| __| / /     
 | |___| (_) | (_) | ||  __/\\__ \\ |_ / /_     
  \\_____\\___/ \\___/ \\__\\___||___/\\__|____|    
    """,
    "ocean": """
  .     .     .     .     .     .     
~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    Context7 Ocean - Dive Deep!     
~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
  .     .     .     .     .     .     
    """,
    "forest": """
   /\\*\\  /\\*\\  /\\*\\   /\\*\\  /\\*\\  /\\*\\
  <___> <___> <___> <___> <___> <___>
    Context7 Forest - Explore...
    """,
    "sunset": """
========================================
    Context7 Sunset - Calm Reflections
========================================
    """
}

def get_theme_styles(theme: str) -> dict:
    """
    Returns a dictionary of Rich styles for a given theme.
    Includes all necessary keys for the immersive UI.
    """
    if theme == "cyberpunk":
        return {
            "panel": "bold magenta on black",
            "response": "cyan on black",
            "preview": "green on black",
            "header": "bold yellow on black",
            "footer": "dim white on black",
            "loader": "blink magenta",
            "chat_user": "bold blue",
            "chat_agent": "bold green",
            "result": "italic cyan",
            "error": "bold white on red",
            "particle": "Searching MCP [magenta]•[/][cyan]•[/][blue]•[/]"
        }
    elif theme == "ocean":
        return {
            "panel": "bold blue on cyan",
            "response": "white on blue",
            "preview": "green on cyan",
            "header": "bold white on blue",
            "footer": "dim black on cyan",
            "loader": "blink blue",
            "chat_user": "bold yellow",
            "chat_agent": "bold white",
            "result": "italic white",
            "error": "bold white on red",
            "particle": "Diving deep [blue]~[/][cyan]~[/][white]~[/]"
        }
    elif theme == "forest":
        return {
            "panel": "bold green on default",
            "response": "yellow on default",
            "preview": "bright_green on default",
            "header": "bold white on green",
            "footer": "dim white on green",
            "loader": "blink green",
            "chat_user": "bold yellow",
            "chat_agent": "bold green",
            "result": "italic white",
            "error": "bold white on red",
            "particle": "Exploring the woods [green]•[/][yellow]•[/]"
        }
    elif theme == "sunset":
        return {
            "panel": "bold yellow on red",
            "response": "white on dark_orange",
            "preview": "cyan on dark_orange",
            "header": "bold white on red",
            "footer": "dim white on red",
            "loader": "blink yellow",
            "chat_user": "bold cyan",
            "chat_agent": "bold yellow",
            "result": "italic white",
            "error": "bold white on red",
            "particle": "Chasing the horizon [yellow]•[/][red]•[/]"
        }
    # Provide a robust default to prevent key errors if a theme is misconfigured.
    return get_theme_styles("cyberpunk")
