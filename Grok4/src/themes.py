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
    # Add more ASCII art...
}

def get_theme_styles(theme: str) -> dict:
    if theme == "cyberpunk":
        return {
            "panel": "bold magenta on black",
            "response": "cyan on black",
            "preview": "green on black",
            "header": "bold yellow on black",
            "footer": "dim white on black",
            "loader": "blink magenta"
        }
    # Add styles for other themes...
    return {}
