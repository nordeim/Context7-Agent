# src/themes.py
"""Theme definitions for Context7 Agent."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class Theme:
    """Theme definition."""
    
    name: str
    primary: str
    secondary: str
    accent: str
    background: str
    text: str
    success: str
    error: str
    warning: str
    info: str
    
    # Gradient colors
    gradient_start: str
    gradient_end: str
    
    # ASCII art
    ascii_art: str


THEMES: Dict[str, Theme] = {
    "cyberpunk": Theme(
        name="Cyberpunk",
        primary="#FF006E",
        secondary="#8338EC",
        accent="#3A86FF",
        background="#0A0E27",
        text="#FFFFFF",
        success="#06FFA5",
        error="#FF006E",
        warning="#FFBE0B",
        info="#3A86FF",
        gradient_start="#FF006E",
        gradient_end="#8338EC",
        ascii_art=r"""
╔═══════════════════════════════════════════════════════════════╗
║   _____ ____  _   _ _____ _______  _______   ______          ║
║  / ____/ __ \| \ | |_   _|  ___\ \/ /_   _| |____  |         ║
║ | |   | |  | |  \| | | | | |__  \  /  | |       / /          ║
║ | |   | |  | | . ` | | | |  __| /  \  | |      / /           ║
║ | |___| |__| | |\  | | | | |___/ /\ \ | |     / /            ║
║  \_____\____/|_| \_| |_| |_____/_/  \_\|_|   /_/   AGENT     ║
╚═══════════════════════════════════════════════════════════════╝
        """
    ),
    "ocean": Theme(
        name="Ocean",
        primary="#006BA6",
        secondary="#0496FF",
        accent="#1D3557",
        background="#E8F4FD",
        text="#1D3557",
        success="#52B788",
        error="#D00000",
        warning="#FFBA08",
        info="#0496FF",
        gradient_start="#006BA6",
        gradient_end="#0496FF",
        ascii_art=r"""
～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～
    ____                            ___                    __   
   / __ \________  ____ _____     /   | ____ ____  ____  / /_  
  / / / / ___/ _ \/ __ `/ __ \   / /| |/ __ `/ _ \/ __ \/ __/  
 / /_/ / /__/  __/ /_/ / / / /  / ___ / /_/ /  __/ / / / /_    
 \____/\___/\___/\__,_/_/ /_/  /_/  |_\__, /\___/_/ /_/\__/    
                                     /____/                     
～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～
        """
    ),
    "forest": Theme(
        name="Forest",
        primary="#2D6A4F",
        secondary="#40916C",
        accent="#52B788",
        background="#F1FAEE",
        text="#1B4332",
        success="#95D5B2",
        error="#BC4749",
        warning="#F4A261",
        info="#74C69D",
        gradient_start="#2D6A4F",
        gradient_end="#52B788",
        ascii_art=r"""
🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲
    ______                     __     ___                    __ 
   / ____/___  ________  _____/ /_   /   | ____ ____  ____  / /_
  / /_  / __ \/ ___/ _ \/ ___/ __/  / /| |/ __ `/ _ \/ __ \/ __/
 / __/ / /_/ / /  /  __(__  ) /_   / ___ / /_/ /  __/ / / / /_  
/_/    \____/_/   \___/____/\__/  /_/  |_\__, /\___/_/ /_/\__/  
                                        /____/                   
🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲
        """
    ),
    "sunset": Theme(
        name="Sunset",
        primary="#F72585",
        secondary="#7209B7",
        accent="#4361EE",
        background="#FFF3E0",
        text="#240046",
        success="#06FFA5",
        error="#F72585",
        warning="#F77F00",
        info="#4361EE",
        gradient_start="#F72585",
        gradient_end="#4361EE",
        ascii_art=r"""
☀️ ═══════════════════════════════════════════════════════════ ☀️
   ____                       __     ___                    __   
  / __/__  ______  ________  / /_   /   | ____ ____  ____  / /_  
 _\ \/ / / / __ \/ ___/ _ \/ __/  / /| |/ __ `/ _ \/ __ \/ __/  
/___/ /_/ / / / (__  )  __/ /_   / ___ / /_/ /  __/ / / / /_    
   /_____/_/ /_/____/\___/\__/  /_/  |_\__, /\___/_/ /_/\__/    
                                      /____/                     
☀️ ═══════════════════════════════════════════════════════════ ☀️
        """
    ),
}
