# File: src/__init__.py
"""
Context7 Agent - AI-powered document search and analysis with beautiful TUI.

A sophisticated AI agent built with Pydantic AI that integrates with the Context7 MCP server
for enhanced document discovery, analysis, and management. Features a stunning terminal
interface with themes, animations, and conversational interactions.
"""

from .agent import Context7Agent
from .config import config
# We also expose 'Theme' as it's used for type hinting in the CLI
from .themes import get_theme, list_themes, Theme
# This import is corrected to reflect the refactored, simpler HistoryManager
from .history import HistoryManager

# Version is bumped to 2.0.0 to signify the major architectural refactoring
__version__ = "2.0.0"
__author__ = "Context7 Agent Team"
__email__ = "agent@context7.ai"   

# The __all__ list is updated to define the new, correct public API
__all__ = [
    "Context7Agent",
    "config",
    "get_theme",
    "list_themes",
    "Theme",
    "HistoryManager"
]
