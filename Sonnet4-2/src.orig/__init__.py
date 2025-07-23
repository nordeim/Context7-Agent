# File: src/__init__.py
"""
Context7 Agent - AI-powered document search and analysis with beautiful TUI.

A sophisticated AI agent built with Pydantic AI that integrates with the Context7 MCP server
for enhanced document discovery, analysis, and management. Features a stunning terminal
interface with themes, animations, and conversational interactions.
"""

from .agent import Context7Agent
from .config import config
from .themes import get_theme, list_themes
from .history import ConversationHistory, SearchHistory, BookmarkManager, SessionManager

__version__ = "1.0.0"
__author__ = "Context7 Agent Team"
__email__ = "agent@context7.ai"

__all__ = [
    "Context7Agent",
    "config", 
    "get_theme",
    "list_themes",
    "ConversationHistory",
    "SearchHistory", 
    "BookmarkManager",
    "SessionManager"
]
