# src/config.py
"""Configuration management for Context7 Agent."""

import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class Config:
    """Application configuration."""
    
    # OpenAI Configuration
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    
    # Application Configuration
    history_file: Path
    session_file: Path
    bookmarks_file: Path
    
    # UI Configuration
    default_theme: str = "cyberpunk"
    enable_animations: bool = True
    
    def __post_init__(self):
        """Create necessary directories."""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self.session_file.parent.mkdir(parents=True, exist_ok=True)
        self.bookmarks_file.parent.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        data_dir = Path.home() / ".context7-agent"
        
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4"),
            history_file=data_dir / "history.json",
            session_file=data_dir / "sessions.json",
            bookmarks_file=data_dir / "bookmarks.json",
            default_theme=os.getenv("CONTEXT7_THEME", "cyberpunk"),
            enable_animations=os.getenv("CONTEXT7_ANIMATIONS", "true").lower() == "true",
        )
    
    def validate(self) -> Optional[str]:
        """Validate configuration."""
        if not self.openai_api_key:
            return "OPENAI_API_KEY environment variable is required"
        return None


config = Config.from_env()
