# File: src/config.py
"""
Enhanced configuration module for Context7 Document Explorer.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path


@dataclass
class Config:
    """Configuration for the Context7 Document Explorer."""
    
    # OpenAI configuration
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    
    # Context7 configuration
    context7_workspace: str = "default"
    context7_index_path: str = "./documents"
    
    # UI configuration
    theme: str = "cyberpunk"  # cyberpunk, ocean, forest, sunset
    animations_enabled: bool = True
    sound_effects: bool = True
    
    # Search configuration
    max_results: int = 20
    preview_lines: int = 5
    highlight_matches: bool = True
    fuzzy_search: bool = True
    
    # Application settings
    data_dir: Path = field(default_factory=lambda: Path.home() / ".context7_explorer")
    history_file: str = "search_history.json"
    bookmarks_file: str = "bookmarks.json"
    sessions_dir: str = "sessions"
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        config = cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4"),
            context7_workspace=os.getenv("CONTEXT7_WORKSPACE", "default"),
            context7_index_path=os.getenv("CONTEXT7_INDEX_PATH", "./documents"),
            theme=os.getenv("THEME", "cyberpunk"),
            animations_enabled=os.getenv("ANIMATIONS_ENABLED", "true").lower() == "true",
            sound_effects=os.getenv("SOUND_EFFECTS", "false").lower() == "true",
        )
        
        # Ensure directories exist
        config.data_dir.mkdir(parents=True, exist_ok=True)
        (config.data_dir / config.sessions_dir).mkdir(exist_ok=True)
        
        return config
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY environment variable is required")
        
        if not Path(self.context7_index_path).exists():
            Path(self.context7_index_path).mkdir(parents=True, exist_ok=True)
        
        return errors


# Global config instance
config = Config.from_env()
