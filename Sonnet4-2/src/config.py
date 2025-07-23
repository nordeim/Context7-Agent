# File: src/config.py
"""
Configuration management using Pydantic-Settings for a modern, robust setup.
This is the stable, best-practice reference implementation.
"""
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    """
    Application configuration with automatic environment variable loading.
    Defines a clear, type-safe structure for all application settings.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CONTEXT7_", # Fallback prefix
        case_sensitive=False,
        extra="ignore"
    )
    
    # OpenAI Configuration - will look for OPENAI_API_KEY first.
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1")
    openai_model: str = Field(default="gpt-4o-mini")
    
    # Application Settings
    default_theme: str = Field(default="cyberpunk")
    animation_speed: float = Field(default=0.05)
    max_history: int = Field(default=50)

    # File Paths - uses pathlib for robustness
    history_file: Path = Field(default=Path("data/history.json"))
    sessions_file: Path = Field(default=Path("data/sessions.json"))
    bookmarks_file: Path = Field(default=Path("data/bookmarks.json"))
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_data_directory()

    def _ensure_data_directory(self):
        """Ensures the data directory for history/bookmarks/sessions exists."""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

# Global, immutable config instance
config = Config()
