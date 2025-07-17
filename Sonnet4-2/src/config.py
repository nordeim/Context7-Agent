# File: src/config.py
"""
Configuration module for the Context7 Agent.

Handles environment variables, API keys, and application settings.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class Config:
    """Configuration class for the Context7 Agent."""
    
    # OpenAI Configuration
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    
    # Context7 MCP Configuration
    mcp_server_command: str
    mcp_server_args: list[str]
    
    # Application Settings
    max_history: int
    session_timeout: int
    default_theme: str
    animation_speed: float
    
    # File Paths
    history_file: str
    sessions_file: str
    bookmarks_file: str
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables."""
        return cls(
            # OpenAI Configuration
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            
            # Context7 MCP Configuration
            mcp_server_command=os.getenv("MCP_SERVER_COMMAND", "npx"),
            mcp_server_args=os.getenv("MCP_SERVER_ARGS", "-y,@upstash/context7-mcp@latest").split(","),
            
            # Application Settings
            max_history=int(os.getenv("MAX_HISTORY", "1000")),
            session_timeout=int(os.getenv("SESSION_TIMEOUT", "3600")),
            default_theme=os.getenv("DEFAULT_THEME", "cyberpunk"),
            animation_speed=float(os.getenv("ANIMATION_SPEED", "0.05")),
            
            # File Paths
            history_file=os.getenv("HISTORY_FILE", "data/history.json"),
            sessions_file=os.getenv("SESSIONS_FILE", "data/sessions.json"),
            bookmarks_file=os.getenv("BOOKMARKS_FILE", "data/bookmarks.json"),
        )
    
    def validate(self) -> Optional[str]:
        """Validate configuration and return error message if invalid."""
        if not self.openai_api_key:
            return "OpenAI API key is required"
        
        if not self.openai_model:
            return "OpenAI model is required"
            
        if not self.mcp_server_command:
            return "MCP server command is required"
            
        return None
    
    def to_mcp_config(self) -> Dict[str, Any]:
        """Convert to MCP server configuration format."""
        return {
            "command": self.mcp_server_command,
            "args": self.mcp_server_args
        }

# Global configuration instance
config = Config.from_env()
