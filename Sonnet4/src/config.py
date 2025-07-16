"""
Configuration module for the Context7 Agent.

Handles environment variables, validation, and MCP server configuration.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    """Configuration class for the Context7 Agent."""
    
    # OpenAI Configuration
    openai_api_key: str
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    
    # MCP Configuration
    mcp_server_command: str = "npx"
    mcp_server_args: list = None
    
    # Application Configuration
    data_dir: Path = Path.home() / ".context7-agent"
    max_history_size: int = 1000
    default_theme: str = "cyberpunk"
    
    def __post_init__(self):
        """Post-initialization setup."""
        if self.mcp_server_args is None:
            self.mcp_server_args = ["-y", "@upstash/context7-mcp@latest"]
        
        # Ensure data directory exists
        self.data_dir.mkdir(exist_ok=True)
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables."""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            mcp_server_command=os.getenv("MCP_SERVER_COMMAND", "npx"),
            default_theme=os.getenv("DEFAULT_THEME", "cyberpunk"),
        )
    
    def validate(self) -> Optional[str]:
        """Validate configuration and return error message if invalid."""
        if not self.openai_api_key:
            return "OPENAI_API_KEY environment variable is required"
        
        if not self.openai_model:
            return "OPENAI_MODEL must be specified"
        
        return None
    
    def get_mcp_config(self) -> Dict[str, Any]:
        """Get MCP server configuration dictionary."""
        return {
            "command": self.mcp_server_command,
            "args": self.mcp_server_args
        }

# Global configuration instance
config = Config.from_env()
