# File: src/config.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Literal, Dict, Any
from pydantic import BaseSettings, Field, validator

ROOT = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o-mini", env="OPENAI_MODEL")
    openai_base_url: str | None = Field(default=None, env="OPENAI_BASE_URL")

    # MCP
    mcp_server_alias: Literal["context7"] = "context7"               # JSON config key
    mcp_config_file: Path = ROOT / "mcp.config.json"                 # saved below

    # UX / TUI
    theme: Literal["cyberpunk", "ocean", "forest", "sunset"] = "cyberpunk"
    history_file: Path = ROOT / ".history.json"

    class Config:
        env_file = ROOT / ".env"
        env_file_encoding = "utf-8"

    # Extra runtime validation
    @validator("openai_api_key")
    def _key_must_exist(cls, v):
        if not v or "YOUR_" in v:
            raise ValueError("OPENAI_API_KEY not configured.")
        return v

settings = Settings()
