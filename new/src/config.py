# File: src/config.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings  # <-- Correct import for v2+
from pydantic import Field, validator

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")      # cascade env vars


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o-mini", env="OPENAI_MODEL")
    openai_base_url: str | None = Field(None, env="OPENAI_BASE_URL")

    # MCP
    mcp_alias: str = "context7"                         # key in mcp.config.json
    mcp_config_path: Path = ROOT / "mcp.config.json"

    # UX
    theme: Literal["cyberpunk", "ocean", "forest", "sunset"] = Field(
        default="cyberpunk", env="THEME"
    )
    history_path: Path = ROOT / ".history.json"

    class Config:
        env_file = ".env"

    # Extra validation
    @validator("openai_api_key", allow_reuse=True)
    def _check_key(cls, v: str):
        if not v or v.startswith("sk-YOUR"):
            raise ValueError("OPENAI_API_KEY missing or placeholder")
        return v


settings = Settings()
config = settings
