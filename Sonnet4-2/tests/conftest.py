# File: tests/conftest.py
"""Shared test fixtures and utilities for Context7 Agent tests."""
import asyncio
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from pydantic_ai import RunResult
from pydantic_ai.mcp import MCPServerStdio

from src.agent import Context7Agent
from src.config import Config
from src.history import HistoryManager
from src.themes import Theme


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def test_config(temp_dir: Path) -> Config:
    """Create test configuration with temporary paths."""
    config_data = {
        "openai_api_key": "test-key",
        "openai_base_url": "https://test-api.openai.com/v1",
        "openai_model": "gpt-4o-mini",
        "default_theme": "cyberpunk",
        "animation_speed": 0.01,
        "max_history": 10,
        "history_file": temp_dir / "test_history.json",
        "sessions_file": temp_dir / "test_sessions.json",
        "bookmarks_file": temp_dir / "test_bookmarks.json",
    }
    return Config(**config_data)


@pytest.fixture
def sample_history_data() -> Dict[str, List[Dict[str, Any]]]:
    """Sample history data for testing."""
    return {
        "test_conv_1": [
            {
                "role": "user",
                "content": "What is FastAPI?",
                "timestamp": "2024-01-01T10:00:00"
            },
            {
                "role": "assistant",
                "content": "FastAPI is a modern web framework...",
                "timestamp": "2024-01-01T10:00:01"
            }
        ],
        "test_conv_2": [
            {
                "role": "user",
                "content": "How to use async?",
                "timestamp": "2024-01-01T11:00:00"
            }
        ]
    }


@pytest.fixture
def mock_run_result() -> RunResult:
    """Create a mock RunResult for agent testing."""
    result = MagicMock(spec=RunResult)
    result.data = "Mock response from agent"
    return result


@pytest.fixture
def mock_mcp_server() -> AsyncMock:
    """Create a mock MCP server."""
    mock = AsyncMock(spec=MCPServerStdio)
    mock.start = AsyncMock()
    mock.stop = AsyncMock()
    return mock


@pytest_asyncio.fixture
async def history_manager(test_config: Config) -> HistoryManager:
    """Create a HistoryManager instance for testing."""
    manager = HistoryManager()
    manager.history_path = test_config.history_file
    manager.max_history = test_config.max_history
    yield manager


@pytest_asyncio.fixture
async def agent_instance(test_config: Config) -> Context7Agent:
    """Create an agent instance for testing."""
    agent = Context7Agent()
    # Mock the actual initialization to avoid external calls
    agent.history.history_path = test_config.history_file
    agent.history.max_history = test_config.max_history
    yield agent


@pytest.fixture
def sample_theme() -> Theme:
    """Create a sample theme for testing."""
    return Theme(
        name="test",
        primary="#ff0000",
        secondary="#00ff00",
        accent="#0000ff",
        background="#000000",
        text="#ffffff",
        success="#00ff00",
        warning="#ffff00",
        error="#ff0000",
        gradient_colors=["#ff0000", "#00ff00"],
        ascii_art="Test ASCII Art"
    )
