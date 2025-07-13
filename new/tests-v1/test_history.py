# File: tests/test_history.py
from pathlib import Path
from src.history import History

def test_history_roundtrip(tmp_path: Path):
    h = History(tmp_path / "hist.json")
    h.add("user", "Hello")
    h.add("assistant", "Hi!")
    h2 = History(tmp_path / "hist.json")
    assert len(h2.messages) == 2

# tests/test_agent.py
import asyncio
from src.agent import create_agent

def test_agent_can_instantiate():
    agent = create_agent()
    assert agent.model
    assert agent.mcp_server

async def test_agent_short_chat():
    agent = create_agent()
    resp = await agent.chat("Ping?")
    assert "Ping" in resp.content
