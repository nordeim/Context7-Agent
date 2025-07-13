# File: tests/test_agent.py
import pytest
import asyncio
from src.agent import create_agent

@pytest.mark.asyncio
async def test_agent_basic():
    agent = create_agent()
    reply = await agent.chat("Hello!")
    assert isinstance(reply.content, str)
