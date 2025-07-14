# File: tests/test_agent.py
import pytest
from src.agent import Context7Agent

def test_agent_init():
    agent = Context7Agent()
    assert agent.model.model == "gpt-4o-mini"  # Assuming default

def test_detect_intent():
    agent = Context7Agent()
    assert agent.detect_intent("Tell me about AI") == "search"
    assert agent.detect_intent("/help") == "command"
    assert agent.detect_intent("Hello") == "chat"
