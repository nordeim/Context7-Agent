# File: src/agent.py
"""
Agent module for the Context7 Agent.

This module implements a Pydantic AI agent with Context7 MCP server integration.
The agent uses an OpenAI model with configuration from environment variables.
"""

import os
import sys
from typing import Dict, Any, Optional, List, Union, AsyncGenerator

import anyio
import openai
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from history import History
from utils import fuzzy_match


class Context7Agent:
    """
    Context7 Agent implementation using Pydantic AI.

    This agent integrates with the Context7 MCP server for enhanced context management
    and uses an OpenAI model with OpenAIProvider as the underlying LLM provider.
    Supports intent detection, MCP searches, and conversational responses.
    """

    def __init__(self):
        """
        Initialize the Context7 Agent with configuration from environment variables.

        Sets up the OpenAI model, providers, and Context7 MCP server integration
        following a robust, stable, and asynchronous pattern.
        """
        error = config.validate()
        if error:
            raise ValueError(error)

        # 1. Create the provider for its synchronous client and for the Agent model.
        self.provider = OpenAIProvider(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
        )

        # 2. Create a dedicated async client for our async methods (chat, recommendations).
        self.async_client = openai.AsyncOpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
        )

        # 3. Create the pydantic-ai model wrapper, passing the provider.
        self.llm = OpenAIModel(
            model_name=config.openai_model, provider=self.provider
        )

        # 4. Configure the MCP server.
        self.mcp_server = MCPServerStdio(
            **config.mcp_config["mcpServers"]["context7"]
        )

        # 5. Create the agent, passing the model and MCP servers.
        #    The agent will manage the MCP server's lifecycle via an async context manager.
        self.agent = Agent(model=self.llm, mcp_servers=[self.mcp_server])

        # 6. Initialize history.
        self.history = History()

    def detect_intent(self, message: str, context: List[Dict]) -> str:
        """Detect intent with conversation context."""
        full_context = (
            " ".join([msg["content"] for msg in context[-5:]]) + " " + message
        )
        if message.startswith("/search") or any(
            keyword in full_context.lower()
            for keyword in ["search", "find", "docs on", "tell me about"]
        ):
            return "search"
        elif message.startswith("/"):
            return "command"
        return "chat"

    async def query_mcp(
        self, query: str, filters: Optional[Dict] = None
    ) -> AsyncGenerator[Dict, None]:
        """Stream query results from Context7 MCP (simulated async streaming for demo)."""
        mock_docs = [
            {"id": 1, "title": f"Doc on {query}", "content": f"Sample content about {query}.", "tags": ["ai"], "date": "2025-07-13"},
            {"id": 2, "title": f"Advanced {query}", "content": f"Deep dive into {query}.", "tags": ["ethics"], "date": "2025-07-12"},
            {"id": 3, "title": f"Related to {query}", "content": "More info on similar topics.", "tags": ["tech"], "date": "2025-07-11"},
        ]
        results = [doc for doc in mock_docs if fuzzy_match(query, doc["title"])]
        if filters:
            results = [d for d in results if all(d.get(k) == v for k, v in filters.items())]

        streamed_results = []
        for doc in results:
            await anyio.sleep(0.5)  # Simulate non-blocking streaming delay
            streamed_results.append(doc)
            yield doc

        self.history.add_search(query, streamed_results)
        
        # AI recommendations using the async client
        rec_prompt = f"Recommend similar topics based on: {query}"
        rec_response = await self.async_client.chat.completions.create(
            model=config.openai_model,
            messages=[{"role": "user", "content": rec_prompt}],
        )
        yield {"recommendation": rec_response.choices[0].message.content}

    async def generate_response(
        self, message: str, conversation: List[Dict]
    ) -> Union[str, AsyncGenerator[Dict, None]]:
        """Generate response or stream search results using the robust, stable async pattern."""
        intent = self.detect_intent(message, conversation)
        if intent == "search":
            search_query = (
                message.split("about")[-1].strip()
                if "about" in message
                else message.replace("/search", "").strip()
            )
            return self.query_mcp(search_query)
        elif intent == "command":
            return self.handle_command(message)
        else:
            # Bypass the buggy `agent.complete` and use the dedicated async client directly.
            raw_msgs = conversation + [{"role": "user", "content": message}]
            response = await self.async_client.chat.completions.create(
                model=config.openai_model, messages=raw_msgs
            )
            return response.choices[0].message.content

    def handle_command(self, command: str) -> str:
        """Handle hotkey commands."""
        if command == "/help":
            return "Commands: /search <query>, /preview <id>, /bookmark <id>, /theme <name>, /analytics, /exit"
        elif command.startswith("/bookmark"):
            try:
                doc_id = int(command.split()[-1])
                searches = self.history.get_searches()
                if not searches:
                    return "No searches found to bookmark from."
                docs = searches[-1]["results"]
                for doc in docs:
                    if doc.get("id") == doc_id:
                        self.history.add_bookmark(doc)
                        return f"Bookmarked: {doc['title']}"
                return "Doc ID not found in the last search."
            except (ValueError, IndexError):
                return "Invalid command. Use /bookmark <id>."
        elif command == "/analytics":
            searches = self.history.get_searches()
            tags = [tag for search in searches for doc in search.get("results", []) for tag in doc.get("tags", [])]
            common = max(set(tags), key=tags.count) if tags else "None"
            return f"Search count: {len(searches)}\nMost common tag: {common}"
        return "Unknown command."

    def preview_document(self, doc_id: int) -> str:
        """Syntax-highlighted preview (using Rich markup)."""
        searches = self.history.get_searches()
        if not searches:
            return "No search history found."
        docs = searches[-1]["results"]
        for doc in docs:
            if doc.get("id") == doc_id:
                return f"[bold]{doc['title']}[/bold]\n[italic]{doc['content']}[/italic]"
        return "Doc not found in last search results."
