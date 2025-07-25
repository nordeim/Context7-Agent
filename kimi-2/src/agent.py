"""
Production-ready AI agent with Pydantic AI and MCP integration.
This version implements a simplified RAG pattern that works with Context7 MCP server's
actual output format (human-readable markdown responses).
*** THIS VERSION INCLUDES DIAGNOSTIC PRINT STATEMENTS FOR DEBUGGING ***
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime
import logging

# Import rich for styled, clear diagnostic printing
from rich import print as rprint

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from .config import config
from .history import HistoryManager

logger = logging.getLogger(__name__)

class Context7Agent:
    """Production-ready AI agent implementing a simplified RAG pattern for Context7 MCP."""
    
    def __init__(self):
        """Initialize with correct Pydantic AI v0.5+ patterns."""
        self.provider = OpenAIProvider(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url
        )
        
        self.model = OpenAIModel(
            model_name=config.openai_model,
            provider=self.provider
        )
        
        self.mcp_server = MCPServerStdio(
            command="npx",
            args=["-y", "@upstash/context7-mcp@latest"]
        )
        
        self.agent = Agent(
            model=self.model,
            mcp_servers=[self.mcp_server],
            system_prompt="""
            You are a helpful AI research assistant. Your task is to provide a clear and comprehensive answer to the user's question based on the provided context documents. Synthesize the information from the various documents into a single, coherent response. If the context is insufficient to answer the question, state that clearly.
            """
        )
        
        self.history = HistoryManager()
    
    async def initialize(self):
        """Initialize the agent and load history."""
        await self.history.load()
    
    async def chat_stream(
        self, 
        message: str, 
        conversation_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream chat responses using a simplified RAG pipeline for Context7 MCP.
        """
        try:
            rprint("[bold yellow]--- Starting RAG Pipeline ---[/bold yellow]")
            async with self.agent.run_mcp_servers():
                # =====================================================================
                # == STEP 1: FORMULATE SEARCH QUERY                                  ==
                # =====================================================================
                rprint("[bold yellow][DEBUG-STEP-1A][/bold yellow] Formulating search query from user message...")
                query_formulation_prompt = f"""
                Based on the user's request, what is the optimal, concise search query for the Context7 documentation database?
                The user's request is: "{message}"
                Return ONLY the search query string, and nothing else.
                """
                
                query_formulation_agent = Agent(model=self.model)
                query_result = await query_formulation_agent.run(query_formulation_prompt)
                search_query = str(query_result.data).strip().strip('"')

                rprint(f"[bold yellow][DEBUG-STEP-1B][/bold yellow] Formulated Search Query: [cyan]'{search_query}'[/cyan]")

                if not search_query:
                    raise ValueError("LLM failed to formulate a search query.")
                
                # =====================================================================
                # == STEP 2: EXECUTE THE TOOL AND GET DIRECT RESPONSE                ==
                # =====================================================================
                search_prompt = f"search for documentation about: {search_query}"
                rprint(f"[bold yellow][DEBUG-STEP-2A][/bold yellow] Executing tool call with prompt: [cyan]'{search_prompt}'[/cyan]")
                
                retrieval_result = await self.agent.run(search_prompt)
                response_content = str(retrieval_result.data).strip()
                
                rprint(f"[bold green][DEBUG-STEP-2B][/bold green] Received response from Context7 MCP:")
                rprint(f"[dim white]{response_content[:200]}...[/dim white]")
                
                if not response_content:
                    rprint("[bold red][DEBUG-FAIL][/bold red] Empty response received from MCP server")
                    yield {
                        "type": "content",
                        "data": "I could not find any relevant information in the Context7 knowledge base to answer your question.",
                        "timestamp": datetime.now().isoformat()
                    }
                    return
                
                # =====================================================================
                # == STEP 3: STREAM THE RESPONSE                                     ==
                # =====================================================================
                rprint("[bold green][DEBUG-STEP-3A][/bold green] Streaming response to user...")
                rprint("[bold green]--- RAG Pipeline Succeeded ---[/bold green]")

                yield {
                    "type": "content",
                    "data": response_content,
                    "timestamp": datetime.now().isoformat()
                }
                
                await self.history.add_message(conversation_id or "default", "user", message)
                await self.history.add_message(conversation_id or "default", "assistant", response_content)
                
                yield {
                    "type": "complete",
                    "data": response_content,
                    "timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Agent RAG pipeline error: {e}")
            rprint(f"[bold red][DEBUG-FATAL-ERROR][/bold red] An unexpected exception occurred in the RAG pipeline: {e}")
            yield {
                "type": "error",
                "data": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def search_documents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search documents using MCP tools with proper error handling."""
        try:
            async with self.agent.run_mcp_servers():
                search_prompt = f"Search the Context7 knowledge base for documents about: {query}"
                result = await self.agent.run(search_prompt)
                response_text = str(result.data)
                
                # Return the response as a single document
                return [{
                    "title": f"Search Results for: {query}",
                    "content": response_text,
                    "source": "Context7 MCP",
                    "score": 1.0
                }]
        except Exception as e:
            logger.error(f"Search error: {e}")
            return [{"title": "Search Error", "content": f"Unable to search documents: {str(e)}", "source": "Error", "score": 0.0}]
    
    def get_conversations(self) -> List[Dict[str, Any]]:
        """Get all conversation histories."""
        return self.history.get_conversations()
    
    async def clear_history(self, conversation_id: Optional[str] = None):
        """Clear conversation history."""
        await self.history.clear(conversation_id)
