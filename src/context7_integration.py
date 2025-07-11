# File: src/context7_integration.py
"""
Deep integration with Context7 MCP server for document management.
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from datetime import datetime
import hashlib
import os

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from src.config import config


class Document(BaseModel):
    """Document model for Context7."""
    
    id: str
    title: str
    path: str
    content: str
    preview: str
    metadata: Dict[str, Any]
    score: float = 0.0
    last_modified: datetime
    size: int
    file_type: str
    tags: List[str] = []


class SearchQuery(BaseModel):
    """Search query model."""
    
    query: str
    filters: Dict[str, Any] = {}
    limit: int = 20
    offset: int = 0
    sort_by: str = "relevance"
    include_content: bool = False


class Context7Manager:
    """Manager for Context7 MCP server integration."""
    
    def __init__(self):
        self.agent = None
        self.mcp_client = None
        self.index_path = Path(config.context7_index_path)
        self.workspace = config.context7_workspace
        self._ensure_directories()
        
        # Document cache
        self._document_cache: Dict[str, Document] = {}
        self._index_metadata: Dict[str, Any] = {}
        
        # Initialize OpenAI model
        self.model = OpenAIModel(model_name=config.openai_model)
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        self.index_path.mkdir(parents=True, exist_ok=True)
        (self.index_path / ".context7").mkdir(exist_ok=True)
    
    async def initialize(self):
        """Initialize Context7 MCP connection and agent."""
        try:
            # Import MCP-related modules
            from pydantic_ai.tools.mcp import MCPClient
            
            # Create MCP client for Context7
            self.mcp_client = MCPClient(
                command="npx",
                args=["-y", "@upstash/context7-mcp@latest"],
                env={
                    **os.environ.copy(),
                    "CONTEXT7_WORKSPACE": self.workspace,
                    "CONTEXT7_INDEX_PATH": str(self.index_path)
                }
            )
            
            # Connect to the MCP server
            await self.mcp_client.connect()
            
            # Initialize the agent with Context7 tools
            self.agent = Agent(
                llm=self.model,
                system_prompt="""You are a document search and analysis expert with access to 
                Context7 MCP server. You help users find relevant documents, extract insights, 
                and provide intelligent summaries. You can search through documents, analyze 
                their content, and identify relationships between different documents."""
            )
            
            # Register Context7 tools
            tools = await self.mcp_client.list_tools()
            for tool in tools:
                self.agent.register_tool(tool)
            
            # Load index metadata
            await self._load_index_metadata()
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize Context7: {e}")
            return False
    
    async def _load_index_metadata(self):
        """Load metadata about the document index."""
        metadata_file = self.index_path / ".context7" / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self._index_metadata = json.load(f)
    
    async def search_documents(
        self, 
        query: SearchQuery,
        progress_callback: Optional[Callable] = None
    ) -> List[Document]:
        """Search documents using Context7."""
        # --- BEGIN MODIFICATION ---
        # Add guard clause to prevent crash in offline mode.
        if not self.agent:
            print("Search error: Context7 agent not initialized (offline mode).")
            return []
        # --- END MODIFICATION ---
        try:
            # Use the agent to search with Context7 tools
            search_prompt = f"""
            Search for documents matching: {query.query}
            
            Apply these filters if specified:
            {json.dumps(query.filters, indent=2)}
            
            Return up to {query.limit} results sorted by {query.sort_by}.
            Include document preview and metadata.
            """
            
            result = await self.agent.run(search_prompt)
            
            # Parse the results
            documents = self._parse_search_results(result)
            
            # Apply client-side filtering and scoring
            documents = self._apply_filters(documents, query.filters)
            documents = self._calculate_scores(documents, query.query)
            
            # Sort and limit results
            documents.sort(key=lambda d: d.score, reverse=True)
            documents = documents[:query.limit]
            
            # Cache results
            for doc in documents:
                self._document_cache[doc.id] = doc
            
            return documents
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def _parse_search_results(self, raw_results: str) -> List[Document]:
        """Parse search results from Context7."""
        documents = []
        
        try:
            # Parse JSON response from Context7
            results = json.loads(raw_results)
            
            for item in results.get("documents", []):
                doc = Document(
                    id=self._generate_doc_id(item["path"]),
                    title=item.get("title", Path(item["path"]).stem),
                    path=item["path"],
                    content=item.get("content", ""),
                    preview=self._generate_preview(item.get("content", "")),
                    metadata=item.get("metadata", {}),
                    last_modified=datetime.fromisoformat(
                        item.get("last_modified", datetime.now().isoformat())
                    ),
                    size=item.get("size", 0),
                    file_type=Path(item["path"]).suffix[1:],
                    tags=item.get("tags", [])
                )
                documents.append(doc)
                
        except Exception as e:
            print(f"Error parsing results: {e}")
        
        return documents
    
    def _generate_doc_id(self, path: str) -> str:
        """Generate unique document ID from path."""
        return hashlib.md5(path.encode()).hexdigest()
    
    def _generate_preview(self, content: str, max_length: int = 200) -> str:
        """Generate document preview."""
        if not content:
            return "No preview available"
        
        # Clean and truncate content
        preview = content.strip()
        if len(preview) > max_length:
            preview = preview[:max_length] + "..."
        
        return preview
    
    def _apply_filters(
        self, 
        documents: List[Document], 
        filters: Dict[str, Any]
    ) -> List[Document]:
        """Apply filters to document list."""
        filtered = documents
        
        # File type filter
        if "file_types" in filters:
            allowed_types = filters["file_types"]
            filtered = [d for d in filtered if d.file_type in allowed_types]
        
        # Date range filter
        if "date_from" in filters:
            date_from = datetime.fromisoformat(filters["date_from"])
            filtered = [d for d in filtered if d.last_modified >= date_from]
        
        if "date_to" in filters:
            date_to = datetime.fromisoformat(filters["date_to"])
            filtered = [d for d in filtered if d.last_modified <= date_to]
        
        # Size filter
        if "min_size" in filters:
            filtered = [d for d in filtered if d.size >= filters["min_size"]]
        
        if "max_size" in filters:
            filtered = [d for d in filtered if d.size <= filters["max_size"]]
        
        # Tag filter
        if "tags" in filters:
            required_tags = set(filters["tags"])
            filtered = [
                d for d in filtered 
                if required_tags.intersection(set(d.tags))
            ]
        
        return filtered
    
    def _calculate_scores(self, documents: List[Document], query: str) -> List[Document]:
        """Calculate relevance scores for documents."""
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        for doc in documents:
            score = 0.0
            
            # Title match (highest weight)
            title_lower = doc.title.lower()
            if query_lower in title_lower:
                score += 0.5
            else:
                title_terms = set(title_lower.split())
                score += 0.3 * len(query_terms.intersection(title_terms)) / len(query_terms)
            
            # Content match
            content_lower = doc.content.lower()
            if query_lower in content_lower:
                score += 0.3
            else:
                content_terms = set(content_lower.split())
                score += 0.2 * len(query_terms.intersection(content_terms)) / len(query_terms)
            
            # Path match
            if query_lower in doc.path.lower():
                score += 0.1
            
            # Tag match
            tag_terms = set(" ".join(doc.tags).lower().split())
            score += 0.1 * len(query_terms.intersection(tag_terms)) / max(len(query_terms), 1)
            
            doc.score = min(score, 1.0)
        
        return documents
    
    async def get_document_content(self, doc_id: str) -> Optional[str]:
        """Get full document content."""
        if doc_id in self._document_cache:
            doc = self._document_cache[doc_id]
            
            # Load full content if not already loaded
            if not doc.content or len(doc.content) <= len(doc.preview):
                try:
                    file_path = Path(doc.path)
                    if file_path.exists():
                        with open(file_path, 'r', encoding='utf-8') as f:
                            doc.content = f.read()
                except Exception as e:
                    print(f"Error reading document: {e}")
            
            return doc.content
        
        return None
    
    async def analyze_document(self, doc_id: str) -> Dict[str, Any]:
        """Analyze document using AI."""
        # --- BEGIN MODIFICATION ---
        # Add guard clause to prevent crash in offline mode.
        if not self.agent:
            return {"error": "Context7 agent not initialized (offline mode)."}
        # --- END MODIFICATION ---
        content = await self.get_document_content(doc_id)
        if not content:
            return {}
        
        analysis_prompt = f"""
        Analyze this document and provide:
        1. Key topics and themes
        2. Summary (2-3 sentences)
        3. Important entities (people, places, organizations)
        4. Suggested tags
        5. Related document suggestions
        
        Document content:
        {content[:2000]}...
        """
        
        result = await self.agent.run(analysis_prompt)
        
        # Parse and return analysis
        return self._parse_analysis(result)
    
    def _parse_analysis(self, raw_analysis: str) -> Dict[str, Any]:
        """Parse document analysis results."""
        # Simple parsing - in production, use structured output
        return {
            "analysis": raw_analysis,
            "timestamp": datetime.now().isoformat()
        }
    
    async def find_similar_documents(self, doc_id: str, limit: int = 5) -> List[Document]:
        """Find documents similar to the given document."""
        # --- BEGIN MODIFICATION ---
        # Add guard clause to prevent crash in offline mode.
        if not self.agent:
            return []
        # --- END MODIFICATION ---
        if doc_id not in self._document_cache:
            return []
        
        source_doc = self._document_cache[doc_id]
        
        similarity_prompt = f"""
        Find documents similar to: {source_doc.title}
        
        Content preview: {source_doc.preview}
        Tags: {', '.join(source_doc.tags)}
        
        Return up to {limit} similar documents.
        """
        
        result = await self.agent.run(similarity_prompt)
        
        # Parse and return similar documents
        return self._parse_search_results(result)
    
    async def cleanup(self):
        """Clean up resources."""
        if self.mcp_client:
            try:
                await self.mcp_client.disconnect()
            except Exception as e:
                print(f"Error disconnecting MCP client: {e}")
