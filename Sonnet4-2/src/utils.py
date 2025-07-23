"""
Utility functions for the Context7 Agent.

Provides helper functions for file operations, formatting, and async utilities.
"""

import os
import json
import asyncio
import aiofiles
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path

async def ensure_directory(path: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    directory = Path(path).parent
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

async def load_json_file(filepath: str, default: Any = None) -> Any:
    """Load JSON file asynchronously."""
    try:
        await ensure_directory(filepath)
        if not Path(filepath).exists():
            return default or {}
        
        async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
            content = await f.read()
            return json.loads(content) if content.strip() else (default or {})
    except (json.JSONDecodeError, IOError) as e:
        return default or {}

async def save_json_file(filepath: str, data: Any) -> bool:
    """Save data to JSON file asynchronously."""
    try:
        await ensure_directory(filepath)
        async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(data, indent=2, ensure_ascii=False))
        return True
    except IOError:
        return False

def format_timestamp(timestamp: Optional[float] = None) -> str:
    """Format timestamp to readable string."""
    if timestamp is None:
        timestamp = datetime.now().timestamp()
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
    """Extract keywords from text (simple implementation)."""
    # Simple keyword extraction - in production, use NLP libraries
    words = text.lower().split()
    # Filter out common words
    stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "about"}
    keywords = [word.strip('.,!?;:"()[]{}') for word in words if word not in stop_words and len(word) > 3]
    
    # Return unique keywords, limited to max_keywords
    return list(dict.fromkeys(keywords))[:max_keywords]

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity (0-1)."""
    # Simple Jaccard similarity - in production, use proper similarity measures
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def detect_file_type(filename: str) -> str:
    """Detect file type from filename."""
    ext = Path(filename).suffix.lower()
    
    type_mapping = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.html': 'html',
        '.css': 'css',
        '.json': 'json',
        '.xml': 'xml',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.md': 'markdown',
        '.txt': 'text',
        '.pdf': 'pdf',
        '.doc': 'document',
        '.docx': 'document',
        '.xls': 'spreadsheet',
        '.xlsx': 'spreadsheet',
    }
    
    return type_mapping.get(ext, 'unknown')

async def debounce(func, delay: float = 0.5):
    """Debounce function calls."""
    async def debounced(*args, **kwargs):
        await asyncio.sleep(delay)
        return await func(*args, **kwargs)
    return debounced

def create_progress_bar(current: int, total: int, width: int = 20) -> str:
    """Create a simple progress bar."""
    if total == 0:
        return "[" + "─" * width + "]"
    
    filled = int(width * current / total)
    bar = "█" * filled + "─" * (width - filled)
    percentage = int(100 * current / total)
    
    return f"[{bar}] {percentage}%"

def parse_hotkey_command(message: str) -> tuple[Optional[str], str]:
    """Parse hotkey commands from message."""
    message = message.strip()
    if not message.startswith('/'):
        return None, message
    
    parts = message[1:].split(' ', 1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    
    return command, args

def highlight_search_terms(text: str, search_terms: List[str]) -> str:
    """Highlight search terms in text."""
    highlighted = text
    for term in search_terms:
        highlighted = highlighted.replace(
            term, 
            f"[bold yellow]{term}[/bold yellow]"
        )
    return highlighted
