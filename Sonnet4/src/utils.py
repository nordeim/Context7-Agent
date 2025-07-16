"""
Utility functions for the Context7 Agent.

Provides helper functions for file operations, text processing,
and common operations used throughout the application.
"""

import os
import re
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for filesystem use
    """
    # Remove invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    
    # Limit length
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    
    # Ensure not empty
    if not sanitized:
        sanitized = "untitled"
    
    return sanitized

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string (e.g., "1.2 MB")
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text using simple frequency analysis.
    
    Args:
        text: Input text
        max_keywords: Maximum number of keywords to return
        
    Returns:
        List of extracted keywords
    """
    # Simple keyword extraction
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Filter out common stop words
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
        'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
        'her', 'its', 'our', 'their', 'a', 'an', 'as', 'if', 'so', 'than', 'too'
    }
    
    # Count word frequencies
    word_freq = {}
    for word in words:
        if word not in stop_words and len(word) > 3:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in keywords[:max_keywords]]

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple text similarity using keyword overlap.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    keywords1 = set(extract_keywords(text1, 20))
    keywords2 = set(extract_keywords(text2, 20))
    
    if not keywords1 and not keywords2:
        return 1.0
    
    if not keywords1 or not keywords2:
        return 0.0
    
    intersection = keywords1.intersection(keywords2)
    union = keywords1.union(keywords2)
    
    return len(intersection) / len(union)

def hash_content(content: str) -> str:
    """
    Generate a hash for content (useful for deduplication).
    
    Args:
        content: Content to hash
        
    Returns:
        SHA-256 hash string
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def safe_json_load(filepath: Path, default: Any = None) -> Any:
    """
    Safely load JSON from file with error handling.
    
    Args:
        filepath: Path to JSON file
        default: Default value if loading fails
        
    Returns:
        Loaded data or default value
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        return default

def safe_json_save(data: Any, filepath: Path) -> bool:
    """
    Safely save data to JSON file with error handling.
    
    Args:
        data: Data to save
        filepath: Path to save file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        return True
    except (PermissionError, OSError) as e:
        print(f"Error saving JSON to {filepath}: {e}")
        return False

def create_backup_filename(original_path: Path) -> Path:
    """
    Create a backup filename with timestamp.
    
    Args:
        original_path: Original file path
        
    Returns:
        Backup file path
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = original_path.stem
    suffix = original_path.suffix
    
    backup_name = f"{stem}_backup_{timestamp}{suffix}"
    return original_path.parent / backup_name

def ensure_directory(path: Path) -> bool:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except PermissionError:
        return False

def clean_old_files(directory: Path, max_age_days: int = 30, pattern: str = "*") -> int:
    """
    Clean old files from a directory.
    
    Args:
        directory: Directory to clean
        max_age_days: Maximum age in days
        pattern: File pattern to match
        
    Returns:
        Number of files deleted
    """
    if not directory.exists():
        return 0
    
    cutoff_time = datetime.now() - timedelta(days=max_age_days)
    deleted_count = 0
    
    try:
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_time:
                    file_path.unlink()
                    deleted_count += 1
    except PermissionError:
        pass
    
    return deleted_count

def validate_json_schema(data: Dict[str, Any], required_fields: List[str]) -> List[str]:
    """
    Validate JSON data against required fields.
    
    Args:
        data: Data to validate
        required_fields: List of required field names
        
    Returns:
        List of missing fields
    """
    missing_fields = []
    
    for field in required_fields:
        if field not in data:
            missing_fields.append(field)
        elif data[field] is None:
            missing_fields.append(f"{field} (null)")
    
    return missing_fields

def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two dictionaries recursively.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result

def fuzzy_match(query: str, candidates: List[str], threshold: float = 0.6) -> List[str]:
    """
    Perform fuzzy matching of query against candidates.
    
    Args:
        query: Query string
        candidates: List of candidate strings
        threshold: Minimum similarity threshold
        
    Returns:
        List of matching candidates sorted by similarity
    """
    matches = []
    
    for candidate in candidates:
        similarity = calculate_text_similarity(query, candidate)
        if similarity >= threshold:
            matches.append((candidate, similarity))
    
    # Sort by similarity (highest first)
    matches.sort(key=lambda x: x[1], reverse=True)
    
    return [match[0] for match in matches]

class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, operation_name: str = "Operation"):
        """Initialize timer with operation name."""
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and print results."""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        print(f"{self.operation_name} completed in {format_duration(duration)}")
    
    @property
    def duration(self) -> Optional[float]:
        """Get duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
