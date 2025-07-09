# File: src/data/history_manager.py
"""
Search history management for Context7 Explorer.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import threading


@dataclass
class SearchEntry:
    """Search history entry."""
    query: str
    timestamp: datetime
    results_count: int = 0
    execution_time: float = 0.0
    filters: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "timestamp": self.timestamp.isoformat(),
            "results_count": self.results_count,
            "execution_time": self.execution_time,
            "filters": self.filters or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchEntry":
        """Create from dictionary."""
        return cls(
            query=data["query"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            results_count=data.get("results_count", 0),
            execution_time=data.get("execution_time", 0.0),
            filters=data.get("filters", {})
        )


class HistoryManager:
    """Manages search history with analytics."""
    
    def __init__(self, history_file: Path):
        self.history_file = history_file
        self._lock = threading.Lock()
        self._ensure_file()
        self._cache: List[SearchEntry] = []
        self._load_history()
    
    def _ensure_file(self):
        """Ensure history file exists."""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.history_file.exists():
            self._save_history([])
    
    def _load_history(self):
        """Load history from file."""
        try:
            with open(self.history_file, 'r') as f:
                data = json.load(f)
                self._cache = [SearchEntry.from_dict(item) for item in data]
        except Exception:
            self._cache = []
    
    def _save_history(self, entries: List[Dict[str, Any]]):
        """Save history to file."""
        with open(self.history_file, 'w') as f:
            json.dump(entries, f, indent=2)
    
    def add_search(
        self, 
        query: str, 
        results_count: int = 0,
        execution_time: float = 0.0,
        filters: Optional[Dict[str, Any]] = None
    ):
        """Add a search to history."""
        with self._lock:
            entry = SearchEntry(
                query=query,
                timestamp=datetime.now(),
                results_count=results_count,
                execution_time=execution_time,
                filters=filters
            )
            
            self._cache.append(entry)
            
            # Keep only last 1000 entries
            if len(self._cache) > 1000:
                self._cache = self._cache[-1000:]
            
            # Save to file
            self._save_history([e.to_dict() for e in self._cache])
    
    def get_recent_searches(self, limit: int = 10) -> List[SearchEntry]:
        """Get recent search entries."""
        with self._lock:
            return list(reversed(self._cache[-limit:]))
    
    def get_popular_searches(self, days: int = 7, limit: int = 10) -> List[Dict[str, Any]]:
        """Get popular searches in the last N days."""
        with self._lock:
            cutoff = datetime.now().timestamp() - (days * 86400)
            recent = [e for e in self._cache if e.timestamp.timestamp() > cutoff]
            
            # Count occurrences
            query_counts = {}
            for entry in recent:
                query = entry.query.lower()
                query_counts[query] = query_counts.get(query, 0) + 1
            
            # Sort by count
            popular = sorted(
                query_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:limit]
            
            return [{"query": q, "count": c} for q, c in popular]
    
    def get_search_analytics(self) -> Dict[str, Any]:
        """Get search analytics."""
        with self._lock:
            if not self._cache:
                return {
                    "total_searches": 0,
                    "unique_queries": 0,
                    "avg_results": 0,
                    "avg_execution_time": 0
                }
            
            unique_queries = len(set(e.query.lower() for e in self._cache))
            avg_results = sum(e.results_count for e in self._cache) / len(self._cache)
            avg_time = sum(e.execution_time for e in self._cache) / len(self._cache)
            
            return {
                "total_searches": len(self._cache),
                "unique_queries": unique_queries,
                "avg_results": round(avg_results, 1),
                "avg_execution_time": round(avg_time, 3)
            }
    
    def search_history(self, query: str) -> List[SearchEntry]:
        """Search through history."""
        with self._lock:
            query_lower = query.lower()
            return [
                e for e in self._cache 
                if query_lower in e.query.lower()
            ]
    
    def clear_history(self):
        """Clear all history."""
        with self._lock:
            self._cache = []
            self._save_history([])
