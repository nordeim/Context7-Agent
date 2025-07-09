# File: src/data/bookmarks.py
"""
Bookmark management for Context7 Explorer.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import threading


@dataclass
class Bookmark:
    """Document bookmark."""
    doc_id: str
    title: str
    path: str
    timestamp: datetime
    tags: List[str]
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "path": self.path,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "notes": self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Bookmark":
        """Create from dictionary."""
        return cls(
            doc_id=data["doc_id"],
            title=data["title"],
            path=data["path"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            tags=data.get("tags", []),
            notes=data.get("notes", "")
        )


class BookmarkManager:
    """Manages document bookmarks."""
    
    def __init__(self, bookmarks_file: Path):
        self.bookmarks_file = bookmarks_file
        self._lock = threading.Lock()
        self._ensure_file()
        self._bookmarks: Dict[str, Bookmark] = {}
        self._load_bookmarks()
    
    def _ensure_file(self):
        """Ensure bookmarks file exists."""
        self.bookmarks_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.bookmarks_file.exists():
            self._save_bookmarks({})
    
    def _load_bookmarks(self):
        """Load bookmarks from file."""
        try:
            with open(self.bookmarks_file, 'r') as f:
                data = json.load(f)
                self._bookmarks = {
                    k: Bookmark.from_dict(v) 
                    for k, v in data.items()
                }
        except Exception:
            self._bookmarks = {}
    
    def _save_bookmarks(self, bookmarks: Dict[str, Dict[str, Any]]):
        """Save bookmarks to file."""
        with open(self.bookmarks_file, 'w') as f:
            json.dump(bookmarks, f, indent=2)
    
    def add_bookmark(
        self,
        doc_id: str,
        title: str,
        path: str,
        tags: List[str] = None,
        notes: str = ""
    ) -> Bookmark:
        """Add a bookmark."""
        with self._lock:
            bookmark = Bookmark(
                doc_id=doc_id,
                title=title,
                path=path,
                timestamp=datetime.now(),
                tags=tags or [],
                notes=notes
            )
            
            self._bookmarks[doc_id] = bookmark
            
            # Save to file
            self._save_bookmarks({
                k: v.to_dict() 
                for k, v in self._bookmarks.items()
            })
            
            return bookmark
    
    def remove_bookmark(self, doc_id: str) -> bool:
        """Remove a bookmark."""
        with self._lock:
            if doc_id in self._bookmarks:
                del self._bookmarks[doc_id]
                
                # Save to file
                self._save_bookmarks({
                    k: v.to_dict() 
                    for k, v in self._bookmarks.items()
                })
                
                return True
            return False
    
    def get_bookmark(self, doc_id: str) -> Optional[Bookmark]:
        """Get a specific bookmark."""
        with self._lock:
            return self._bookmarks.get(doc_id)
    
    def is_bookmarked(self, doc_id: str) -> bool:
        """Check if document is bookmarked."""
        with self._lock:
            return doc_id in self._bookmarks
    
    def get_all(self) -> List[Bookmark]:
        """Get all bookmarks."""
        with self._lock:
            return list(self._bookmarks.values())
    
    def get_by_tag(self, tag: str) -> List[Bookmark]:
        """Get bookmarks by tag."""
        with self._lock:
            return [
                b for b in self._bookmarks.values()
                if tag in b.tags
            ]
    
    def update_notes(self, doc_id: str, notes: str):
        """Update bookmark notes."""
        with self._lock:
            if doc_id in self._bookmarks:
                self._bookmarks[doc_id].notes = notes
                
                # Save to file
                self._save_bookmarks({
                    k: v.to_dict() 
                    for k, v in self._bookmarks.items()
                })
