# File: src/data/session_manager.py
"""
Session management for Context7 Explorer.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import threading


@dataclass
class Session:
    """Application session."""
    name: str
    timestamp: datetime
    data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            data=data["data"]
        )


class SessionManager:
    """Manages application sessions."""
    
    def __init__(self, sessions_dir: Path):
        self.sessions_dir = Path(sessions_dir)
        self._lock = threading.Lock()
        self._ensure_directory()
    
    def _ensure_directory(self):
        """Ensure sessions directory exists."""
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
    
    def save_session(self, name: str, data: Dict[str, Any]) -> Session:
        """Save a session."""
        with self._lock:
            session = Session(
                name=name,
                timestamp=datetime.now(),
                data=data
            )
            
            # Generate filename
            safe_name = "".join(c if c.isalnum() else "_" for c in name)
            filename = f"{safe_name}_{session.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            
            # Save session
            session_file = self.sessions_dir / filename
            with open(session_file, 'w') as f:
                json.dump(session.to_dict(), f, indent=2)
            
            return session
    
    def load_session(self, filename: str) -> Optional[Session]:
        """Load a specific session."""
        with self._lock:
            session_file = self.sessions_dir / filename
            
            if session_file.exists():
                try:
                    with open(session_file, 'r') as f:
                        data = json.load(f)
                        return Session.from_dict(data)
                except Exception:
                    pass
            
            return None
    
    def get_all_sessions(self) -> List[Session]:
        """Get all saved sessions."""
        with self._lock:
            sessions = []
            
            for file in self.sessions_dir.glob("*.json"):
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                        sessions.append(Session.from_dict(data))
                except Exception:
                    pass
            
            # Sort by timestamp (newest first)
            sessions.sort(key=lambda s: s.timestamp, reverse=True)
            
            return sessions
    
    def get_last_session(self) -> Optional[Session]:
        """Get the most recent session."""
        sessions = self.get_all_sessions()
        return sessions[0] if sessions else None
    
    def delete_session(self, filename: str) -> bool:
        """Delete a session."""
        with self._lock:
            session_file = self.sessions_dir / filename
            
            if session_file.exists():
                session_file.unlink()
                return True
            
            return False
    
    def cleanup_old_sessions(self, days: int = 30):
        """Remove sessions older than specified days."""
        with self._lock:
            cutoff = datetime.now().timestamp() - (days * 86400)
            
            for file in self.sessions_dir.glob("*.json"):
                if file.stat().st_mtime < cutoff:
                    file.unlink()
