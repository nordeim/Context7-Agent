# File: src/utils.py
"""
Utility functions for the Context7 Agent.
"""

def fuzzy_match(query: str, text: str) -> bool:
    """Simple fuzzy matching."""
    return query.lower() in text.lower()

# Add more utils...
