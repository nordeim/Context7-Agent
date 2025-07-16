# File: src/utils.py
"""
Utility functions for the Context7 Agent.
"""

from difflib import SequenceMatcher

def fuzzy_match(query: str, text: str, ratio: float = 0.7) -> bool:
    """
    Performs a more sophisticated fuzzy string match using Python's SequenceMatcher.
    It compares the similarity of two strings and returns True if the ratio
    of similarity is above the specified threshold. This is more effective than
    a simple substring check.

    For example, it can match 'AI ethics' with 'AI ethic' or 'Context 7' with 'Context7'.
    """
    return SequenceMatcher(None, query.lower(), text.lower()).ratio() >= ratio

# Add more utils here as the application grows...
