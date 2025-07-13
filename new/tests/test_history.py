# File: tests/test_history.py
from pathlib import Path
from src.history import History

def test_roundtrip(tmp_path: Path):
    f = tmp_path / "hist.json"
    h = History(f)
    h.add("user", "hello")
    h2 = History(f)
    assert len(h2.messages) == 1
