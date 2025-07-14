# File: tests/test_history.py
import pytest
import os
from src.history import History, HISTORY_FILE

def test_history_save_load():
    history = History()
    history.add_message("user", "test")
    history.save()
    assert os.path.exists(HISTORY_FILE)

    new_history = History()
    assert len(new_history.get_conversation()) == 1

    os.remove(HISTORY_FILE)
