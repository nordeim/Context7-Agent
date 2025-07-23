# File: tests/test_utils.py
"""Tests for utility functions."""
import asyncio
from pathlib import Path
from unittest.mock import patch, mock_open

import pytest

from src.utils import (
    ensure_directory,
    load_json_file,
    save_json_file,
    format_timestamp,
    truncate_text,
    extract_keywords,
    calculate_similarity,
    format_file_size,
    detect_file_type,
    debounce,
    create_progress_bar,
    parse_hotkey_command,
    highlight_search_terms
)


class TestFileOperations:
    """Test file operation utilities."""
    
    @pytest.mark.asyncio
    async def test_ensure_directory(self, tmp_path):
        """Test directory creation."""
        test_path = tmp_path / "deep" / "nested" / "file.txt"
        await ensure_directory(str(test_path))
        assert test_path.parent.exists()
    
    @pytest.mark.asyncio
    async def test_load_json_file_success(self, tmp_path):
        """Test successful JSON file loading."""
        test_file = tmp_path / "test.json"
        test_data = {"key": "value", "number": 42}
        
        with open(test_file, 'w') as f:
            import json
            json.dump(test_data, f)
        
        result = await load_json_file(str(test_file))
        assert result == test_data
    
    @pytest.mark.asyncio
    async def test_load_json_file_not_found(self):
        """Test loading non-existent file."""
        result = await load_json_file("/nonexistent/file.json")
        assert result == {}
    
    @pytest.mark.asyncio
    async def test_load_json_file_corrupted(self, tmp_path):
        """Test handling of corrupted JSON."""
        test_file = tmp_path / "corrupt.json"
        test_file.write_text("invalid json")
        
        result = await load_json_file(str(test_file))
        assert result == {}
    
    @pytest.mark.asyncio
    async def test_save_json_file_success(self, tmp_path):
        """Test successful JSON file saving."""
        test_file = tmp_path / "test.json"
        test_data = {"key": "value"}
        
        success = await save_json_file(str(test_file), test_data)
        assert success is True
        assert test_file.exists()
        
        with open(test_file) as f:
            import json
            assert json.load(f) == test_data
    
    @pytest.mark.asyncio
    async def test_save_json_file_error(self, tmp_path):
        """Test handling of save errors."""
        test_file = tmp_path / "readonly" / "test.json"
        
        with patch('aiofiles.open', side_effect=IOError()):
            success = await save_json_file(str(test_file), {"test": "data"})
            assert success is False


class TestTextUtilities:
    """Test text processing utilities."""
    
    def test_format_timestamp(self):
        """Test timestamp formatting."""
        result = format_timestamp(1640995200)  # 2022-01-01 00:00:00 UTC
        assert "2022-01-01" in result
    
    def test_truncate_text(self):
        """Test text truncation."""
        long_text = "This is a very long text that should be truncated"
        result = truncate_text(long_text, 20)
        assert len(result) <= 23  # 20 + "..."
        assert result.endswith("...")
    
    def test_truncate_text_short(self):
        """Test truncation with short text."""
        short_text = "Hello"
        result = truncate_text(short_text, 10)
        assert result == "Hello"
    
    def test_extract_keywords(self):
        """Test keyword extraction."""
        text = "FastAPI is a modern web framework for building APIs"
        keywords = extract_keywords(text, 3)
        assert len(keywords) <= 3
        assert all(isinstance(k, str) for k in keywords)
        assert "FastAPI" in keywords or "framework" in keywords
    
    def test_calculate_similarity(self):
        """Test text similarity calculation."""
        text1 = "hello world programming"
        text2 = "hello world coding"
        text3 = "completely different"
        
        similarity1 = calculate_similarity(text1, text2)
        similarity2 = calculate_similarity(text1, text3)
        
        assert 0 <= similarity1 <= 1
        assert 0 <= similarity2 <= 1
        assert similarity1 > similarity2
    
    def test_calculate_similarity_empty(self):
        """Test similarity with empty strings."""
        assert calculate_similarity("", "") == 1.0
        assert calculate_similarity("hello", "") == 0.0
    
    def test_format_file_size(self):
        """Test file size formatting."""
        assert format_file_size(0) == "0 B"
        assert format_file_size(500) == "500.0 B"
        assert format_file_size(1024) == "1.0 KB"
        assert format_file_size(1048576) == "1.0 MB"
        assert format_file_size(1073741824) == "1.0 GB"
    
    def test_detect_file_type(self):
        """Test file type detection."""
        assert detect_file_type("test.py") == "python"
        assert detect_file_type("document.js") == "javascript"
        assert detect_file_type("image.png") == "unknown"
        assert detect_file_type("config.yaml") == "yaml"
        assert detect_file_type("no_extension") == "unknown"
    
    @pytest.mark.asyncio
    async def test_debounce(self):
        """Test debounce decorator."""
        call_count = 0
        
        async def test_func(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return "result"
        
        debounced_func = await debounce(test_func, delay=0.1)
        
        # Call multiple times rapidly
        tasks = [debounced_func() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        assert call_count == 1  # Should only be called once
        assert all(r == "result" for r in results)
    
    def test_create_progress_bar(self):
        """Test progress bar creation."""
        bar = create_progress_bar(50, 100)
        assert "50%" in bar
        assert "█████" in bar or "─────" in bar
        
        bar_complete = create_progress_bar(100, 100)
        assert "100%" in bar_complete
        
        bar_empty = create_progress_bar(0, 100)
        assert "0%" in bar_empty
    
    def test_parse_hotkey_command(self):
        """Test hotkey command parsing."""
        assert parse_hotkey_command("/help") == ("help", "")
        assert parse_hotkey_command("/theme cyberpunk") == ("theme", "cyberpunk")
        assert parse_hotkey_command("/clear") == ("clear", "")
        assert parse_hotkey_command("normal message") == (None, "normal message")
        assert parse_hotkey_command("/") == ("", "")
    
    def test_highlight_search_terms(self):
        """Test search term highlighting."""
        text = "The FastAPI framework is fast and modern"
        terms = ["FastAPI", "fast"]
        result = highlight_search_terms(text, terms)
        
        assert "[bold yellow]FastAPI[/bold yellow]" in result
        assert "[bold yellow]fast[/bold yellow]" in result
        assert "framework" in result  # Unchanged
