# File: tests/test_config.py
"""Tests for configuration management."""
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.config import Config


class TestConfig:
    """Test suite for configuration management."""
    
    def test_default_config_creation(self, tmp_path):
        """Test config creation with defaults."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            config = Config()
            assert config.openai_api_key == "test-key"
            assert config.openai_model == "gpt-4o-mini"
            assert config.default_theme == "cyberpunk"
            assert config.max_history == 50
    
    def test_custom_config_values(self, tmp_path):
        """Test config with custom values."""
        env_vars = {
            "OPENAI_API_KEY": "custom-key",
            "CONTEXT7_OPENAI_MODEL": "gpt-4",
            "CONTEXT7_DEFAULT_THEME": "ocean",
            "CONTEXT7_MAX_HISTORY": "100",
            "CONTEXT7_HISTORY_FILE": str(tmp_path / "custom_history.json")
        }
        
        with patch.dict(os.environ, env_vars):
            config = Config()
            assert config.openai_api_key == "custom-key"
            assert config.openai_model == "gpt-4"
            assert config.default_theme == "ocean"
            assert config.max_history == 100
            assert config.history_file == tmp_path / "custom_history.json"
    
    def test_missing_api_key(self):
        """Test error handling for missing API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Config()
            assert "openai_api_key" in str(exc_info.value)
    
    def test_directory_creation(self, tmp_path):
        """Test automatic directory creation for data files."""
        custom_path = tmp_path / "deep" / "nested" / "history.json"
        env_vars = {
            "OPENAI_API_KEY": "test-key",
            "CONTEXT7_HISTORY_FILE": str(custom_path)
        }
        
        with patch.dict(os.environ, env_vars):
            config = Config()
            assert custom_path.parent.exists()
    
    def test_path_validation(self):
        """Test path configuration validation."""
        env_vars = {
            "OPENAI_API_KEY": "test-key",
            "CONTEXT7_HISTORY_FILE": "relative/path/history.json"
        }
        
        with patch.dict(os.environ, env_vars):
            config = Config()
            assert isinstance(config.history_file, Path)
            assert config.history_file.name == "history.json"
    
    def test_type_validation(self):
        """Test type validation for configuration values."""
        env_vars = {
            "OPENAI_API_KEY": "test-key",
            "CONTEXT7_MAX_HISTORY": "invalid"
        }
        
        with patch.dict(os.environ, env_vars):
            with pytest.raises(ValidationError):
                Config()
