# File: tests/test_themes.py
"""Tests for the theme system."""
from unittest.mock import patch

import pytest

from src.themes import (
    Theme,
    THEMES,
    get_theme,
    list_themes,
    AnimationEffects
)


class TestTheme:
    """Test suite for Theme class."""
    
    def test_theme_creation(self):
        """Test theme instantiation."""
        theme = Theme(
            name="test",
            primary="#ff0000",
            secondary="#00ff00",
            accent="#0000ff",
            background="#000000",
            text="#ffffff",
            success="#00ff00",
            warning="#ffff00",
            error="#ff0000",
            gradient_colors=["#ff0000", "#00ff00"],
            ascii_art="Test"
        )
        
        assert theme.name == "test"
        assert theme.primary == "#ff0000"
        assert len(theme.gradient_colors) == 2
    
    def test_get_gradient(self):
        """Test gradient text generation."""
        theme = THEMES["cyberpunk"]
        gradient_text = theme.get_gradient("Test")
        assert gradient_text is not None
    
    def test_panel_style(self):
        """Test panel style generation."""
        theme = THEMES["cyberpunk"]
        style = theme.get_panel_style()
        assert style.color == theme.primary
    
    def test_header_style(self):
        """Test header style generation."""
        theme = THEMES["cyberpunk"]
        style = theme.get_header_style()
        assert style.color == theme.accent


class TestThemeFunctions:
    """Test theme utility functions."""
    
    def test_get_theme_existing(self):
        """Test getting existing theme."""
        theme = get_theme("cyberpunk")
        assert theme.name == "Cyberpunk"
    
    def test_get_theme_nonexistent(self):
        """Test getting non-existent theme returns default."""
        theme = get_theme("nonexistent")
        assert theme.name == "Cyberpunk"
    
    def test_list_themes(self):
        """Test theme listing."""
        themes = list_themes()
        assert "cyberpunk" in themes
        assert "ocean" in themes
        assert "forest" in themes
        assert "sunset" in themes


class TestAnimationEffects:
    """Test animation utilities."""
    
    def test_typing_effect(self):
        """Test typing effect (simplified for testing)."""
        # This is more of an integration test
        # We'll test the AnimationEffects class structure
        effects = AnimationEffects()
        assert hasattr(effects, 'typing_effect')
        assert hasattr(effects, 'particle_loader')
        assert hasattr(effects, 'pulse_text')
    
    def test_particle_loader(self):
        """Test particle loader existence."""
        effects = AnimationEffects()
        assert callable(effects.particle_loader)
    
    def test_pulse_text(self):
        """Test pulse text generation."""
        effects = AnimationEffects()
        theme = THEMES["cyberpunk"]
        pulsed = effects.pulse_text("Test", theme)
        assert pulsed is not None


class TestThemeValidation:
    """Test theme data validation."""
    
    def test_all_themes_complete(self):
        """Test that all themes have required attributes."""
        for name, theme in THEMES.items():
            assert theme.name
            assert theme.primary.startswith('#')
            assert theme.secondary.startswith('#')
            assert theme.accent.startswith('#')
            assert theme.background.startswith('#')
            assert theme.text.startswith('#')
            assert theme.success.startswith('#')
            assert theme.warning.startswith('#')
            assert theme.error.startswith('#')
            assert len(theme.gradient_colors) >= 2
            assert theme.ascii_art
