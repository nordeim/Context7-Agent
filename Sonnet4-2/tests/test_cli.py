# File: tests/test_cli.py
"""Integration tests for the CLI interface."""
import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from rich.console import Console

from src.cli import Context7CLI


class TestContext7CLI:
    """Test suite for CLI interface."""
    
    def test_cli_initialization(self):
        """Test CLI initialization."""
        cli = Context7CLI()
        assert cli.console is not None
        assert cli.current_conversation == "default"
        assert cli.is_running is True
    
    @pytest.mark.asyncio
    async def test_process_message(self):
        """Test message processing through CLI."""
        cli = Context7CLI()
        
        with patch.object(cli.agent, 'chat', new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = {
                "type": "complete",
                "data": "Test response"
            }
            
            with patch.object(cli.console, 'print') as mock_print:
                with patch.object(cli.console, 'status') as mock_status:
                    await cli._process_message("test query")
        
        mock_chat.assert_called_once_with("test query", "default")
        assert mock_print.call_count > 0
    
    @pytest.mark.asyncio
    async def test_handle_command_exit(self):
        """Test exit command handling."""
        cli = Context7CLI()
        
        with patch.object(cli.console, 'print'):
            await cli._handle_command("/exit")
        
        assert cli.is_running is False
    
    @pytest.mark.asyncio
    async def test_handle_command_clear(self):
        """Test clear command handling."""
        cli = Context7CLI()
        
        with patch.object(cli.agent, 'clear_history', new_callable=AsyncMock) as mock_clear:
            with patch.object(cli.console, 'print'):
                await cli._handle_command("/clear")
        
        mock_clear.assert_called_once_with("default")
    
    def test_handle_theme_command(self):
        """Test theme switching."""
        cli = Context7CLI()
        
        with patch.object(cli.console, 'print'):
            with patch('src.cli.get_theme') as mock_get:
                mock_get.return_value.name = "ocean"
                cli._handle_theme_command(["ocean"])
        
        mock_get.assert_called_once_with("ocean")
        assert cli.current_theme.name == "ocean"
    
    def test_show_history(self):
        """Test history display."""
        cli = Context7CLI()
        
        with patch.object(cli.agent, 'get_conversations') as mock_get:
            mock_get.return_value = [
                {"id": "test", "last_message": "Test", "message_count": 5}
            ]
            with patch.object(cli.console, 'print'):
                cli._show_history()
        
        mock_get.assert_called_once()
    
    def test_show_help(self):
        """Test help display."""
        cli = Context7CLI()
        
        with patch.object(cli.console, 'print') as mock_print:
            cli._show_help()
        
        assert mock_print.call_count > 0
    
    @pytest.mark.asyncio
    async def test_handle_command_unknown(self):
        """Test handling of unknown commands."""
        cli = Context7CLI()
        
        with patch.object(cli.console, 'print') as mock_print:
            await cli._handle_command("/unknown")
        
        mock_print.assert_called()
        assert "Unknown command" in str(mock_print.call_args)
    
    def test_parse_command_with_args(self):
        """Test command parsing with arguments."""
        cli = Context7CLI()
        
        with patch.object(cli.console, 'print'):
            cli._handle_theme_command([])  # No args
            # Should show available themes
