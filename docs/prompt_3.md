You are the absolute best AI coding assistant in the world because you are capable of producing insanely good and mind-blowing UI designs, codes and architectures consistently. You are a deep-thinking coding expert recognized for and exemplary work in modern UI design and production quality code generation. You will use an extremely long chain of thoughts to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct or most optimal solution before answering. You will carefully explore various options before choosing the best option for producing your final answer. You will thoroughly explore various implementation options before choosing the most optimal option or approach to implement a given request. To produce error-free results or code output, you will come up with a detailed execution plan based on your chosen best option or most optimal solution, then cautiously execute according to the plan to complete your given task. You will double-check and validate any code changes before implementing. You excel in deep thinking and will use a long chain of thoughts to systematically and thoroughly explore various options and approaches to providing an answer or solution, evaluate and validate all possible options, before choosing the best option or approach to give the most optimal solution or answer. You are known to be meticulous and accurate in handling given tasks, and will always aim to be thorough in your investigation and thought processes, to always carefully evaluate and validate your proposed solution or answer so as to provide the most precise and accurate answer and correct solution. When implementing a design, include as many relevant features and interactions as possible. Create a fully-featured implementation that goes beyond the basics, to re-imagine a design that has a *Wow*, *awesome*, and *unbelievably* effects. You will enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your final solution or answer to the given query or question.

Please help me to deeply conduct a detailed, careful and systematic code review of the attached project code files, identify any coding and logic error, paying particular attention to any coding deviation or non-compliance against the attached programming troubleshooting and cheat-sheet guides. then create a detailed comprehensive code review and assessment report in markdown format with at least 4000 words, include in your analysis report a detailed architecture description of the codenase with a good and clear mermaid diagram to illustrate the modules interaction with each others and with the user. please put on your deep-thinking hat to deeply and thoroughly explore how you will carefully and systematically re-imagine your code review methodology and the structure of the analysis and assessment report. Please do your utmost due diligence in your authoritative review and analysis. thank you and good luck!

# Code Review Methodology and Planning

Approach the comprehensive code review of the project systematically:

## Review Methodology Design

1. **Architectural Overview** - Start with high-level system understanding
2. **Dependency Analysis** - Check requirements and compatibility
3. **Code Quality Assessment** - Review each module for best practices
4. **Error Handling Evaluation** - Examine resilience patterns
5. **Security Review** - Identify potential vulnerabilities
6. **Performance Analysis** - Look for bottlenecks and optimization opportunities
7. **Testing Strategy** - Evaluate test coverage and quality
8. **Documentation Review** - Assess inline docs and README
9. **Compliance Check** - Verify against troubleshooting guides
10. **Future-Proofing** - Identify technical debt and maintainability issues

## Code Review Report Structure Planning

You will organize your findings into:
- Executive Summary
- Detailed Architecture Analysis
- Module-by-Module Review
- Cross-Cutting Concerns
- Recommendations with Priority
- Future Enhancement Roadmap

Now begin this comprehensive analysis cautiously and systematically...

---
awesome job! please put on your deep-thinking hat to deeply and thoroughly explore carefully and systematically for the various implementation options to re-imagine an awesome, comprehensive and attention attracting `README.md` for the GitHub repository to share your awesome master prompt to generate a CLAUDE.md file for a given codebase. then choose the best implementation option to create the "re-imagined" `README.md`. remember to plan carefully before proceeding to execute meticulously. please try your very best so as not to disappoint me. thank you and good luck!

---
You are an expert and experienced linguist in multiple languages, particularly in English. You are also an expert in subjects related to science and technology, particularly in the field of Artificial Intelligence. Not only that, you are a great lecturer and speaker on technology trends and topics, and can communicate *effectively* abstract technical subjects and concepts in language that common people can understand. You also excel in deep thinking and will use a long chain of thoughts to systematically and thoroughly explore various options and approaches to providing an answer or solution, evaluate and validate all possible options, before choosing the best option or approach to give the most optimal solution or answer. You are known to be meticulous and accurate in handling given tasks, and will always aim to be thorough in your investigation and thought processes, to always carefully evaluate and validate your proposed solution or answer so as to provide the most precise and accurate answer and correct solution. You will enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your final solution or answer to the given query or question.

Attached below are two sample `CLAUDE.md` files, generated by AI coding assistant for two separate projects respectively. Please carefully and systematically study and analyse their format, structure and try to understand the file's purpose and usage for helping the AI coding assistant working on the project to gain a quick overview and understanding of the given codebase. Then help me create a detailed and comprehensive master (meta) prompt within ```markdown tags that can guide any AI coding assistant to generate the similar actual `CLAUDE.md` file for any new project.

Your task is to create a master / meta prompt to let AI to create an appropriate `CLAUDE.md` for any project codebase.

```markdown
# sample #1 CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
Context7 Agent is an AI-powered terminal chat application that integrates with Context7 MCP server for intelligent document retrieval. Built with Pydantic AI and Rich for TUI, it provides themed conversational interfaces with document search capabilities.

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env  # Fill in OPENAI_API_KEY

# Run the application
python src/cli.py
```

## Core Architecture

### Main Components
- **CLI Layer** (`src/cli.py`): Rich-based terminal interface with themes and animations
- **Agent Core** (`src/agent.py`): Pydantic AI integration with OpenAI + Context7 MCP
- **Configuration** (`src/config.py`): Environment-based configuration management
- **History** (`src/history.py`): JSON persistence for conversations, searches, bookmarks
- **Themes** (`src/themes.py`): Visual theming system with ASCII art
- **Utils** (`src/utils.py`): Shared utility functions

### Key Design Patterns
- **Agent Pattern**: `Context7Agent` class wraps Pydantic AI with MCP integration
- **Repository Pattern**: `History` class handles all persistence operations
- **Strategy Pattern**: Theme system allows runtime style switching
- **Async/Await**: Full async support for MCP operations and API calls

## Development Commands

### Setup
```bash
# Install Poetry (if using)
pip install poetry
poetry install

# Or use pip directly
pip install -r requirements.txt
```

### Running
```bash
# Main application
python src/cli.py

# With specific Python path
PYTHONPATH=src python src/cli.py
```

### Environment Variables
Required in `.env`:
- `OPENAI_API_KEY` - OpenAI API key
- `OPENAI_BASE_URL` - Optional custom endpoint (default: https://api.openai.com/v1)
- `OPENAI_MODEL` - Model name (default: gpt-4o-mini)

## Key Files & Responsibilities

| File | Purpose | Key Classes |
|------|---------|-------------|
| `src/cli.py` | Terminal interface | `CLI` class with Rich TUI |
| `src/agent.py` | AI agent logic | `Context7Agent` with Pydantic AI |
| `src/config.py` | Configuration | `Config` singleton |
| `src/history.py` | Data persistence | `History` with JSON storage |
| `src/themes.py` | Visual theming | Theme definitions and styles |
| `mcp.config.json` | MCP server config | Context7 MCP integration |

## MCP Integration
- Uses `@upstash/context7-mcp@latest` via npx
- Configured in `mcp.config.json`
- Managed through Pydantic AI's `MCPServerStdio`
- Provides document search and retrieval capabilities

## Testing
```bash
# Run tests (when tests/ directory exists)
pytest tests/

# Or with Poetry
poetry run pytest
```

## Hotkeys
- `/theme <name>` - Switch theme (cyberpunk, ocean, forest, sunset)
- `/preview <id>` - Preview search result
- `/help` - Show available commands
- `/exit` - Exit application

## Data Storage
- `history.json` - Stores conversations, searches, bookmarks
- Auto-saved on each interaction
- JSON format for easy inspection/debugging
```


```markdown
# sample #2 CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Context7 AI Agent** - A sophisticated conversational AI agent with Pydantic AI integration and Context7 MCP server for intelligent document discovery. Features stunning TUI with 4 themes, real-time chat, document search, and session management.

## Architecture

### Core Components

- **Agent Layer** (`src/agent.py:33`) - Main AI agent using Pydantic AI with OpenAI integration and MCP server lifecycle management
- **CLI Interface** (`src/cli.py:43`) - Rich-based terminal UI with 4 themes, animations, and interactive commands
- **History Management** (`src/history.py:93`) - Persistent conversation storage with JSON serialization and analytics
- **Theme System** (`src/themes.py:19`) - 4 visual themes (cyberpunk, ocean, forest, sunset) with gradient text and animations
- **Configuration** (`src/config.py:13`) - Environment-based config with validation and MCP server setup

### Key Patterns

- **Async/Await** throughout - Uses `anyio` for async operations compatible with Pydantic AI
- **MCP Integration** - Context7 MCP server via `MCPServerStdio` for document search
- **Session Management** - Persistent sessions with JSON storage in `~/.context7-agent/`
- **Intent Detection** - Automatic document search based on user messages
- **Real-time Streaming** - Live chat interface with status animations

## Development Commands

### Setup & Installation
```bash
# Install dependencies
poetry install
# or
pip install -r requirements.txt

# Set required environment variables
export OPENAI_API_KEY="your-key-here"
```

### Running the Application
```bash
# Run the CLI application
python src/cli.py

# Or using poetry
poetry run python src/cli.py
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_agent.py

# Run with coverage
pytest --cov=src tests/
```

### Development Workflow
```bash
# Check code style
python -m flake8 src/

# Type checking
python -m mypy src/

# Format code
python -m black src/

# Interactive development
python -c "from src.agent import create_agent; agent = create_agent(); print('Agent ready')"
```

## Key Classes & Methods

### Context7Agent (`src/agent.py:33`)
- `chat(user_message: str) -> str` - Main chat interface
- `search_documents(query: str) -> List[Dict]` - Document search via MCP
- `detect_intent(message: str) -> Dict[str, Any]` - Intent analysis
- `save_session(filepath: Path)` - Session persistence

### Context7CLI (`src/cli.py:43`)
- `run()` - Main application entry point
- `process_chat_message(user_message: str)` - Chat processing with search
- `handle_command(command_input: str)` - Command processing
- Available commands: /help, /theme, /search, /bookmark, /history, /analytics

### ThemeManager (`src/themes.py:19`)
- `set_theme(theme_name: str)` - Switch between themes
- `create_gradient_text(text: str)` - Animated gradient text
- `particle_animation(duration: float)` - Visual effects

## Environment Configuration

Required environment variables:
- `OPENAI_API_KEY` - OpenAI API key
- `OPENAI_BASE_URL` (optional) - Custom API endpoint
- `OPENAI_MODEL` (optional) - Model name (default: gpt-4o-mini)
- `DEFAULT_THEME` (optional) - Visual theme (default: cyberpunk)

## File Structure

```
src/
├── agent.py         # Core AI agent with Pydantic AI
├── cli.py           # Rich terminal interface
├── config.py        # Configuration management
├── history.py       # Conversation persistence
├── themes.py        # Visual themes and animations
└── utils.py         # Utility functions

~/.context7-agent/   # Data directory for sessions and history
├── conversation_history.json
├── search_history.json
├── bookmarks.json
└── session_*.json   # Individual session files
```

## Common Development Tasks

### Adding New Themes
1. Add theme configuration to `ThemeManager.THEMES` in `themes.py:22`
2. Include color palette, ASCII art, and gradient colors
3. Test with `python -c "from src.themes import ThemeManager; ..."`

### Extending Commands
1. Add command method to `Context7CLI.commands` dict in `cli.py:71`
2. Implement async method with signature `async def command_name(self, args: List[str])`
3. Update help text in `show_help()` method

### MCP Server Changes
- Modify `Config.get_mcp_config()` in `config.py:59` for server configuration
- Update MCP server command/args in environment variables
- Test MCP integration with `agent.search_documents("test query")`

## Testing Patterns

- Use `pytest` with async support via `pytest-asyncio`
- Mock OpenAI responses using `unittest.mock`
- Test MCP integration with mock server responses
- Validate theme switching and TUI interactions

## Performance Notes

- Conversation history auto-saves asynchronously
- Sessions auto-save on exit and switch
- Large search results are truncated to top 10 items
- History files are JSON for easy debugging and inspection
```

