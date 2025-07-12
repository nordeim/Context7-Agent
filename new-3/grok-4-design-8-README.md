# Context7 Agent

![Context7 Agent Logo](https://via.placeholder.com/800x200?text=Context7+Agent) <!-- Placeholder; replace with actual logo if available -->

An awesomely cool and user-friendly AI agent built with Pydantic AI, deeply integrated with the Context7 MCP (Model Context Protocol) server. This agent transforms terminal interactions into an immersive, conversational experience, allowing users to chat naturally about subjects while the AI intelligently searches for and presents relevant documents via MCP. The re-imagined design features a dynamic Terminal User Interface (TUI) with stunning themes, smooth animations, live result streaming, and interactive elements like hotkeys and inline previews.

Powered by OpenAI for chat and intent detection, and Node.js for the MCP server, this agent excels in contextual document discovery, analysis, and management. Whether you're exploring "quantum computing" or bookmarking findings, it feels like chatting with a futuristic librarian!

## Features

- **Stunning Visual Interface**: 4 beautiful themes (Cyberpunk, Ocean, Forest, Sunset) with gradients, glowing text, smooth animations (e.g., typing effects, particle loaders), rich layouts (panels, tables), and theme-specific ASCII art welcome screens.
- **Powerful Search Capabilities**: AI-powered, real-time document discovery via Context7 MCP, with fuzzy matching, advanced filters (file type, date range, size, tags), and search analytics.
- **Document Management**: Smart syntax-highlighted previews, bookmarks, search history, session management, and AI-recommended similar documents.
- **Context7 Integration**: Full MCP server support for contextual searches (based on meaning, not just keywords), document analysis, auto-indexing, and quantum-mode enhancements.
- **Conversational TUI**: Unified chat interface with intent detection – discuss a subject, and the agent automatically searches MCP, streams results live, and offers interactions (e.g., /preview 1).
- **Additional Perks**: Hotkeys for commands (/theme, /bookmark, /analytics), error alerts, session auto-save/load, and performance optimizations with async operations.

## Current Features Implemented

The following features are partially or fully implemented in the current codebase, based on the re-imagined design:

- **Fully Implemented**:
  - Conversational chat with OpenAI integration (e.g., natural language responses).
  - Intent detection for subject-based searches (e.g., "Tell me about AI ethics" triggers MCP query).
  - Real-time MCP searches with live streaming results and fuzzy matching.
  - Dynamic TUI layout (split-screen: header, chat, results, footer) using Rich.
  - Theme switching with animations and ASCII art.
  - Animations (typing simulation, particle loaders) for fluid UX.
  - Document previews (syntax-highlighted), bookmarks, and similar recommendations.
  - JSON-based persistence for conversations, searches, bookmarks, and sessions.
  - Hotkeys (/help, /exit, /analytics, etc.) within the chat.
  - Error handling with styled alerts and retries for MCP connections.

- **Partially Implemented**:
  - Search analytics (basic tracking and dashboard; lacks advanced visualizations like charts).
  - Advanced filters (CLI options exist but not fully interactive in TUI; e.g., no in-chat filter refinement).
  - Auto-indexing (initiated on MCP start, but lacks user-configurable paths).
  - Session management (auto-save/load for default session; multi-session support is basic).

These implementations reflect recent code changes, such as the shift to a Live-based TUI in `cli.py` (with Layout for split-screen and async handling for real-time updates) and enhanced MCP integration in `agent.py` (e.g., `detect_intent_and_search` for conversational flows).

## Roadmap

### Immediate Goals (Next 1-3 Months)
- Enhance search analytics with visual charts (e.g., using Rich text-based graphs) and export options.
- Add interactive filter refinement in the TUI (e.g., "refine by date: 2023").
- Improve MCP error recovery with auto-reconnect and user notifications.
- Expand tests to 80% coverage, including TUI simulations.
- Integrate caching for frequent searches to boost performance.

### Long-Term Goals (3-12 Months)
- Add voice-like input simulation (e.g., via speech-to-text libs) for hands-free use.
- Support multi-user sessions and cloud syncing (e.g., via a real database like PostgreSQL).
- Extend MCP features with custom plugins (e.g., user-defined indexing rules).
- Develop a web-based companion interface for non-terminal users.
- Open-source contributions: Add CI/CD pipelines and community-driven themes.

This roadmap is flexible and prioritizes user feedback.

## Project Codebase File Hierarchy

The project follows a clean, standard Python structure for maintainability. Below is the file hierarchy:

```
context7-agent/
├── src/                  # Core source code
│   ├── __init__.py       # Package initializer
│   ├── agent.py          # AI agent logic and MCP integration
│   ├── cli.py            # Interactive TUI and entry point
│   ├── config.py         # Environment configuration and validation
│   ├── history.py        # Data persistence (JSON-based)
│   ├── themes.py         # Theme definitions and styles
│   └── utils.py          # Utility functions (e.g., fuzzy matching, JSON helpers)
├── tests/                # Unit tests
│   ├── __init__.py
│   ├── test_agent.py     # Tests for agent and MCP logic
│   └── test_history.py   # Tests for persistence
├── .env.example          # Example environment variables
├── .gitignore            # Git ignore rules
├── README.md             # This documentation file
├── requirements.txt      # Python dependencies
└── pyproject.toml        # Build and dependency management (Poetry-compatible)
```

This hierarchy separates concerns: `src/` for logic, `tests/` for validation, and root files for setup.

## File Descriptions

Here’s a detailed list and description of each file in the hierarchy:

- **src/__init__.py**: Empty file that marks `src/` as a Python package, enabling imports.
- **src/agent.py**: Core module for the AI agent. Handles OpenAI chat, intent detection, MCP integration (via async subprocess), searches, and document analysis. Recent changes include the `detect_intent_and_search` method for conversational MCP queries.
- **src/cli.py**: Main entry point and TUI implementation. Manages the Rich-based layout, live updates, animations, input handling, and hotkeys. Updated in the re-imagined design to use `Live` and `Layout` for immersive interactions.
- **src/config.py**: Loads and validates environment variables (e.g., OpenAI keys, MCP config). Ensures error-free startup.
- **src/history.py**: Manages JSON storage for conversations, searches, bookmarks, and sessions. Includes methods like `save_conversation` and `get_analytics`.
- **src/themes.py**: Defines theme dictionaries with colors, styles, gradients, and ASCII art. Supports dynamic switching.
- **src/utils.py**: Helper functions, such as fuzzy matching (`rapidfuzz`) and JSON load/save operations.
- **tests/__init__.py**: Package initializer for tests.
- **tests/test_agent.py**: Pytest tests for agent methods (e.g., chat, search mocks).
- **tests/test_history.py**: Tests for persistence operations (e.g., save/load validation).
- **.env.example**: Template for environment variables (e.g., `OPENAI_API_KEY`).
- **.gitignore**: Ignores temporary files, virtualenvs, etc., to keep the repo clean.
- **README.md**: This file – comprehensive project documentation.
- **requirements.txt**: Lists dependencies (e.g., `rich`, `pydantic-ai`, `openai`).
- **pyproject.toml**: Configures project metadata and dependencies (e.g., for Poetry builds).

These files reflect the modular design, with recent updates focusing on TUI enhancements in `cli.py` and intent detection in `agent.py`.

## Flowchart Diagram

The following Mermaid flowchart illustrates interactions between files/modules and the user. It shows data flow from user input to MCP searches and back to the TUI.

```mermaid
graph TD
    User[User] -->|Inputs message/hotkey| CLI[src/cli.py: TUI Loop]
    CLI -->|Parses & Handles| HandleInput[handle_input Function]
    HandleInput -->|Commands e.g. /theme| Themes[src/themes.py: Update Theme]
    HandleInput -->|Chat Message| Agent[src/agent.py: detect_intent_and_search]
    Agent -->|Intent Detection| OpenAI[OpenAI Provider]
    Agent -->|Search Query| MCP[Context7MCPIntegration: Async Search]
    MCP -->|Subprocess| MCPServer[External MCP Server (Node.js)]
    MCPServer -->|Documents| MCP
    MCP -->|Results| Agent
    Agent -->|Response/Results| HandleInput
    HandleInput -->|Updates Layout| RichUI[Rich Layout & Live]
    RichUI -->|Renders| CLI
    CLI -->|Displays| User
    History[src/history.py: Save/Load] <-->|Persistence| HandleInput
    Config[src/config.py: Env Vars] -->|Configs| Agent
    Utils[src/utils.py: Helpers] -->|e.g. Fuzzy| Agent
    Themes -->|Styles| RichUI
```

**Explanation**: User inputs flow through `cli.py` to `agent.py` for processing. MCP interactions are async, with results updating the TUI via Rich. Support files (e.g., `themes.py`) provide data on demand.

## Prerequisites

- Python 3.11 or higher
- Node.js and npm (for Context7 MCP server)
- OpenAI API key (or compatible endpoint)
- Optional: Docker for containerized deployment

## Deployment Guide

This guide covers cloning from GitHub, installing on a host machine (e.g., your local development or production server, interpreted here as the "POS machine" for running the code), setting up configurations, and deploying a database server via Docker (for future scalability, e.g., replacing JSON history with PostgreSQL). The app runs locally but can be containerized.

### Step 1: Clone from GitHub
1. Ensure Git is installed on your host machine.
2. Clone the repository:  
   ```
   git clone https://github.com/yourusername/context7-agent.git
   cd context7-agent
   ```

### Step 2: Install Dependencies on Host Machine
1. Create and activate a virtual environment:  
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install Python dependencies:  
   ```
   pip install -r requirements.txt
   ```
3. Install Node.js dependencies (for MCP):  
   ```
   npm install -g @upstash/context7-mcp@latest  # If not using npx
   ```

### Step 3: Setup and Configure on Host Machine
1. Copy and configure `.env`:  
   ```
   cp .env.example .env
   ```
   Edit `.env` with your values:  
   ```
   OPENAI_API_KEY=your_openai_key
   OPENAI_API_BASE=https://api.openai.com/v1
   OPENAI_MODEL=gpt-4
   ```
2. Validate config: Run `python src/config.py` (custom script if added; otherwise, app startup will validate).
3. (Optional) Configure MCP: Edit `src/config.py` for custom MCP args if needed.

### Step 4: Setup Database Server via Docker
The current app uses JSON for persistence, but for scalability (e.g., multi-user history), we can add a PostgreSQL database. Here's how to deploy it via Docker:

1. Install Docker on your host machine (download from docker.com).
2. Create a `docker-compose.yml` in the project root:  
   ```
   version: '3.8'
   services:
     db:
       immagine: postgres:14
       environment:
         POSTGRES_USER: user
         POSTGRES_PASSWORD: password
         POSTGRES_DB: context7_db
       ports:
         - "5432:5432"
       volumes:
         - db_data:/var/lib/postgresql/data

     app:
       build: .
       depends_on:
         - db
       environment:
         - DATABASE_URL=postgresql://user:password@db:5432/context7_db
       volumes:
         - .:/app
       command: python src/cli.py

   volumes:
     db_data:
   ```
3. Build and run:  
   ```
   docker-compose up -d
   ```
4. Migrate database (future step: Add SQLAlchemy in `history.py` for DB integration). Connect via `psql` for testing:  
   ```
   docker exec -it <container_id> psql -U user -d context7_db
   ```
5. For MCP in Docker: Add a service to `docker-compose.yml` for the Node.js MCP, or run it separately.

This setups a containerized environment; access the app at `localhost` via the host machine.

### Step 5: Run the App
See the User Guide below for running commands.

## User Guide

### Running the Program
1. Activate the virtual environment: `source venv/bin/activate`.
2. Launch the TUI:  
   ```
   python src/cli.py
   ```
   - This starts the interactive session with a theme picker and welcome screen.

### Using the Application
- **Starting a Session**: On launch, choose a theme (e.g., "cyberpunk"). The TUI loads with split panels.
- **Chatting and Searching**: Type messages like "What are the latest on climate change?" – the agent detects intent, queries MCP, and streams results live in the right panel.
- **Interacting with Results**: Results show numbered (e.g., 1. Title). Use hotkeys:
  - `/preview <num>`: Show syntax-highlighted preview.
  - `/bookmark <num>`: Save to bookmarks.
  - `/analytics`: Display search stats.
- **Commands**:
  - `/theme <name>`: Switch theme (cyberpunk, ocean, forest, sunset).
  - `/help`: List commands.
  - `/exit`: Quit and auto-save session.
- **Sessions**: On restart, the default session loads automatically.
- **Advanced**: For non-TUI use (legacy), add flags like `python src/cli.py --legacy` (custom if implemented).

Troubleshooting: Check `.env` for keys; restart MCP if searches fail.

## Contributing
Fork the repo, create a branch, submit PRs. Follow code style (PEP8).

## License
MIT License – see LICENSE file (add if missing).

