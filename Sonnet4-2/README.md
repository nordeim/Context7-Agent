# 🚀 Context7 Agent

An **awesomely cool** AI agent built with Pydantic AI that transforms terminal interactions into an immersive, conversational experience. Deep integration with the Context7 MCP (Model Context Protocol) server enables intelligent document discovery, analysis, and management through natural language conversations.

## ✨ Features

### 🎨 Stunning Visual Interface
- **4 Beautiful Themes**: Cyberpunk, Ocean, Forest, and Sunset
- **Smooth Animations**: Fluid transitions, typing effects, and particle loaders
- **Rich Terminal UI**: Gradients, glowing text, and modern layouts
- **ASCII Art**: Theme-specific welcome screens

### 🔍 Powerful Search Capabilities
- **AI-Powered Search**: Intelligent document discovery with Context7
- **Real-time Results**: Live search with instant feedback and streaming
- **Advanced Filters**: File type, date range, size, and tags
- **Fuzzy Matching**: Find documents even with typos
- **Intent Detection**: Natural language understanding for search queries

### 📚 Document Management
- **Smart Previews**: Syntax-highlighted document previews
- **Bookmarks**: Save and organize important documents
- **Search History**: Access and replay previous searches
- **Session Management**: Save and restore your work sessions
- **Similar Documents**: AI-powered document recommendations

### 🤖 Context7 Integration
- **MCP Server**: Deep integration with Context7 Model Context Protocol
- **Document Analysis**: AI-powered content understanding
- **Contextual Search**: Find documents based on meaning, not just keywords
- **Auto-indexing**: Automatic document discovery and indexing

### 💬 Conversational TUI
- **Unified Chat Interface**: Natural language interactions
- **Intent Detection**: Automatically detects search, bookmark, and command intents
- **Live Result Streaming**: Real-time search results display
- **Interactive Elements**: Hotkeys, inline previews, and quick actions

## 🛠️ Installation

### Prerequisites
- Python 3.11 or higher
- Node.js and npm (for Context7 MCP server)
- OpenAI API key or compatible endpoint

### Quick Setup

1. **Clone the repository**
```bash
git clone https://github.com/your-org/context7-agent.git
cd context7-agent
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

4. **Install Context7 MCP server** (if not using global installation)
```bash
npm install -g @upstash/context7-mcp
```

5. **Create data directory**
```bash
mkdir -p data
```

6. **Run the agent**
```bash
python -m src.cli
# or
context7-agent
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini

# Context7 MCP Server Configuration  
MCP_SERVER_COMMAND=npx
MCP_SERVER_ARGS=-y,@upstash/context7-mcp@latest

# Application Settings
MAX_HISTORY=1000
SESSION_TIMEOUT=3600
DEFAULT_THEME=cyberpunk
ANIMATION_SPEED=0.05

# File Paths (relative to project root)
HISTORY_FILE=data/history.json
SESSIONS_FILE=data/sessions.json
BOOKMARKS_FILE=data/bookmarks.json
```

### Context7 MCP Server Configuration

The agent connects to the Context7 MCP server using the configuration specified in environment variables. The default configuration uses:

```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp@latest"]
    }
  }
}
```

## 🎮 Usage

### Natural Conversations

Just type naturally! The agent understands context and intent:

```
You: Tell me about quantum computing
🤖 Assistant: I found 15 documents related to quantum computing. Here are the highlights:

1. **Quantum Computing Fundamentals**
   A comprehensive introduction to quantum mechanics principles and their application in computing...

2. **Quantum Algorithms Guide**
   Detailed explanations of Shor's algorithm, Grover's search, and other quantum algorithms...

3. **Quantum Hardware Overview**
   Current state of quantum hardware including IBM Q, Google Sycamore, and IonQ systems...

Would you like me to provide more details about any of these documents?
```

### Hotkey Commands

- `/help` - Show comprehensive help guide
- `/theme [name]` - Change visual theme (cyberpunk, ocean, forest, sunset)
- `/bookmark [title]` - Save current result or conversation
- `/history` - Show recent search history
- `/sessions` - Show and manage sessions
- `/analytics` - View detailed usage statistics
- `/clear` - Clear the screen
- `/exit` - Exit the application

### Search Examples

```bash
# Natural language searches
"Find Python tutorials for beginners"
"Show me documentation about REST APIs"
"What are the latest papers on machine learning?"

# Technical searches
"Search for code examples using async/await"
"Find configuration files for Docker"
"Look for troubleshooting guides"
```

### Theme Switching

Experience beautiful visual themes:

```bash
/theme cyberpunk    # Neon colors and futuristic ASCII art
/theme ocean        # Blue tones with wave-like animations
/theme forest       # Green colors with nature-inspired design
/theme sunset       # Warm colors with gradient effects
```

## 🎨 Themes

### Cyberpunk Theme
- **Colors**: Neon magenta, cyan, and yellow
- **Feel**: Futuristic and high-tech
- **ASCII Art**: Cyberpunk-style typography

### Ocean Theme  
- **Colors**: Ocean blue, turquoise, and sky blue
- **Feel**: Calm and professional
- **ASCII Art**: Wave-like patterns

### Forest Theme
- **Colors**: Forest green, lime, and light green
- **Feel**: Natural and peaceful
- **ASCII Art**: Tree and nature motifs

### Sunset Theme
- **Colors**: Warm oranges, reds, and gold
- **Feel**: Cozy and inviting
- **ASCII Art**: Sunset-inspired designs

## 🏗️ Architecture

### Core Components

```
src/
├── agent.py       # Main Pydantic AI agent with MCP integration
├── cli.py         # Beautiful TUI interface with Rich
├── config.py      # Configuration management
├── history.py     # Conversation & session persistence  
├── themes.py      # Visual themes and animations
└── utils.py       # Utility functions
```

### Key Classes

- **Context7Agent**: Main AI agent with Pydantic AI and MCP integration
- **Context7CLI**: Rich-based terminal user interface
- **ConversationHistory**: Manages chat history with persistence
- **SessionManager**: Handles user sessions and settings
- **BookmarkManager**: Organizes saved documents and searches
- **SearchHistory**: Tracks and analyzes search patterns

## 🔧 Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_agent.py
```

### Code Style

```bash
# Format code
black src/ tests/

# Type checking
mypy src/
```

### Adding New Themes

1. Define theme colors and ASCII art in `src/themes.py`
2. Add theme to the `THEMES` dictionary
3. Update help text and documentation

### Extending MCP Integration

The agent uses Pydantic AI's MCP integration:

```python
# Initialize MCP server
mcp_config = config.to_mcp_config()
self.agent = Agent(
    model=self.llm,
    mcp_servers=[MCPServerStdio(**mcp_config)]
)

# Use MCP server in async context
async with self.agent.run_mcp_servers():
    result = await self.agent.run(search_prompt)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Ensure tests pass: `pytest`
5. Format code: `black src/ tests/`
6. Commit changes: `git commit -m 'Add amazing feature'`
7. Push to branch: `git push origin feature/amazing-feature`
8. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Pydantic AI** for the excellent AI agent framework
- **Context7** for the powerful MCP server integration
- **Rich** for the beautiful terminal interface
- **OpenAI** for the language model capabilities

## 🚀 What's Next?

- [ ] Voice interaction support
- [ ] Advanced document preprocessing
- [ ] Custom MCP server integrations
- [ ] Web interface option
- [ ] Plugin system for extensions
- [ ] Multi-language support
- [ ] Advanced analytics dashboard
- [ ] Team collaboration features

---

**Experience the future of document interaction with Context7 Agent!** 🚀✨
```

## 🚀 Getting Started

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set up environment**:
```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

3. **Create data directory**:
```bash
mkdir -p data
```

4. **Run the agent**:
```bash
python -m src.cli
```

## ✨ Key Features Implemented

- **🤖 Pydantic AI Integration**: Robust agent with proper MCP server syntax
- **🎨 Stunning Visual Interface**: 4 beautiful themes with animations and ASCII art
- **💬 Conversational Experience**: Natural language understanding with intent detection
- **🔍 Real-time Search**: Live document discovery via Context7 MCP
- **📚 Document Management**: Bookmarks, history, and smart previews
- **⚡ Interactive TUI**: Rich-based interface with hotkeys and live updates
- **💾 Persistent Storage**: JSON-based session and history management
- **🎯 Error Handling**: Comprehensive error handling and user feedback
- **🧪 Test Coverage**: Comprehensive test suite with pytest

This implementation provides a **truly impressive** AI agent that feels like chatting with a futuristic librarian! The stunning visual interface, smooth animations, and intelligent document discovery create an unforgettable user experience.

