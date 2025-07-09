# 🚀 Context7 Document Explorer

<div align="center">

![Context7 Explorer](https://img.shields.io/badge/Context7-Explorer-ff006e?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![AI Powered](https://img.shields.io/badge/AI-Powered-8338ec?style=for-the-badge)

*An incredibly cool and beautiful CLI for intelligent document search*

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Themes](#themes) • [Contributing](#contributing)

</div>

---

## ✨ Features

### 🎨 Stunning Visual Interface
- **4 Beautiful Themes**: Cyberpunk, Ocean, Forest, and Sunset
- **Smooth Animations**: Fluid transitions and loading effects
- **Rich Terminal UI**: Gradients, glowing text, and modern layouts
- **ASCII Art**: Theme-specific welcome screens

### 🔍 Powerful Search Capabilities
- **AI-Powered Search**: Intelligent document discovery with Context7
- **Real-time Results**: Live search with instant feedback
- **Advanced Filters**: File type, date range, size, and tags
- **Fuzzy Matching**: Find documents even with typos
- **Search Analytics**: Track and analyze your search patterns

### 📚 Document Management
- **Smart Previews**: Syntax-highlighted document previews
- **Bookmarks**: Save and organize important documents
- **Search History**: Access and replay previous searches
- **Session Management**: Save and restore your work sessions
- **Similar Documents**: AI-powered document recommendations

### ⚡ Context7 Integration
- **MCP Server**: Deep integration with Context7 Model Context Protocol
- **Document Analysis**: AI-powered content understanding
- **Contextual Search**: Find documents based on meaning, not just keywords
- **Auto-indexing**: Automatic document discovery and indexing

## 🚀 Installation

### Prerequisites
- Python 3.11 or higher
- Node.js and npm (for Context7 MCP server)
- OpenAI API key or compatible endpoint

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/nordeim/Context7-Explorer
cd Context7-Explorer
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
# pip install pydantic-ai pydantic openai rich click prompt-toolkit textual aiofiles python-dotenv orjson pytest pytest-asyncio pytest-cov black isort flake8 mypy
pip install -r requirements.txt
```

4. **Set up environment**
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Run the explorer**
```bash
python main.py
```

## 🎯 Usage

### Basic Commands

```bash
# Start with default theme
python main.py

# Choose a different theme
python main.py --theme ocean

# Disable animations for faster performance
python main.py --no-animations

# Specify custom document directory
python main.py --index-path /path/to/documents
```

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `/` | Enter search mode |
| `↑/↓` | Navigate results |
| `Enter` | View document |
| `Esc` | Go back |
| `Ctrl+B` | Show bookmarks |
| `Ctrl+H` | Search history |
| `Ctrl+S` | Save session |
| `Ctrl+Q` | Quit |

### Search Syntax

- **Basic Search**: `python tutorial`
- **Exact Match**: `"exact phrase"`
- **Boolean**: `python AND (tutorial OR guide)`
- **Wildcards**: `pyth*` or `doc?.pdf`
- **Fuzzy Search**: `pythn~2` (allows 2 character differences)
- **File Type**: `@pdf financial report`
- **Tags**: `#important #urgent meeting notes`

## 🎨 Themes

### 🌆 Cyberpunk
Neon colors with a futuristic feel. Perfect for night coding sessions.

### 🌊 Ocean Breeze
Calming blues and aqua tones. Great for long document reading.

### 🌲 Forest Deep
Natural greens with earthy accents. Reduces eye strain.

### 🌅 Sunset Glow
Warm oranges and purples. Energizing and vibrant.

## 🔧 Configuration

Create a `.env` file with:

```env
# OpenAI Configuration
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4

# Context7 Configuration
CONTEXT7_WORKSPACE=default
CONTEXT7_INDEX_PATH=./documents

# UI Configuration
THEME=cyberpunk
ANIMATIONS_ENABLED=true
SOUND_EFFECTS=false

# Search Configuration
MAX_RESULTS=20
FUZZY_SEARCH=true
```

## 📊 Architecture

```
context7-explorer/
├── src/
│   ├── ui/              # Beautiful UI components
│   │   ├── themes.py    # Theme definitions
│   │   ├── components.py # UI widgets
│   │   └── dashboard.py  # Main layout
│   ├── data/            # Data management
│   │   ├── history.py   # Search history
│   │   ├── bookmarks.py # Bookmarks
│   │   └── sessions.py  # Session management
│   ├── context7_integration.py  # Context7 MCP integration
│   ├── explorer_cli.py          # Main CLI application
│   └── config.py                # Configuration
├── tests/               # Test suite
├── docs/               # Documentation
└── main.py            # Entry point
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Pydantic AI](https://github.com/pydantic/pydantic-ai)
- Powered by [Context7 MCP](https://github.com/upstash/context7-mcp)
- Beautiful terminal UI with [Rich](https://github.com/Textualize/rich)

---

<div align="center">

Made with ❤️ by the Context7 Explorer Team

⭐ Star us on GitHub!

</div>
```

This reimagined Context7 Document Explorer features:

1. **Stunning Visual Design**:
   - 4 beautiful themes with gradients and effects
   - Smooth animations and transitions
   - ASCII art and visual elements
   - Modern dashboard layout

2. **Powerful Search**:
   - AI-powered document discovery
   - Advanced search syntax
   - Real-time results
   - Smart filtering and sorting

3. **Great User Experience**:
   - Intuitive keyboard shortcuts
   - Search history and analytics
   - Bookmarks and sessions
   - Document previews with syntax highlighting

4. **Deep Context7 Integration**:
   - MCP server connection
   - Document analysis
   - Similar document suggestions
   - Contextual search

The application provides an incredibly user-friendly and visually appealing way to search and explore documents using the Context7 MCP server!

