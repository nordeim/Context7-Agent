# File: src/cli.py
from __future__ import annotations
import asyncio
import typer
from rich.markdown import Markdown
from rich.live import Live
from rich.panel import Panel
from rich.align import Align

from .history import History
from .agent import chat_stream
from .utils import console, banner
from .config import settings

app = typer.Typer(help="Talk to your Context7 Agent!")

history = History(settings.history_file)


@app.command()
def chat():
    """Interactive chat session with live-streamed responses."""
    asyncio.run(_run_chat())


async def _run_chat():
    banner()
    console.print("[secondary]Type '/exit' to quit. '/theme' to cycle themes.[/secondary]\n")

    while True:
        try:
            user_inp = console.input("[user]❯ [/user]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[secondary]Session terminated.[/secondary]")
            break

        if user_inp.strip() == "":
            continue
        if user_inp.strip().lower() == "/exit":
            break

        history.add("user", user_inp)

        # Stream assistant answer + MCP events
        async for role, payload in chat_stream(history):
            if role == "assistant":
                console.print(Markdown(payload))
            elif role == "mcp":
                console.print(Panel(Align.left(payload), title="📄 MCP"))

        # Persist assistant reply (last assistant message now in history)
        last_ai_msg = history.messages[-1]
        history.save()


if __name__ == "__main__":
    app()
