"""Thin CLI layer — the only terminal-coupled file."""
from __future__ import annotations

import asyncio
import os
import sys

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

from peach.agent import Agent
from peach.tools import get_builtin_tools
from peach.types import AgentEvent


console = Console()

def _print_peach_art():
    art = Text()
    art.append("                           █\n", style="green")
    art.append("                            █  █\n", style="green")
    art.append("                             ██\n", style="#8B4513")
    for line in [
        "                         ██████████",
        "                        ████████████",
        "                       ███░░█████████",
        "                       ████░█████████",
        "                        ████████████",
        "                        ████████████",
        "                          ████████",
        "                             ██",
    ]:
        art.append(line + "\n", style="#FF9B7A")
    console.print(art)


def _make_provider():
    """Auto-detect provider from environment."""
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if anthropic_key:
        from peach.providers.anthropic import AnthropicProvider
        model = os.environ.get("PEACH_MODEL", "claude-sonnet-4-20250514")
        return AnthropicProvider(api_key=anthropic_key, model=model)
    elif openai_key:
        from peach.providers.openai import OpenAIProvider
        model = os.environ.get("PEACH_MODEL", "gpt-4o")
        return OpenAIProvider(api_key=openai_key, model=model)
    else:
        console.print(
            "[red]No API key found.[/red] Set ANTHROPIC_API_KEY or OPENAI_API_KEY.",
        )
        sys.exit(1)


class CLIRenderer:
    """Subscribes to agent events and renders streaming output."""

    def __init__(self):
        self._current_text = ""
        self._live: Live | None = None
        self._in_message = False

    def handle_event(self, event: AgentEvent) -> None:
        if event.type == "agent_start":
            pass

        elif event.type == "message_start":
            if event.message and event.message.role == "assistant":
                self._current_text = ""
                self._in_message = True
                self._live = Live(Text(""), console=console, refresh_per_second=15)
                self._live.start()

        elif event.type == "message_delta":
            if self._in_message and event.text:
                self._current_text += event.text
                if self._live:
                    try:
                        self._live.update(Markdown(self._current_text))
                    except Exception:
                        self._live.update(Text(self._current_text))

        elif event.type == "message_end":
            if self._in_message and self._live:
                try:
                    self._live.update(Markdown(self._current_text))
                except Exception:
                    self._live.update(Text(self._current_text))
                self._live.stop()
                self._live = None
                self._in_message = False
                self._current_text = ""

        elif event.type == "tool_execution_start":
            name = event.tool_name or "unknown"
            console.print(f"  [dim]▶ {name}[/dim]", highlight=False)

        elif event.type == "tool_execution_end":
            name = event.tool_name or "unknown"
            if event.is_error:
                console.print(f"  [red]✗ {name}: {_truncate(event.tool_result)}[/red]")
            else:
                console.print(f"  [green]✓ {name}[/green] [dim]{_truncate(event.tool_result)}[/dim]")

        elif event.type == "agent_end":
            pass

    def stop(self) -> None:
        if self._live:
            self._live.stop()
            self._live = None


def _truncate(text: str | None, max_len: int = 120) -> str:
    if not text:
        return ""
    first_line = text.split("\n")[0]
    if len(first_line) > max_len:
        return first_line[:max_len] + "…"
    return first_line


async def _main() -> None:
    provider = _make_provider()
    system_prompt = os.environ.get(
        "PEACH_SYSTEM_PROMPT",
        "You are a helpful terminal assistant. Use the available tools to help the user. "
        "Be concise in your responses.",
    )
    agent = Agent(provider=provider, system_prompt=system_prompt, tools=get_builtin_tools())
    renderer = CLIRenderer()
    agent.subscribe(renderer.handle_event)

    _print_peach_art()
    console.print("[bold]peach[/bold] — terminal agent", highlight=False)
    console.print("[dim]Type a message, Ctrl+C to abort, Ctrl+D to quit.[/dim]\n")

    while True:
        try:
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, lambda: input("› ")
            )
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit"):
            console.print("[dim]Goodbye.[/dim]")
            break

        if user_input.lower() == "/reset":
            agent.reset()
            console.print("[dim]Conversation reset.[/dim]")
            continue

        try:
            task = asyncio.create_task(agent.prompt(user_input))
            # Allow Ctrl+C to abort the running agent
            while not task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                except KeyboardInterrupt:
                    renderer.stop()
                    agent.abort()
                    console.print("\n[yellow]Aborted.[/yellow]")
                    break
        except KeyboardInterrupt:
            renderer.stop()
            agent.abort()
            console.print("\n[yellow]Aborted.[/yellow]")

        console.print()


def main() -> None:
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
