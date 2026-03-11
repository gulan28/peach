"""Blessed-based TUI layer — the only terminal-coupled file."""
from __future__ import annotations

import asyncio
import os
import sys
import textwrap

from blessed import Terminal

from peach.agent import Agent
from peach.art import print_peach_art
from peach.tools import get_builtin_tools
from peach.types import AgentEvent


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

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
        print("\033[31mNo API key found.\033[0m Set ANTHROPIC_API_KEY or OPENAI_API_KEY.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str | None, max_len: int = 120) -> str:
    if not text:
        return ""
    first_line = text.split("\n")[0]
    if len(first_line) > max_len:
        return first_line[:max_len] + "\u2026"
    return first_line


def _wrap_lines(text: str, width: int) -> list[str]:
    """Wrap text to *width*, preserving existing newlines."""
    result: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            result.append("")
        else:
            result.extend(textwrap.wrap(paragraph, width) or [""])
    return result


# ---------------------------------------------------------------------------
# Component system
# ---------------------------------------------------------------------------

class Component:
    """Base component: render(width) -> list[str]. Finalized = immutable."""

    def __init__(self) -> None:
        self.finalized = False
        self._cached_lines: list[str] | None = None

    def render(self, width: int) -> list[str]:
        if self.finalized and self._cached_lines is not None:
            return self._cached_lines
        lines = self._render(width)
        if self.finalized:
            self._cached_lines = lines
        return lines

    def _render(self, width: int) -> list[str]:
        return []

    def finalize(self) -> None:
        self.finalized = True
        self._cached_lines = None


class UserMessageComp(Component):
    """Renders a user message with `>` prefix."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text
        self.finalize()

    def _render(self, width: int) -> list[str]:
        usable = max(width - 4, 20)
        wrapped = _wrap_lines(self.text, usable)
        lines: list[str] = []
        for i, ln in enumerate(wrapped):
            prefix = "\033[1m\u276f\033[0m " if i == 0 else "  "
            lines.append(prefix + ln)
        return lines


class AssistantComp(Component):
    """Streams assistant text deltas; finalized on message_end."""

    def __init__(self) -> None:
        super().__init__()
        self.text = ""

    def append(self, delta: str) -> None:
        self.text += delta

    def _render(self, width: int) -> list[str]:
        if not self.text:
            return ["\033[2m...\033[0m"]
        usable = max(width - 2, 20)
        wrapped = _wrap_lines(self.text, usable)
        return wrapped


_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class SpinnerComp(Component):
    """Animated spinner shown while waiting for LLM response."""

    def __init__(self, label: str = "Thinking") -> None:
        super().__init__()
        self.label = label
        self._frame = 0

    def tick(self) -> None:
        self._frame = (self._frame + 1) % len(_SPINNER_FRAMES)

    def _render(self, width: int) -> list[str]:
        ch = _SPINNER_FRAMES[self._frame]
        return [f"\033[2m{ch} {self.label}\033[0m"]


class ToolComp(Component):
    """Shows tool execution status."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.result: str | None = None
        self.is_error = False

    def complete(self, result: str | None, is_error: bool) -> None:
        self.result = result
        self.is_error = is_error
        self.finalize()

    def _render(self, width: int) -> list[str]:
        if self.finalized:
            if self.is_error:
                return [f"  \033[31m\u2717 {self.name}: {_truncate(self.result)}\033[0m"]
            else:
                return [f"  \033[32m\u2713 {self.name}\033[0m \033[2m{_truncate(self.result)}\033[0m"]
        return [f"  \033[2m\u25b6 {self.name}\033[0m"]


# ---------------------------------------------------------------------------
# InputBox widget
# ---------------------------------------------------------------------------

class InputBox:
    """Multiline text input with cursor tracking and Emacs keybindings."""

    def __init__(self) -> None:
        self.lines: list[str] = [""]
        self.cy = 0
        self.cx = 0
        self._kill_ring = ""

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def clear(self) -> None:
        self.lines = [""]
        self.cy = 0
        self.cx = 0

    def insert_char(self, ch: str) -> None:
        line = self.lines[self.cy]
        self.lines[self.cy] = line[: self.cx] + ch + line[self.cx :]
        self.cx += len(ch)

    def insert_newline(self) -> None:
        line = self.lines[self.cy]
        before, after = line[: self.cx], line[self.cx :]
        self.lines[self.cy] = before
        self.lines.insert(self.cy + 1, after)
        self.cy += 1
        self.cx = 0

    def backspace(self) -> None:
        if self.cx > 0:
            line = self.lines[self.cy]
            self.lines[self.cy] = line[: self.cx - 1] + line[self.cx :]
            self.cx -= 1
        elif self.cy > 0:
            prev = self.lines[self.cy - 1]
            self.cx = len(prev)
            self.lines[self.cy - 1] = prev + self.lines[self.cy]
            del self.lines[self.cy]
            self.cy -= 1

    def delete(self) -> None:
        line = self.lines[self.cy]
        if self.cx < len(line):
            self.lines[self.cy] = line[: self.cx] + line[self.cx + 1 :]
        elif self.cy < len(self.lines) - 1:
            self.lines[self.cy] = line + self.lines[self.cy + 1]
            del self.lines[self.cy + 1]

    def move_left(self) -> None:
        if self.cx > 0:
            self.cx -= 1
        elif self.cy > 0:
            self.cy -= 1
            self.cx = len(self.lines[self.cy])

    def move_right(self) -> None:
        if self.cx < len(self.lines[self.cy]):
            self.cx += 1
        elif self.cy < len(self.lines) - 1:
            self.cy += 1
            self.cx = 0

    def move_up(self) -> None:
        if self.cy > 0:
            self.cy -= 1
            self.cx = min(self.cx, len(self.lines[self.cy]))

    def move_down(self) -> None:
        if self.cy < len(self.lines) - 1:
            self.cy += 1
            self.cx = min(self.cx, len(self.lines[self.cy]))

    def home(self) -> None:
        self.cx = 0

    def end(self) -> None:
        self.cx = len(self.lines[self.cy])

    # Emacs editing operations

    def kill_to_end(self) -> None:
        """Ctrl+K: kill from cursor to end of line."""
        line = self.lines[self.cy]
        if self.cx < len(line):
            self._kill_ring = line[self.cx :]
            self.lines[self.cy] = line[: self.cx]
        elif self.cy < len(self.lines) - 1:
            # At end of line: join with next line
            self._kill_ring = "\n"
            self.lines[self.cy] = line + self.lines[self.cy + 1]
            del self.lines[self.cy + 1]

    def kill_line_backward(self) -> None:
        """Ctrl+U: kill from cursor to beginning of line."""
        line = self.lines[self.cy]
        self._kill_ring = line[: self.cx]
        self.lines[self.cy] = line[self.cx :]
        self.cx = 0

    def yank(self) -> None:
        """Ctrl+Y: yank (paste) from kill ring."""
        if self._kill_ring:
            # Handle multi-line yank
            parts = self._kill_ring.split("\n")
            if len(parts) == 1:
                self.insert_char(parts[0])
            else:
                for i, part in enumerate(parts):
                    if i > 0:
                        self.insert_newline()
                    if part:
                        self.insert_char(part)

    def kill_word_backward(self) -> None:
        """Ctrl+W / Alt+Backspace: kill word backward."""
        line = self.lines[self.cy]
        if self.cx == 0:
            if self.cy > 0:
                self.backspace()
            return
        # Skip whitespace, then skip word chars
        end = self.cx
        i = self.cx - 1
        while i > 0 and line[i - 1] == " ":
            i -= 1
        while i > 0 and line[i - 1] != " ":
            i -= 1
        self._kill_ring = line[i:end]
        self.lines[self.cy] = line[:i] + line[end:]
        self.cx = i

    def kill_word_forward(self) -> None:
        """Alt+D: kill word forward."""
        line = self.lines[self.cy]
        if self.cx >= len(line):
            if self.cy < len(self.lines) - 1:
                self.delete()
            return
        start = self.cx
        i = self.cx
        while i < len(line) and line[i] == " ":
            i += 1
        while i < len(line) and line[i] != " ":
            i += 1
        self._kill_ring = line[start:i]
        self.lines[self.cy] = line[:start] + line[i:]

    def move_word_forward(self) -> None:
        """Alt+F: move cursor forward one word."""
        line = self.lines[self.cy]
        if self.cx >= len(line):
            if self.cy < len(self.lines) - 1:
                self.cy += 1
                self.cx = 0
            return
        i = self.cx
        while i < len(line) and line[i] == " ":
            i += 1
        while i < len(line) and line[i] != " ":
            i += 1
        self.cx = i

    def move_word_backward(self) -> None:
        """Alt+B: move cursor backward one word."""
        line = self.lines[self.cy]
        if self.cx == 0:
            if self.cy > 0:
                self.cy -= 1
                self.cx = len(self.lines[self.cy])
            return
        i = self.cx - 1
        while i > 0 and line[i - 1] == " ":
            i -= 1
        while i > 0 and line[i - 1] != " ":
            i -= 1
        self.cx = i

    def transpose_chars(self) -> None:
        """Ctrl+T: transpose characters before cursor."""
        line = self.lines[self.cy]
        if self.cx > 0 and len(line) >= 2:
            pos = self.cx if self.cx < len(line) else self.cx - 1
            if pos > 0:
                chars = list(line)
                chars[pos - 1], chars[pos] = chars[pos], chars[pos - 1]
                self.lines[self.cy] = "".join(chars)
                self.cx = min(pos + 1, len(line))

    def render(self, width: int) -> list[str]:
        out: list[str] = []
        for i, line in enumerate(self.lines):
            prefix = "\033[1m\u276f\033[0m " if i == 0 else "  "
            out.append(prefix + line)
        return out

    def cursor_screen_offset(self) -> tuple[int, int]:
        """Return (row_from_top_of_input, col) for cursor positioning."""
        prefix_len = 2  # "❯ " or "  "
        return (self.cy, prefix_len + self.cx)


# ---------------------------------------------------------------------------
# TUI Engine — differential rendering with scrollback
# ---------------------------------------------------------------------------

class TUI:
    """Manages rendering components with diff-based viewport updates.

    Instead of erasing and rewriting the entire viewport on every paint,
    we compare the new lines against the previously rendered lines and
    only update rows that actually changed.  This eliminates the visible
    flickering / duplication that occurs in terminals (especially VSCode)
    when rapid full-redraws are performed via cursor-up + clear sequences.
    """

    def __init__(self, term: Terminal) -> None:
        self.term = term
        self.components: list[Component] = []
        self.input_box = InputBox()
        self.show_input = True
        # Differential rendering state
        self._prev_lines: list[str] = []  # what is currently on screen
        self._cursor_row = 0              # 0-indexed row within viewport
        self._vp_height = 0               # total rows the viewport occupies

    def add_component(self, comp: Component) -> None:
        self.components.append(comp)

    # -- cursor helpers ---------------------------------------------------

    def _move_to_row(self, target: int) -> None:
        """Move cursor from _cursor_row to *target* row."""
        delta = target - self._cursor_row
        if delta < 0:
            sys.stdout.write(f"\033[{-delta}A")
        elif delta > 0:
            sys.stdout.write(f"\033[{delta}B")
        self._cursor_row = target

    # -- differential paint -----------------------------------------------

    def paint(self) -> None:
        """Repaint only the lines that changed since the last paint."""
        w = self.term.width or 80

        # Collect new lines from all components
        new_lines: list[str] = []
        for comp in self.components:
            new_lines.extend(comp.render(w))

        input_lines: list[str] = []
        if self.show_input:
            input_lines = self.input_box.render(w)
            new_lines.extend(input_lines)

        prev = self._prev_lines

        if not prev and not new_lines:
            return

        if not prev:
            # First paint — write everything from scratch
            for i, line in enumerate(new_lines):
                if i > 0:
                    sys.stdout.write("\n")
                sys.stdout.write(line)
            self._cursor_row = len(new_lines) - 1 if new_lines else 0
        else:
            # --- Differential update ---
            # Find the first line that differs
            min_len = min(len(prev), len(new_lines))
            first_diff = min_len  # assume all common lines match
            for i in range(min_len):
                if prev[i] != new_lines[i]:
                    first_diff = i
                    break

            max_len = max(len(prev), len(new_lines))

            if first_diff < max_len:
                # Move cursor to the first changed row
                self._move_to_row(first_diff)

                for i in range(first_diff, max_len):
                    sys.stdout.write("\r\033[2K")  # clear entire line
                    if i < len(new_lines):
                        sys.stdout.write(new_lines[i])
                    # Move to next row (except after the last one)
                    if i < max_len - 1:
                        sys.stdout.write("\n")
                        self._cursor_row = i + 1

                self._cursor_row = max_len - 1

            # If the new viewport is shorter, the extra rows were already
            # cleared inside the loop above.  Move cursor back to the end
            # of actual content so subsequent paints track correctly.
            if new_lines:
                final_row = len(new_lines) - 1
            else:
                final_row = 0
            self._move_to_row(final_row)

        self._prev_lines = new_lines[:]
        self._vp_height = max(self._vp_height, len(new_lines))

        # Position cursor inside the input box
        if self.show_input and input_lines:
            cy_off, cx_off = self.input_box.cursor_screen_offset()
            input_start = len(new_lines) - len(input_lines)
            target = input_start + cy_off
            self._move_to_row(target)
            sys.stdout.write(f"\r\033[{cx_off}C")

        sys.stdout.flush()

    # -- commit to scrollback ---------------------------------------------

    def commit_all(self) -> None:
        """Erase the viewport, re-print finalized content as normal
        scrollback output, and reset rendering state."""
        w = self.term.width or 80

        # Erase the entire viewport area
        if self._prev_lines:
            self._move_to_row(0)
            for i in range(self._vp_height):
                sys.stdout.write("\r\033[2K")
                if i < self._vp_height - 1:
                    sys.stdout.write("\n")
            # Back to top
            if self._vp_height > 1:
                sys.stdout.write(f"\033[{self._vp_height - 1}A")
            sys.stdout.write("\r")

        # Print as normal terminal output (enters scrollback)
        for comp in self.components:
            if not comp.finalized:
                comp.finalize()
            for line in comp.render(w):
                sys.stdout.write(line + "\n")

        self.components.clear()
        self._prev_lines = []
        self._cursor_row = 0
        self._vp_height = 0
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# TUIRenderer — subscribes to AgentEvents
# ---------------------------------------------------------------------------

class TUIRenderer:
    """Subscribes to agent events and updates TUI components."""

    def __init__(self, tui: TUI) -> None:
        self.tui = tui
        self._current_assistant: AssistantComp | None = None
        self._current_tool: ToolComp | None = None
        self._spinner: SpinnerComp | None = None

    def start_spinner(self) -> None:
        """Show a spinner while waiting for the LLM to respond."""
        spinner = SpinnerComp()
        self._spinner = spinner
        self.tui.add_component(spinner)
        self.tui.paint()

    def tick_spinner(self) -> None:
        """Advance spinner animation by one frame."""
        if self._spinner:
            self._spinner.tick()
            self.tui.paint()

    def _remove_spinner(self) -> None:
        """Remove the spinner component from the TUI."""
        if self._spinner:
            try:
                self.tui.components.remove(self._spinner)
            except ValueError:
                pass
            self._spinner = None

    def handle_event(self, event: AgentEvent) -> None:
        if event.type == "message_start":
            if event.message and event.message.role == "assistant":
                self._remove_spinner()
                comp = AssistantComp()
                self._current_assistant = comp
                self.tui.add_component(comp)

        elif event.type == "message_delta":
            if self._current_assistant and event.text:
                self._current_assistant.append(event.text)
                self.tui.paint()

        elif event.type == "message_end":
            if self._current_assistant:
                self._current_assistant.finalize()
                self._current_assistant = None
                self.tui.paint()

        elif event.type == "tool_execution_start":
            self._remove_spinner()
            name = event.tool_name or "unknown"
            comp = ToolComp(name)
            self._current_tool = comp
            self.tui.add_component(comp)
            self.tui.paint()

        elif event.type == "tool_execution_end":
            if self._current_tool:
                self._current_tool.complete(event.tool_result, event.is_error)
                self._current_tool = None
                self.tui.paint()

    def stop(self) -> None:
        """Clean up on abort."""
        self._remove_spinner()
        if self._current_assistant:
            self._current_assistant.finalize()
            self._current_assistant = None
        if self._current_tool:
            self._current_tool.finalize()
            self._current_tool = None


# ---------------------------------------------------------------------------
# Key reading
# ---------------------------------------------------------------------------

def _read_key(term: Terminal):
    """Read a keystroke in cbreak mode. Returns Keystroke or None."""
    val = term.inkey(timeout=0.05)
    if not val:
        return None
    return val


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def _main() -> None:
    provider = _make_provider()
    system_prompt = os.environ.get(
        "PEACH_SYSTEM_PROMPT",
        "You are a helpful terminal assistant. Use the available tools to help the user. "
        "Be concise in your responses.",
    )
    agent = Agent(provider=provider, system_prompt=system_prompt, tools=get_builtin_tools())

    term = Terminal()
    tui = TUI(term)
    renderer = TUIRenderer(tui)
    agent.subscribe(renderer.handle_event)

    # Print banner using pure ANSI
    print_peach_art()
    print("\033[1mpeach\033[0m \u2014 terminal agent")
    print("\033[2mType a message, Ctrl+C to abort, Ctrl+D to quit.\033[0m")
    print("\033[2mCtrl+O for newline, Enter to submit.\033[0m")
    print()

    loop = asyncio.get_event_loop()

    with term.cbreak():
        while True:
            tui.show_input = True
            tui.input_box.clear()
            tui.paint()

            submitted_text: str | None = None
            exit_requested = False

            while True:
                key = await loop.run_in_executor(None, lambda: _read_key(term))

                if key is None:
                    continue

                name = key.name if hasattr(key, "name") else None

                # Ctrl+D — exit (only on empty input, otherwise delete char)
                if key == "\x04":
                    if tui.input_box.text:
                        tui.input_box.delete()
                        tui.paint()
                        continue
                    exit_requested = True
                    break

                # Enter — submit
                if name == "KEY_ENTER" or key == "\n" or key == "\r":
                    submitted_text = tui.input_box.text.strip()
                    break

                # Ctrl+O — insert newline
                if key == "\x0f":
                    tui.input_box.insert_newline()
                    tui.paint()
                    continue

                # --- Emacs keybindings ---

                # Ctrl+A — beginning of line
                if key == "\x01":
                    tui.input_box.home()
                    tui.paint()
                    continue

                # Ctrl+E — end of line
                if key == "\x05":
                    tui.input_box.end()
                    tui.paint()
                    continue

                # Ctrl+B — move left
                if key == "\x02":
                    tui.input_box.move_left()
                    tui.paint()
                    continue

                # Ctrl+F — move right
                if key == "\x06":
                    tui.input_box.move_right()
                    tui.paint()
                    continue

                # Ctrl+P — move up
                if key == "\x10":
                    tui.input_box.move_up()
                    tui.paint()
                    continue

                # Ctrl+N — move down
                if key == "\x0e":
                    tui.input_box.move_down()
                    tui.paint()
                    continue

                # Ctrl+K — kill to end of line
                if key == "\x0b":
                    tui.input_box.kill_to_end()
                    tui.paint()
                    continue

                # Ctrl+U — kill line backward
                if key == "\x15":
                    tui.input_box.kill_line_backward()
                    tui.paint()
                    continue

                # Ctrl+Y — yank
                if key == "\x19":
                    tui.input_box.yank()
                    tui.paint()
                    continue

                # Ctrl+W — kill word backward
                if key == "\x17":
                    tui.input_box.kill_word_backward()
                    tui.paint()
                    continue

                # Ctrl+T — transpose chars
                if key == "\x14":
                    tui.input_box.transpose_chars()
                    tui.paint()
                    continue

                # Alt+Enter — insert newline (ESC followed by CR/LF)
                # Also handle Alt+F, Alt+B, Alt+D via ESC prefix
                if key == "\x1b":
                    next_key = await loop.run_in_executor(None, lambda: _read_key(term))
                    if next_key is None:
                        continue
                    # Alt+Enter
                    if next_key == "\r" or next_key == "\n":
                        tui.input_box.insert_newline()
                        tui.paint()
                        continue
                    # Alt+F — word forward
                    if str(next_key) == "f":
                        tui.input_box.move_word_forward()
                        tui.paint()
                        continue
                    # Alt+B — word backward
                    if str(next_key) == "b":
                        tui.input_box.move_word_backward()
                        tui.paint()
                        continue
                    # Alt+D — kill word forward
                    if str(next_key) == "d":
                        tui.input_box.kill_word_forward()
                        tui.paint()
                        continue
                    # Alt+Backspace — kill word backward
                    nname = next_key.name if hasattr(next_key, "name") else None
                    if nname == "KEY_BACKSPACE" or next_key == "\x7f" or next_key == "\x08":
                        tui.input_box.kill_word_backward()
                        tui.paint()
                        continue
                    # Unrecognized alt sequence, ignore
                    continue

                # Backspace
                if name == "KEY_BACKSPACE" or key == "\x7f" or key == "\x08":
                    tui.input_box.backspace()
                    tui.paint()
                    continue

                # Delete
                if name == "KEY_DELETE":
                    tui.input_box.delete()
                    tui.paint()
                    continue

                # Arrow keys
                if name == "KEY_LEFT":
                    tui.input_box.move_left()
                    tui.paint()
                    continue
                if name == "KEY_RIGHT":
                    tui.input_box.move_right()
                    tui.paint()
                    continue
                if name == "KEY_UP":
                    tui.input_box.move_up()
                    tui.paint()
                    continue
                if name == "KEY_DOWN":
                    tui.input_box.move_down()
                    tui.paint()
                    continue

                # Home / End
                if name == "KEY_HOME":
                    tui.input_box.home()
                    tui.paint()
                    continue
                if name == "KEY_END":
                    tui.input_box.end()
                    tui.paint()
                    continue

                # Regular printable character
                if key.isprintable() and len(key) == 1:
                    tui.input_box.insert_char(key)
                    tui.paint()
                    continue

            if exit_requested:
                tui.commit_all()
                print("\033[2mGoodbye.\033[0m")
                break

            if not submitted_text:
                continue

            if submitted_text.lower() in ("/quit", "/exit"):
                tui.commit_all()
                print("\033[2mGoodbye.\033[0m")
                break

            if submitted_text.lower() == "/reset":
                agent.reset()
                tui.commit_all()
                sys.stdout.write("\033[2mConversation reset.\033[0m\n\n")
                sys.stdout.flush()
                continue

            # Add user message component, show spinner, hide input
            user_comp = UserMessageComp(submitted_text)
            tui.add_component(user_comp)
            tui.show_input = False
            renderer.start_spinner()

            # Run agent
            try:
                task = asyncio.create_task(agent.prompt(submitted_text))
                while not task.done():
                    try:
                        await asyncio.wait_for(asyncio.shield(task), timeout=0.08)
                    except asyncio.TimeoutError:
                        renderer.tick_spinner()
                        continue
                    except KeyboardInterrupt:
                        renderer.stop()
                        agent.abort()
                        tui.paint()
                        sys.stdout.write("\n\033[33mAborted.\033[0m\n")
                        sys.stdout.flush()
                        break
            except KeyboardInterrupt:
                renderer.stop()
                agent.abort()
                tui.paint()
                sys.stdout.write("\n\033[33mAborted.\033[0m\n")
                sys.stdout.flush()

            tui.commit_all()
            print()


def main() -> None:
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
