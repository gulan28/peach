from __future__ import annotations

import asyncio
import glob as globmod
import os
from dataclasses import dataclass
from typing import Any, Callable, Awaitable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    execute: Callable[..., Awaitable[str]]


# ---- Built-in tool implementations ----

async def _shell_command(command: str, timeout: int = 30) -> str:
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        parts: list[str] = []
        if stdout:
            parts.append(stdout.decode(errors="replace"))
        if stderr:
            parts.append(f"[stderr]\n{stderr.decode(errors='replace')}")
        if proc.returncode and proc.returncode != 0:
            parts.append(f"[exit code: {proc.returncode}]")
        return "\n".join(parts) if parts else "(no output)"
    except asyncio.TimeoutError:
        return f"Command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


async def _read_file(path: str, offset: int = 0, limit: int | None = None) -> str:
    try:
        with open(path, "r") as f:
            lines = f.readlines()
        if offset:
            lines = lines[offset:]
        if limit is not None:
            lines = lines[:limit]
        return "".join(lines)
    except Exception as e:
        return f"Error reading file: {e}"


async def _write_file(path: str, content: str) -> str:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


async def _list_directory(path: str = ".", pattern: str | None = None) -> str:
    try:
        if pattern:
            full_pattern = os.path.join(path, pattern)
            entries = globmod.glob(full_pattern)
        else:
            entries = os.listdir(path)
        if not entries:
            return "(empty directory)"
        return "\n".join(sorted(entries))
    except Exception as e:
        return f"Error listing directory: {e}"


# ---- Tool definitions ----

SHELL_COMMAND = Tool(
    name="shell_command",
    description="Execute a shell command and return its output",
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to execute"},
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default 30)",
            },
        },
        "required": ["command"],
    },
    execute=_shell_command,
)

READ_FILE = Tool(
    name="read_file",
    description="Read the contents of a file",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file"},
            "offset": {
                "type": "integer",
                "description": "Line offset to start reading from (0-indexed)",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to read",
            },
        },
        "required": ["path"],
    },
    execute=_read_file,
)

WRITE_FILE = Tool(
    name="write_file",
    description="Write content to a file (creates parent directories if needed)",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file"},
            "content": {"type": "string", "description": "Content to write"},
        },
        "required": ["path", "content"],
    },
    execute=_write_file,
)

LIST_DIRECTORY = Tool(
    name="list_directory",
    description="List files and directories in a given path",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path (default: current directory)",
            },
            "pattern": {
                "type": "string",
                "description": "Glob pattern to filter entries",
            },
        },
        "required": [],
    },
    execute=_list_directory,
)


def get_builtin_tools() -> list[Tool]:
    return [SHELL_COMMAND, READ_FILE, WRITE_FILE, LIST_DIRECTORY]
