from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class AgentMessage:
    role: Literal["user", "assistant", "tool_result"]
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    tool_name: str | None = None
    is_error: bool = False
    stop_reason: str | None = None
    usage: Usage | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentEvent:
    """Union-style event emitted throughout the agent lifecycle."""

    type: Literal[
        "agent_start",
        "turn_start",
        "message_start",
        "message_delta",
        "message_end",
        "tool_execution_start",
        "tool_execution_end",
        "turn_end",
        "agent_end",
    ]
    message: AgentMessage | None = None
    text: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: str | None = None
    is_error: bool = False
    messages: list[AgentMessage] = field(default_factory=list)


# --- Stream events from LLM providers ---

@dataclass
class StreamEvent:
    """Union-style event emitted by LLM provider streams."""

    type: Literal[
        "text_delta",
        "tool_call_start",
        "tool_call_delta",
        "tool_call_end",
        "done",
        "error",
    ]
    # text_delta
    text: str | None = None
    # tool_call_start / tool_call_delta / tool_call_end
    id: str | None = None
    name: str | None = None
    json_delta: str | None = None
    # done
    message: AgentMessage | None = None
    usage: Usage | None = None
    # error
    error_message: str | None = None
