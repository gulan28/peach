from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable

from peach.types import AgentMessage, StreamEvent
from peach.tools import Tool


@runtime_checkable
class LLMProvider(Protocol):
    async def stream(
        self,
        system_prompt: str,
        messages: list[AgentMessage],
        tools: list[Tool],
    ) -> AsyncIterator[StreamEvent]:
        """Stream a response from the LLM.

        Yields StreamEvent objects:
          - text_delta: partial text content
          - tool_call_start: beginning of a tool call (id, name)
          - tool_call_delta: partial JSON for tool arguments
          - tool_call_end: tool call complete
          - done: final message with usage
          - error: something went wrong
        """
        ...
