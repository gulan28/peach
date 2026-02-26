from __future__ import annotations

import json
import os
from typing import AsyncIterator

from peach.tools import Tool
from peach.types import AgentMessage, StreamEvent, ToolCall, Usage


class AnthropicProvider:
    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model

    async def stream(
        self,
        system_prompt: str,
        messages: list[AgentMessage],
        tools: list[Tool],
    ) -> AsyncIterator[StreamEvent]:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self.api_key)

        # Convert AgentMessages to Anthropic format
        api_messages = _convert_messages(messages)
        api_tools = _convert_tools(tools)

        kwargs: dict = {
            "model": self.model,
            "max_tokens": 8192,
            "messages": api_messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if api_tools:
            kwargs["tools"] = api_tools

        async with client.messages.stream(**kwargs) as stream:
            accumulated_tool_calls: dict[str, dict] = {}  # index -> {id, name, json}
            input_tokens = 0
            output_tokens = 0

            async for event in stream:
                if event.type == "message_start":
                    if hasattr(event, "message") and hasattr(event.message, "usage"):
                        input_tokens = getattr(event.message.usage, "input_tokens", 0)

                elif event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        idx = str(event.index)
                        accumulated_tool_calls[idx] = {
                            "id": block.id,
                            "name": block.name,
                            "json": "",
                        }
                        yield StreamEvent(
                            type="tool_call_start",
                            id=block.id,
                            name=block.name,
                        )

                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        yield StreamEvent(type="text_delta", text=delta.text)
                    elif delta.type == "input_json_delta":
                        idx = str(event.index)
                        if idx in accumulated_tool_calls:
                            accumulated_tool_calls[idx]["json"] += delta.partial_json
                            yield StreamEvent(
                                type="tool_call_delta",
                                id=accumulated_tool_calls[idx]["id"],
                                json_delta=delta.partial_json,
                            )

                elif event.type == "content_block_stop":
                    idx = str(event.index)
                    if idx in accumulated_tool_calls:
                        tc = accumulated_tool_calls.pop(idx)
                        yield StreamEvent(
                            type="tool_call_end",
                            id=tc["id"],
                        )

                elif event.type == "message_delta":
                    if hasattr(event, "usage") and event.usage:
                        output_tokens = getattr(event.usage, "output_tokens", 0)

            # Build final message from the accumulated response
            final = await stream.get_final_message()

            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            for block in final.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            id=block.id,
                            name=block.name,
                            arguments=block.input if isinstance(block.input, dict) else {},
                        )
                    )

            yield StreamEvent(
                type="done",
                message=AgentMessage(
                    role="assistant",
                    content="\n".join(text_parts),
                    tool_calls=tool_calls,
                    stop_reason=final.stop_reason,
                ),
                usage=Usage(
                    input_tokens=final.usage.input_tokens,
                    output_tokens=final.usage.output_tokens,
                ),
            )


def _convert_messages(messages: list[AgentMessage]) -> list[dict]:
    """Convert AgentMessages to Anthropic API format."""
    result = []
    for msg in messages:
        if msg.role == "user":
            result.append({"role": "user", "content": msg.content})
        elif msg.role == "assistant":
            content: list[dict] = []
            if msg.content:
                content.append({"type": "text", "text": msg.content})
            for tc in msg.tool_calls:
                content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                })
            result.append({"role": "assistant", "content": content})
        elif msg.role == "tool_result":
            result.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content,
                        "is_error": msg.is_error,
                    }
                ],
            })
    return result


def _convert_tools(tools: list[Tool]) -> list[dict]:
    """Convert Tool objects to Anthropic API format."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        }
        for t in tools
    ]
