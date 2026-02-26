from __future__ import annotations

import json
import os
from typing import AsyncIterator

from peach.tools import Tool
from peach.types import AgentMessage, StreamEvent, ToolCall, Usage


class OpenAIProvider:
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model

    async def stream(
        self,
        system_prompt: str,
        messages: list[AgentMessage],
        tools: list[Tool],
    ) -> AsyncIterator[StreamEvent]:
        import openai

        client = openai.AsyncOpenAI(api_key=self.api_key)

        api_messages = _convert_messages(system_prompt, messages)
        api_tools = _convert_tools(tools) or None  # OpenAI wants None, not []

        kwargs: dict = {
            "model": self.model,
            "messages": api_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if api_tools:
            kwargs["tools"] = api_tools

        accumulated_tool_calls: dict[int, dict] = {}  # index -> {id, name, json}
        full_text = ""
        input_tokens = 0
        output_tokens = 0
        finish_reason = None

        stream = await client.chat.completions.create(**kwargs)

        async for chunk in stream:
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens or 0
                output_tokens = chunk.usage.completion_tokens or 0

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            finish_reason = choice.finish_reason or finish_reason
            delta = choice.delta

            if delta and delta.content:
                full_text += delta.content
                yield StreamEvent(type="text_delta", text=delta.content)

            if delta and delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in accumulated_tool_calls:
                        accumulated_tool_calls[idx] = {
                            "id": tc_delta.id or "",
                            "name": tc_delta.function.name if tc_delta.function and tc_delta.function.name else "",
                            "json": "",
                        }
                        if accumulated_tool_calls[idx]["id"]:
                            yield StreamEvent(
                                type="tool_call_start",
                                id=accumulated_tool_calls[idx]["id"],
                                name=accumulated_tool_calls[idx]["name"],
                            )
                    else:
                        if tc_delta.id:
                            accumulated_tool_calls[idx]["id"] = tc_delta.id
                        if tc_delta.function and tc_delta.function.name:
                            accumulated_tool_calls[idx]["name"] = tc_delta.function.name

                    if tc_delta.function and tc_delta.function.arguments:
                        accumulated_tool_calls[idx]["json"] += tc_delta.function.arguments
                        yield StreamEvent(
                            type="tool_call_delta",
                            id=accumulated_tool_calls[idx]["id"],
                            json_delta=tc_delta.function.arguments,
                        )

        # Emit tool_call_end for all accumulated tool calls
        tool_calls: list[ToolCall] = []
        for idx in sorted(accumulated_tool_calls):
            tc = accumulated_tool_calls[idx]
            yield StreamEvent(type="tool_call_end", id=tc["id"])
            try:
                args = json.loads(tc["json"]) if tc["json"] else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=tc["id"], name=tc["name"], arguments=args))

        stop_reason = "end_turn" if finish_reason == "stop" else finish_reason or "end_turn"

        yield StreamEvent(
            type="done",
            message=AgentMessage(
                role="assistant",
                content=full_text,
                tool_calls=tool_calls,
                stop_reason=stop_reason,
            ),
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        )


def _convert_messages(system_prompt: str, messages: list[AgentMessage]) -> list[dict]:
    """Convert AgentMessages to OpenAI API format."""
    result: list[dict] = []
    if system_prompt:
        result.append({"role": "system", "content": system_prompt})

    for msg in messages:
        if msg.role == "user":
            result.append({"role": "user", "content": msg.content})
        elif msg.role == "assistant":
            entry: dict = {"role": "assistant"}
            if msg.content:
                entry["content"] = msg.content
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            result.append(entry)
        elif msg.role == "tool_result":
            result.append({
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content,
            })

    return result


def _convert_tools(tools: list[Tool]) -> list[dict]:
    """Convert Tool objects to OpenAI API format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]
