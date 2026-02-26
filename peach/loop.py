from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, Callable, Awaitable

from peach.types import AgentEvent, AgentMessage, StreamEvent, ToolCall, Usage
from peach.tools import Tool
from peach.providers.base import LLMProvider


async def agent_loop(
    prompts: list[AgentMessage],
    system_prompt: str,
    provider: LLMProvider,
    tools: list[Tool],
    messages: list[AgentMessage],
    get_steering_messages: Callable[[], Awaitable[list[AgentMessage]]] | None = None,
    get_follow_up_messages: Callable[[], Awaitable[list[AgentMessage]]] | None = None,
    abort_event: asyncio.Event | None = None,
) -> AsyncIterator[AgentEvent]:
    """Core agent loop.

    Adds prompts to the message history, then enters the LLM→tools loop.
    Yields AgentEvent objects throughout for real-time updates.

    Args:
        prompts: New user messages to process.
        system_prompt: System prompt for the LLM.
        provider: LLM provider to stream from.
        tools: Available tools.
        messages: Mutable message history (modified in-place).
        get_steering_messages: Async callable returning queued steering messages.
        get_follow_up_messages: Async callable returning queued follow-up messages.
        abort_event: asyncio.Event that, when set, aborts the loop.
    """
    new_messages: list[AgentMessage] = list(prompts)
    messages.extend(prompts)

    yield AgentEvent(type="agent_start")
    yield AgentEvent(type="turn_start")

    for prompt in prompts:
        yield AgentEvent(type="message_start", message=prompt)
        yield AgentEvent(type="message_end", message=prompt)

    # Check for steering messages at start
    pending: list[AgentMessage] = []
    if get_steering_messages:
        pending = await get_steering_messages()

    # Outer loop: continues when follow-up messages arrive after agent would stop
    while True:
        has_more_tool_calls = True
        first_turn = True

        # Inner loop: process tool calls and steering messages
        while has_more_tool_calls or pending:
            if abort_event and abort_event.is_set():
                yield AgentEvent(type="agent_end", messages=new_messages)
                return

            if not first_turn:
                yield AgentEvent(type="turn_start")
            else:
                first_turn = False

            # Inject pending messages before next assistant response
            if pending:
                for msg in pending:
                    yield AgentEvent(type="message_start", message=msg)
                    yield AgentEvent(type="message_end", message=msg)
                    messages.append(msg)
                    new_messages.append(msg)
                pending = []

            # Stream assistant response
            assistant_msg, events = await _stream_assistant(
                system_prompt, messages, provider, tools, abort_event
            )
            for ev in events:
                yield ev

            new_messages.append(assistant_msg)

            if assistant_msg.stop_reason in ("error", "aborted"):
                yield AgentEvent(type="turn_end", message=assistant_msg)
                yield AgentEvent(type="agent_end", messages=new_messages)
                return

            # Check for tool calls
            tool_calls = assistant_msg.tool_calls
            has_more_tool_calls = len(tool_calls) > 0

            if has_more_tool_calls:
                tool_results, steering = await _execute_tool_calls(
                    tools, tool_calls, abort_event, get_steering_messages
                )
                for ev in _tool_result_events(tool_results):
                    yield ev

                for result_msg in tool_results:
                    messages.append(result_msg)
                    new_messages.append(result_msg)

                if steering:
                    pending = steering

            yield AgentEvent(type="turn_end", message=assistant_msg)

            # Get steering messages after turn
            if not pending and get_steering_messages:
                pending = await get_steering_messages()

        # Agent would stop — check for follow-ups
        if get_follow_up_messages:
            follow_ups = await get_follow_up_messages()
            if follow_ups:
                pending = follow_ups
                continue

        break

    yield AgentEvent(type="agent_end", messages=new_messages)


async def _stream_assistant(
    system_prompt: str,
    messages: list[AgentMessage],
    provider: LLMProvider,
    tools: list[Tool],
    abort_event: asyncio.Event | None,
) -> tuple[AgentMessage, list[AgentEvent]]:
    """Stream an assistant response, returning the final message and events to emit."""
    events: list[AgentEvent] = []
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    # Track in-progress tool calls by id
    pending_tool_json: dict[str, str] = {}
    pending_tool_names: dict[str, str] = {}
    usage = Usage()
    stop_reason = "end_turn"
    error_message: str | None = None

    try:
        stream = provider.stream(system_prompt, messages, tools)
        async for event in stream:
            if abort_event and abort_event.is_set():
                stop_reason = "aborted"
                break

            if event.type == "text_delta":
                text_parts.append(event.text or "")
                events.append(AgentEvent(type="message_delta", text=event.text))

            elif event.type == "tool_call_start":
                pending_tool_json[event.id] = ""
                pending_tool_names[event.id] = event.name or ""

            elif event.type == "tool_call_delta":
                if event.id in pending_tool_json:
                    pending_tool_json[event.id] += event.json_delta or ""

            elif event.type == "tool_call_end":
                tc_id = event.id or ""
                raw_json = pending_tool_json.pop(tc_id, "")
                tc_name = pending_tool_names.pop(tc_id, "")
                try:
                    args = json.loads(raw_json) if raw_json else {}
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(id=tc_id, name=tc_name, arguments=args))

            elif event.type == "done":
                if event.message:
                    text_parts = [event.message.content] if event.message.content else text_parts
                    tool_calls = event.message.tool_calls or tool_calls
                    stop_reason = event.message.stop_reason or stop_reason
                if event.usage:
                    usage = event.usage

            elif event.type == "error":
                stop_reason = "error"
                error_message = event.error_message

    except Exception as e:
        stop_reason = "error"
        error_message = str(e)

    assistant_msg = AgentMessage(
        role="assistant",
        content="".join(text_parts),
        tool_calls=tool_calls,
        stop_reason=stop_reason,
        usage=usage,
    )
    messages.append(assistant_msg)

    events.insert(0, AgentEvent(type="message_start", message=assistant_msg))
    events.append(AgentEvent(type="message_end", message=assistant_msg))

    return assistant_msg, events


async def _execute_tool_calls(
    tools: list[Tool],
    tool_calls: list[ToolCall],
    abort_event: asyncio.Event | None,
    get_steering_messages: Callable[[], Awaitable[list[AgentMessage]]] | None,
) -> tuple[list[AgentMessage], list[AgentMessage] | None]:
    """Execute tool calls sequentially. Returns (tool_result_messages, steering_messages)."""
    results: list[AgentMessage] = []
    steering: list[AgentMessage] | None = None
    tool_map = {t.name: t for t in tools}

    for i, tc in enumerate(tool_calls):
        if abort_event and abort_event.is_set():
            # Skip remaining tool calls
            for skipped in tool_calls[i:]:
                results.append(AgentMessage(
                    role="tool_result",
                    content="Skipped: agent was aborted.",
                    tool_call_id=skipped.id,
                    tool_name=skipped.name,
                    is_error=True,
                ))
            break

        tool = tool_map.get(tc.name)
        is_error = False

        if not tool:
            result_text = f"Error: tool '{tc.name}' not found"
            is_error = True
        else:
            try:
                result_text = await tool.execute(**tc.arguments)
            except Exception as e:
                result_text = f"Error executing {tc.name}: {e}"
                is_error = True

        results.append(AgentMessage(
            role="tool_result",
            content=result_text,
            tool_call_id=tc.id,
            tool_name=tc.name,
            is_error=is_error,
        ))

        # Check for steering after each tool execution
        if get_steering_messages:
            msgs = await get_steering_messages()
            if msgs:
                steering = msgs
                # Skip remaining tool calls
                for skipped in tool_calls[i + 1:]:
                    results.append(AgentMessage(
                        role="tool_result",
                        content="Skipped due to queued user message.",
                        tool_call_id=skipped.id,
                        tool_name=skipped.name,
                        is_error=True,
                    ))
                break

    return results, steering


def _tool_result_events(results: list[AgentMessage]) -> list[AgentEvent]:
    """Generate events for tool results."""
    events: list[AgentEvent] = []
    for msg in results:
        events.append(AgentEvent(
            type="tool_execution_start",
            tool_call_id=msg.tool_call_id,
            tool_name=msg.tool_name,
        ))
        events.append(AgentEvent(
            type="tool_execution_end",
            tool_call_id=msg.tool_call_id,
            tool_name=msg.tool_name,
            tool_result=msg.content,
            is_error=msg.is_error,
        ))
    return events
