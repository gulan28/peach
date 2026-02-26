from __future__ import annotations

import asyncio
from typing import Callable, Literal

from peach.types import AgentEvent, AgentMessage
from peach.tools import Tool
from peach.loop import agent_loop
from peach.providers.base import LLMProvider


class Agent:
    """High-level agent wrapping the core loop.

    No terminal dependency — takes/emits AgentMessage and AgentEvent objects.
    cli.py is just one consumer of this interface.
    """

    def __init__(
        self,
        provider: LLMProvider,
        system_prompt: str = "You are a helpful assistant with access to tools.",
        tools: list[Tool] | None = None,
        steering_mode: Literal["all", "one_at_a_time"] = "one_at_a_time",
        follow_up_mode: Literal["all", "one_at_a_time"] = "one_at_a_time",
    ):
        self.provider = provider
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.messages: list[AgentMessage] = []
        self.is_streaming = False

        self._steering_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self._follow_up_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self._steering_mode = steering_mode
        self._follow_up_mode = follow_up_mode

        self._listeners: list[Callable[[AgentEvent], None]] = []
        self._abort_event = asyncio.Event()
        self._running_task: asyncio.Task | None = None

    # --- Public API ---

    def subscribe(self, callback: Callable[[AgentEvent], None]) -> Callable[[], None]:
        """Subscribe to agent events. Returns an unsubscribe function."""
        self._listeners.append(callback)
        return lambda: self._listeners.remove(callback)

    async def prompt(self, text: str) -> None:
        """Send a user prompt and run the agent loop to completion."""
        if self.is_streaming:
            raise RuntimeError(
                "Agent is already running. Use steer() or follow_up() to queue messages."
            )

        user_msg = AgentMessage(role="user", content=text)
        self._abort_event.clear()
        self.is_streaming = True

        try:
            async for event in agent_loop(
                prompts=[user_msg],
                system_prompt=self.system_prompt,
                provider=self.provider,
                tools=self.tools,
                messages=self.messages,
                get_steering_messages=self._dequeue_steering,
                get_follow_up_messages=self._dequeue_follow_up,
                abort_event=self._abort_event,
            ):
                self._emit(event)
        finally:
            self.is_streaming = False

    async def prompt_async(self, text: str) -> asyncio.Task:
        """Send a prompt and return the running task (non-blocking)."""
        task = asyncio.create_task(self.prompt(text))
        self._running_task = task
        return task

    def steer(self, text: str) -> None:
        """Queue a steering message to interrupt the agent mid-run.

        Delivered after the current tool execution completes, skipping remaining tools.
        """
        self._steering_queue.put_nowait(
            AgentMessage(role="user", content=text)
        )

    def follow_up(self, text: str) -> None:
        """Queue a follow-up message processed after the agent finishes.

        Delivered only when the agent has no more tool calls or steering messages.
        """
        self._follow_up_queue.put_nowait(
            AgentMessage(role="user", content=text)
        )

    def abort(self) -> None:
        """Abort the currently running agent loop."""
        self._abort_event.set()

    def reset(self) -> None:
        """Clear all state: messages and queues."""
        self.messages.clear()
        self._clear_queue(self._steering_queue)
        self._clear_queue(self._follow_up_queue)
        self._abort_event.clear()
        self.is_streaming = False

    # --- Private helpers ---

    def _emit(self, event: AgentEvent) -> None:
        for listener in self._listeners:
            listener(event)

    async def _dequeue_steering(self) -> list[AgentMessage]:
        return self._drain_queue(self._steering_queue, self._steering_mode)

    async def _dequeue_follow_up(self) -> list[AgentMessage]:
        return self._drain_queue(self._follow_up_queue, self._follow_up_mode)

    def _drain_queue(
        self, queue: asyncio.Queue[AgentMessage], mode: str
    ) -> list[AgentMessage]:
        if queue.empty():
            return []
        if mode == "one_at_a_time":
            return [queue.get_nowait()]
        # "all" mode
        items: list[AgentMessage] = []
        while not queue.empty():
            items.append(queue.get_nowait())
        return items

    @staticmethod
    def _clear_queue(queue: asyncio.Queue) -> None:
        while not queue.empty():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break
