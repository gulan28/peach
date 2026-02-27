
# Peach

Peach is a coding agent. Lavishly copied from the pi agent. [https://pi.dev](https://pi.dev)


## What is Peach?

Imagine you have a really smart assistant that lives in your terminal. You type a question, it thinks about it, and sometimes it needs to do things — like run a command, read a file, or list a folder — before it can answer you. Peach is the brain and plumbing that makes all of that work.

The key idea: **the assistant part has no idea it's running in a terminal**. It just takes messages in and pushes events out. The terminal is just one way to talk to it — you could just as easily wire it up to a web app, a Slack bot, or anything else.

---

## The Big Picture

```
 You type something
       |
       v
  ┌─────────┐         ┌─────────────┐
  │  cli.py  │ ------> │   Agent     │
  │ (terminal│ <events │  (agent.py) │
  │  UI only)│         └──────┬──────┘
  └─────────┘                 │
                              │ calls
                              v
                     ┌─────────────────┐
                     │   agent_loop    │
                     │   (loop.py)     │
                     └───┬─────────┬───┘
                         │         │
              asks LLM   │         │  runs tools
                         v         v
                   ┌──────────┐  ┌─────────┐
                   │ Provider │  │  Tools   │
                   │(anthropic│  │(tools.py)│
                   │ /openai) │  └─────────┘
                   └──────────┘
```

In plain English:

1. **You type** into the terminal (`cli.py`)
2. The CLI hands your message to the **Agent** (`agent.py`)
3. The Agent runs the **agent loop** (`loop.py`)
4. The loop asks an **LLM provider** (Claude or GPT) what to do
5. If the LLM says "I need to use a tool", the loop runs that **tool** and feeds the result back
6. Steps 4-5 repeat until the LLM has a final answer
7. Events stream back through the Agent to the CLI, which prints them live

---

## File Map

```
peach/
├── __init__.py            # Front door — exports the public API
├── __main__.py            # "python -m peach" entry point (2 lines)
├── types.py               # All the data shapes (messages, events, etc.)
├── tools.py               # What the agent can DO (shell, files, etc.)
├── loop.py                # The engine — the think→act→repeat cycle
├── agent.py               # The driver — owns state, queues, subscriptions
├── providers/
│   ├── __init__.py        # Re-exports providers
│   ├── base.py            # The "contract" any LLM provider must follow
│   ├── anthropic.py       # Claude adapter
│   └── openai.py          # GPT adapter
├── art.py                 # The pretty "peach" banner
└── cli.py                 # Terminal UI (the ONLY file that knows about terminals)
```

---

## Deep Dive: Every Layer Explained

### 1. `types.py` — The Vocabulary

Before anything can talk to anything else, they need to agree on the shape of data. That's what `types.py` does. Think of it like defining the kinds of LEGO bricks everyone will use.

#### `ToolCall`
```python
@dataclass
class ToolCall:
    id: str                    # unique ID (from the LLM)
    name: str                  # e.g. "shell_command"
    arguments: dict[str, Any]  # e.g. {"command": "ls -la"}
```
When the LLM decides it needs to use a tool, it produces one of these. The `id` is important because when we send the result back, we need to say "this result is for *that* tool call."

#### `Usage`
```python
@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
```
Tracks how many tokens (roughly, word-pieces) the LLM consumed. Useful for cost tracking.

#### `AgentMessage`
```python
@dataclass
class AgentMessage:
    role: "user" | "assistant" | "tool_result"
    content: str
    tool_calls: list[ToolCall]    # only for assistant messages
    tool_call_id: str | None      # only for tool_result messages
    tool_name: str | None         # only for tool_result messages
    is_error: bool                # did the tool fail?
    stop_reason: str | None       # why the LLM stopped
    usage: Usage | None
    timestamp: float
```
This is the universal message format. Every message flowing through the system — whether it came from you, the LLM, or a tool — is one of these. The `role` field tells you who's talking:

- **`user`**: You said something
- **`assistant`**: The LLM replied (might include `tool_calls`)
- **`tool_result`**: A tool finished and here's its output

This is deliberately **provider-agnostic**. It doesn't care if you're talking to Claude or GPT. The providers translate to/from this format at the edges.

#### `AgentEvent`
```python
@dataclass
class AgentEvent:
    type: "agent_start" | "turn_start" | "message_start" |
          "message_delta" | "message_end" | "tool_execution_start" |
          "tool_execution_end" | "turn_end" | "agent_end"
    message: AgentMessage | None
    text: str | None              # for message_delta (streaming text chunks)
    tool_call_id: str | None
    tool_name: str | None
    tool_result: str | None
    is_error: bool
    messages: list[AgentMessage]  # for agent_end (all new messages)
```
These are the real-time play-by-play of what the agent is doing. The lifecycle looks like this:

```
agent_start
  turn_start
    message_start (user message)
    message_end
    message_start (assistant starts responding)
      message_delta ("Here's")     ← streaming text, word by word
      message_delta (" what I")
      message_delta (" found...")
    message_end (assistant done)
    tool_execution_start           ← "I'm running shell_command now"
    tool_execution_end             ← "Done, here's the result"
  turn_end
  turn_start                       ← another round because the LLM got tool results
    message_start (assistant)
      message_delta (streaming...)
    message_end
  turn_end
agent_end                          ← all done, here are all new messages
```

Anyone who subscribes to these events (like the CLI) gets this stream and can render it however they want.

#### `StreamEvent`
```python
@dataclass
class StreamEvent:
    type: "text_delta" | "tool_call_start" | "tool_call_delta" |
          "tool_call_end" | "done" | "error"
```
This is a **lower-level** event, used only between a provider and the loop. It represents raw chunks coming from the LLM's streaming API. The loop translates these into `AgentEvent`s.

---

### 2. `tools.py` — The Agent's Hands

The agent can think (via the LLM), but it needs hands to *do* things. That's what tools are.

#### The Tool dataclass
```python
@dataclass
class Tool:
    name: str                              # "shell_command"
    description: str                       # what the LLM reads to decide when to use it
    parameters: dict[str, Any]             # JSON Schema — tells the LLM what arguments to provide
    execute: Callable[..., Awaitable[str]] # the actual async function to run
```

A tool is four things:
1. A **name** the LLM uses to call it
2. A **description** so the LLM knows *when* to call it
3. A **parameter schema** (JSON Schema) so the LLM knows *how* to call it
4. An **execute function** that actually does the work

The schema is critical. When the LLM sees:
```json
{
  "name": "read_file",
  "description": "Read the contents of a file",
  "parameters": {
    "type": "object",
    "properties": {
      "path": {"type": "string", "description": "Path to the file"},
      "offset": {"type": "integer", "description": "Line offset..."},
      "limit": {"type": "integer", "description": "Max lines..."}
    },
    "required": ["path"]
  }
}
```
...it knows it can call `read_file` with a path, and optionally an offset and limit.

#### Built-in tools

| Tool | What it does | Key detail |
|------|-------------|------------|
| `shell_command` | Runs any shell command | Has a configurable timeout (default 30s). Captures both stdout and stderr. Reports exit codes. |
| `read_file` | Reads a file's contents | Supports `offset` (skip N lines) and `limit` (read only N lines) for large files. |
| `write_file` | Writes content to a file | Auto-creates parent directories with `os.makedirs`. |
| `list_directory` | Lists a directory | Supports glob patterns for filtering (e.g. `"*.py"`). |

All tool functions are `async` even though the file I/O is synchronous — this keeps the interface uniform and doesn't block the event loop for shell commands (which use `asyncio.create_subprocess_shell`).

#### Adding your own tool

```python
async def _my_tool(query: str) -> str:
    return f"You searched for: {query}"

MY_TOOL = Tool(
    name="search",
    description="Search for something",
    parameters={...json schema...},
    execute=_my_tool,
)

agent = Agent(provider=..., tools=get_builtin_tools() + [MY_TOOL])
```

---

### 3. `providers/` — Talking to LLMs

#### `base.py` — The Contract

```python
@runtime_checkable
class LLMProvider(Protocol):
    async def stream(
        self,
        system_prompt: str,
        messages: list[AgentMessage],
        tools: list[Tool],
    ) -> AsyncIterator[StreamEvent]: ...
```

This is a Python Protocol — basically a contract that says "if you want to be an LLM provider, you must have a `stream` method that takes these inputs and yields `StreamEvent`s." You don't need to inherit from anything, you just need to match the shape (duck typing).

The `@runtime_checkable` decorator means you can do `isinstance(my_provider, LLMProvider)` at runtime.

#### `anthropic.py` — The Claude Adapter

**What it does:** Takes our universal `AgentMessage` list, translates it into Anthropic's API format, streams the response, and translates the chunks back into `StreamEvent`s.

**The translation layer:**

| Peach concept | Anthropic API equivalent |
|--------------|------------------------|
| `AgentMessage(role="user")` | `{"role": "user", "content": "..."}` |
| `AgentMessage(role="assistant", tool_calls=[...])` | `{"role": "assistant", "content": [{"type": "text", ...}, {"type": "tool_use", ...}]}` |
| `AgentMessage(role="tool_result")` | `{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "..."}]}` |
| `Tool` | `{"name": "...", "description": "...", "input_schema": {...}}` |

Note that Anthropic wraps tool results inside a `user` message with a special `tool_result` content block. The system prompt goes in a separate `system` parameter, not as a message.

**Streaming flow:**

```
Anthropic API events          →  Our StreamEvents
─────────────────────            ─────────────────
content_block_start (text)       (nothing yet, wait for deltas)
content_block_delta (text)    →  text_delta
content_block_start (tool_use)→  tool_call_start
content_block_delta (json)    →  tool_call_delta
content_block_stop            →  tool_call_end
message_delta                    (captures usage)
(stream ends)                 →  done (with final message + usage)
```

The provider accumulates tool call JSON incrementally. As chunks arrive (`"{\n  \"command\": \"ls"` then `" -la\""` then `"\n}"`), it buffers them. When the block stops, the full JSON is parsed.

#### `openai.py` — The GPT Adapter

Same idea, different API format:

| Peach concept | OpenAI API equivalent |
|--------------|----------------------|
| System prompt | `{"role": "system", "content": "..."}` (a message, not separate) |
| `AgentMessage(role="assistant", tool_calls=[...])` | `{"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "...", "arguments": "..."}}]}` |
| `AgentMessage(role="tool_result")` | `{"role": "tool", "tool_call_id": "...", "content": "..."}` |
| `Tool` | `{"type": "function", "function": {"name": "...", "parameters": {...}}}` |

Key differences from Anthropic:
- System prompt is a message with `role: "system"`, not a separate parameter
- Tool call arguments are a JSON *string*, not an object
- Tool results use `role: "tool"`, not nested inside a user message
- Tool definitions are wrapped in `{"type": "function", "function": {...}}`
- Usage comes on the final chunk with `stream_options: {"include_usage": true}`
- `tool_call_end` events are emitted after the stream finishes (OpenAI doesn't have explicit block-end events)

---

### 4. `loop.py` — The Engine

This is the heart of peach. It implements the **think → act → observe → repeat** cycle.

#### `agent_loop()` — The Main Function

```
agent_loop(prompts, system_prompt, provider, tools, messages, ...)
    → AsyncIterator[AgentEvent]
```

It's an async generator. You feed it user messages and it yields a stream of events as the agent works. Here's the full flow:

```
agent_loop called with user prompt
│
├─ yield agent_start
├─ yield turn_start
├─ yield message_start/end for each user prompt
├─ check for steering messages (user typed while we were setting up)
│
└─ OUTER LOOP (continues if follow-ups arrive)
   │
   └─ INNER LOOP (continues while there are tool calls or pending messages)
      │
      ├─ If pending steering messages exist:
      │    inject them into conversation (yield events for each)
      │
      ├─ Stream assistant response (_stream_assistant)
      │    ├─ Calls provider.stream()
      │    ├─ Yields message_delta events (streaming text)
      │    ├─ Accumulates tool calls from streaming JSON
      │    └─ Returns final AgentMessage
      │
      ├─ If error or aborted → yield turn_end, agent_end, STOP
      │
      ├─ If assistant made tool calls:
      │    ├─ Execute each tool sequentially (_execute_tool_calls)
      │    │    ├─ After EACH tool, check for steering messages
      │    │    └─ If steering found → skip remaining tools
      │    ├─ Add tool results to conversation
      │    └─ CONTINUE inner loop (LLM needs to see tool results)
      │
      ├─ If no tool calls → inner loop ends
      │
      └─ Check for steering messages after turn
   │
   ├─ Inner loop done. Check for follow-up messages.
   │   If found → set as pending, CONTINUE outer loop
   │
   └─ No follow-ups → BREAK
│
└─ yield agent_end (with all new messages)
```

#### `_stream_assistant()` — Talking to the LLM

This function:
1. Calls `provider.stream(system_prompt, messages, tools)`
2. Iterates through `StreamEvent`s from the provider
3. For `text_delta`: appends text, yields `message_delta` to the caller
4. For `tool_call_start/delta/end`: accumulates the tool call JSON incrementally
5. For `done`: captures the final message and usage
6. For `error`: records the error
7. Checks abort between events
8. Appends the final `AgentMessage` to the conversation history
9. Returns the message + all buffered events

#### `_execute_tool_calls()` — Running Tools

Tools are executed **sequentially**, not in parallel. This is intentional — it keeps things predictable and allows steering to interrupt between tools.

For each tool call:
1. Look up the tool by name
2. Call `tool.execute(**arguments)` — the LLM's JSON arguments are unpacked as keyword args
3. Wrap the result in an `AgentMessage(role="tool_result")`
4. **Check for steering messages** — if the user typed something while tools were running:
   - Capture the steering messages
   - Skip all remaining tool calls (mark them as "Skipped due to queued user message")
   - Break out of the loop
5. Return all results + any steering messages

#### The Two Loops Explained Simply

Think of it like a conversation at a restaurant:

**Inner loop** (tool calls): "I need to check the kitchen" → goes to kitchen → comes back → "I also need to check the wine cellar" → goes → comes back → "OK here's your answer"

**Outer loop** (follow-ups): The waiter would leave, but you say "actually, one more thing..." → whole inner loop runs again.

**Steering**: While the waiter is in the kitchen, you shout "never mind the wine!" → waiter skips the wine cellar, comes back, and addresses your new request.

---

### 5. `agent.py` — The Driver

The Agent class is the high-level wrapper that most users interact with. It owns all the state and provides a clean API.

#### State

```python
self.provider          # which LLM to use
self.system_prompt     # personality / instructions
self.tools             # available tools
self.messages          # full conversation history (mutable list)
self.is_streaming      # True while the agent is working

self._steering_queue   # asyncio.Queue — mid-run interrupts
self._follow_up_queue  # asyncio.Queue — after-completion messages
self._listeners        # event callbacks
self._abort_event      # asyncio.Event — stop signal
```

#### The Public API

| Method | What it does |
|--------|-------------|
| `subscribe(callback)` | Register a function to receive `AgentEvent`s. Returns an unsubscribe function. |
| `prompt(text)` | Send a message and run the full agent loop. Blocks (async) until done. |
| `prompt_async(text)` | Same as prompt but returns an `asyncio.Task` immediately. |
| `steer(text)` | Queue a message that interrupts the agent mid-run. Non-blocking. |
| `follow_up(text)` | Queue a message delivered after the agent finishes. Non-blocking. |
| `abort()` | Set the abort signal. The loop checks this and stops. |
| `reset()` | Clear everything — messages, queues, abort flag. |

#### How Queues Work

The queues use `asyncio.Queue`, which is thread-safe and async-friendly.

**Steering mode** (`one_at_a_time` vs `all`):
- `one_at_a_time` (default): If you typed 3 messages while the agent was busy, it processes them one per turn. Each one gets its own LLM response.
- `all`: All 3 messages are injected at once before the next LLM call.

**Follow-up mode** works the same way.

The loop calls `get_steering_messages()` (which drains the queue) at two points:
1. After each tool execution
2. After each turn completes

It calls `get_follow_up_messages()` only when it would otherwise stop.

#### Why Separate Steering and Follow-up?

They solve different problems:

- **Steering** = "Stop what you're doing and pay attention to this." Interrupts tool execution.
- **Follow-up** = "When you're done, also do this." Extends the conversation without interrupting.

Example:
```python
agent.prompt("Deploy the app")       # starts running
agent.steer("Wait, use staging!")    # interrupts mid-deploy
agent.follow_up("Then run tests")   # queued for after deploy finishes
```

---

### 6. `cli.py` — The Terminal UI

This is deliberately thin. It's the **only file that imports terminal libraries** (Rich). Everything else is pure Python with no UI dependency.

#### Provider Auto-Detection

```python
def _make_provider():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicProvider(...)
    elif os.environ.get("OPENAI_API_KEY"):
        return OpenAIProvider(...)
    else:
        print error and exit
```

Checks env vars in order. You can override the model with `PEACH_MODEL` and the system prompt with `PEACH_SYSTEM_PROMPT`.

#### `CLIRenderer` — Event → Pixels

The renderer subscribes to agent events and translates them into terminal output:

| Event | What it renders |
|-------|----------------|
| `message_start` (assistant) | Start a Rich `Live` display for streaming |
| `message_delta` | Append text, re-render as Markdown in the Live display |
| `message_end` | Stop the Live display |
| `tool_execution_start` | Print `▶ tool_name` (dimmed) |
| `tool_execution_end` (success) | Print `✓ tool_name` (green) + truncated result |
| `tool_execution_end` (error) | Print `✗ tool_name` (red) + error message |

The Rich `Live` display updates 15 times per second, giving smooth streaming output. It tries to render as Markdown first; if that fails, falls back to plain text.

#### The Input Loop

```python
while True:
    user_input = await run_in_executor(input("› "))  # blocking input on a thread
    task = asyncio.create_task(agent.prompt(user_input))

    while not task.done():
        await asyncio.wait_for(shield(task), timeout=0.1)
        # ↑ checks every 100ms, allowing KeyboardInterrupt to be caught
```

The trick here: `input()` is blocking and would freeze asyncio, so it runs on a thread via `run_in_executor`. While the agent is working, the loop polls every 100ms, allowing `Ctrl+C` to be caught and converted to `agent.abort()`.

Special commands: `/quit`, `/exit` → exit. `/reset` → clear conversation.

---

### 7. `art.py` — The Banner

A small file with the FIGlet "peach" text rendered inside a Rich `Panel` (box). Uses `#FF9B7A` (a peachy-orange color) for both the text and the border.

---

## Data Flow: A Complete Example

Let's trace what happens when you type: **"What files are in this directory?"**

```
1. cli.py: input("› ") returns "What files are in this directory?"
2. cli.py: calls agent.prompt("What files are in this directory?")
3. agent.py: creates AgentMessage(role="user", content="What files...")
4. agent.py: calls agent_loop(prompts=[user_msg], ...)
5. loop.py: yields agent_start, turn_start
6. loop.py: yields message_start(user_msg), message_end(user_msg)
7. loop.py: calls _stream_assistant()
8.   anthropic.py: converts messages to Anthropic format
9.   anthropic.py: calls client.messages.stream(model, messages, tools)
10.  anthropic.py: yields text_delta("I'll list") → loop yields message_delta
11.  anthropic.py: yields text_delta(" the files") → loop yields message_delta
12.  anthropic.py: yields tool_call_start(id="tc_1", name="list_directory")
13.  anthropic.py: yields tool_call_delta(json='{"path": "."}')
14.  anthropic.py: yields tool_call_end(id="tc_1")
15.  anthropic.py: yields done(message=..., usage=...)
16. loop.py: assistant_msg has tool_calls → execute them
17. loop.py: calls _execute_tool_calls()
18.  tools.py: _list_directory(path=".") → returns "file1.py\nfile2.py\n..."
19. loop.py: yields tool_execution_start, tool_execution_end
20. loop.py: adds tool_result to messages
21. loop.py: INNER LOOP continues (has_more_tool_calls was true, now check again)
22. loop.py: calls _stream_assistant() again (LLM sees the tool result now)
23.  anthropic.py: streams "Here are the files in this directory:\n- file1.py\n..."
24. loop.py: yields message_delta events for each chunk
25. loop.py: assistant_msg has NO tool_calls → inner loop ends
26. loop.py: checks follow-up queue → empty → outer loop breaks
27. loop.py: yields agent_end
28. agent.py: emits all events to listeners
29. cli.py: CLIRenderer handles each event, rendering live markdown output
```

---

## Key Design Decisions

### Why async generators for the loop?

The agent loop is an `async def` that `yield`s events. This means:
- The caller controls the pace (pull-based)
- No need for a separate event bus or callback system at the loop level
- Natural backpressure — if the caller is slow, the loop waits
- Easy to compose and test

### Why mutable message list?

`messages` is passed by reference and modified in-place. Both the Agent and the loop append to the same list. This avoids copying the full conversation history on every turn, which matters when conversations get long.

### Why sequential tool execution?

Running tools in parallel would be faster, but:
- It prevents steering from interrupting between tools
- Some tools might have dependencies (write then read)
- It's simpler to reason about
- LLMs often intend sequential execution when they emit multiple tool calls

### Why the provider pattern?

By defining `LLMProvider` as a Protocol, any class with a matching `stream` method works. You don't need to inherit from anything. This means:
- Easy to add new providers (Gemini, local LLMs, etc.)
- Easy to mock for testing
- No coupling between the loop and specific LLM APIs

### Why separate AgentEvent and StreamEvent?

They operate at different levels:
- `StreamEvent` is raw LLM output (text chunks, tool call JSON fragments)
- `AgentEvent` is the full agent lifecycle (turns, tool executions, start/end)

The loop translates between them. Consumers (like the CLI) never see `StreamEvent`s.

---

## Module Dependency Graph

```
types.py          ← depends on nothing (the foundation)
    ↑
tools.py          ← depends on types
    ↑
providers/base.py ← depends on types, tools
    ↑
providers/anthropic.py ← depends on types, tools (+ anthropic SDK)
providers/openai.py    ← depends on types, tools (+ openai SDK)
    ↑
loop.py           ← depends on types, tools, providers/base
    ↑
agent.py          ← depends on types, tools, loop, providers/base
    ↑
cli.py            ← depends on agent, tools, types, art (+ rich)
art.py            ← depends on nothing (just rich)
```

No circular dependencies. Each layer only looks down, never up.
