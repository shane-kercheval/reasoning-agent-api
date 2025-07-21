#!/usr/bin/env python3
"""
MonsterUI Web Client for Reasoning Agent API.

A beautiful web interface for streaming conversations with collapsible reasoning steps
and power user controls. Built with MonsterUI and FastHTML.
"""

import os
import json
import uuid
from datetime import datetime

from fasthtml.common import *
from monsterui.all import *
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize MonsterUI app with blue theme
app, rt = fast_app(hdrs=Theme.blue.headers())

# Configuration
REASONING_API_URL = os.getenv("REASONING_API_URL", "http://localhost:8000")
REASONING_API_TOKEN = os.getenv("REASONING_API_TOKEN", "web-client-dev-token")
WEB_CLIENT_PORT = int(os.getenv("WEB_CLIENT_PORT", "8080"))

# Global state for conversations
conversations: dict[str, list[dict]] = {"default": []}  # Use single conversation for now
reasoning_events: dict[str, list[dict]] = {}
active_streams: dict[str, dict] = {}

# Session ID for tracing correlation
session_id = str(uuid.uuid4())

class ChatMessage:
    """Represents a chat message."""

    def __init__(self, role: str, content: str, timestamp: datetime | None = None):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> dict:  # noqa: D102
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }

def format_timestamp(timestamp: datetime) -> str:
    """Format timestamp for display."""
    return timestamp.strftime("%H:%M:%S")

def chat_message_component(
        message: ChatMessage,
        message_id: str,
        reasoning_steps: list | None = None,
    ) -> Div:
    """Render a single chat message with styling."""
    is_user = message.role == "user"
    is_assistant = message.role == "assistant"

    # Avatar and colors
    if is_user:
        avatar_content = "👤"
        avatar_class = "bg-blue-500 text-white"
        align_class = "ml-auto"
        card_class = "bg-blue-50 border-blue-200"
    elif is_assistant:
        avatar_content = "🤖"
        avatar_class = "bg-green-500 text-white"
        align_class = "mr-auto"
        card_class = "bg-green-50 border-green-200"
    else:  # system
        avatar_content = "⚙️"
        avatar_class = "bg-gray-500 text-white"
        align_class = "mr-auto"
        card_class = "bg-gray-50 border-gray-200"

    # Message content
    message_content = Div(
        DivLAligned(
            # Avatar
            Div(
                Span(avatar_content, cls="text-lg"),
                cls=f"flex h-10 w-10 items-center justify-center rounded-full {avatar_class}",
            ),
            # Message content
            Div(
                DivFullySpaced(
                    Strong(message.role.title(), cls="text-sm"),
                    Small(format_timestamp(message.timestamp), cls=TextPresets.muted_sm),
                ),
                Div(
                    # Handle multiline content safely
                    *[P(line) if line and line.strip() else Br() for line in (message.content or "").split('\n')],  # noqa: E501
                    cls="mt-2",
                ),
                cls="ml-3 flex-1",
            ),
        ),
    )

    # Add reasoning steps if this is an assistant message with reasoning
    if is_assistant and reasoning_steps:
        reasoning_accordion = Details(
            Summary("🧠 View reasoning steps", cls="cursor-pointer text-sm text-gray-600 mt-3"),
            Accordion(
                *[reasoning_step_component(step, i+1) for i, step in enumerate(reasoning_steps)],
                cls="mt-2",
            ),
            cls="ml-14",
        )

        return Card(
            message_content,
            reasoning_accordion,
            cls=f"max-w-4xl {align_class} mb-4 {card_class}",
            id=message_id,
        )

    return Card(
        message_content,
        cls=f"max-w-4xl {align_class} mb-4 {card_class}",
        id=message_id,
    )

def streaming_message_component(message_id: str) -> Div:
    """Create a message component for streaming responses."""
    timestamp = datetime.now()

    return Card(
        DivLAligned(
            # Avatar
            Div(
                Span("🤖", cls="text-lg"),
                cls="flex h-10 w-10 items-center justify-center rounded-full bg-green-500 text-white",  # noqa: E501
            ),
            # Message content
            Div(
                DivFullySpaced(
                    Strong("Assistant", cls="text-sm"),
                    Small(format_timestamp(timestamp), cls=TextPresets.muted_sm),
                ),
                # Container for live reasoning steps (tree structure) - FIRST
                Details(
                    Summary(
                        Span(
                            "🧠 Reasoning Process",
                            cls="font-semibold text-gray-700 text-sm cursor-pointer",
                        ),
                        cls="mb-2 pb-1 border-b border-gray-200",
                    ),
                    Div(
                        id=f"{message_id}-reasoning-steps",
                        cls="space-y-1 mt-2",
                    ),
                    id=f"{message_id}-reasoning-container",
                    cls="hidden mt-3 p-3 bg-gray-50 rounded-lg",
                    open=True,  # Start expanded
                ),
                # Final response content - SECOND
                Div(
                    Div(
                        Span("💬 Final Answer", cls="font-semibold text-gray-700 text-sm"),
                        cls="mb-2 pb-1 border-b border-gray-200",
                    ),
                    Div(
                        Span(id=f"{message_id}-content", cls="assistant-streaming-content"),
                        cls="mt-2",
                    ),
                    id=f"{message_id}-answer-container",
                    cls="hidden mt-3 p-3 bg-blue-50 rounded-lg",
                ),
                cls="ml-3 flex-1",
            ),
        ),
        cls="max-w-4xl mr-auto mb-4 bg-green-50 border-green-200",
        id=message_id,
    )

def reasoning_step_component(reasoning_event: dict, step_num: int) -> Div:  # noqa: PLR0912, PLR0915
    """Render a reasoning step as a tree node."""
    event_type = reasoning_event.get("type", "unknown")
    step_iteration = reasoning_event.get("step_iteration", step_num)
    metadata = reasoning_event.get("metadata", {})
    tools = metadata.get("tools", [])

    # Handle new event types from ReasoningEventType enum
    if event_type == "iteration_start":
        content = [
            Div(
                Span(f"🚀 Step {step_iteration} Started", cls="font-semibold text-blue-600"),
                cls="flex items-center mb-2",
            ),
        ]

        # Only add thought if one was provided
        if metadata.get("thought"):
            content.append(
                Div(
                    P(f"💭 {metadata['thought']}", cls="text-gray-700"),
                    cls="ml-4 mb-2",
                ),
            )

    elif event_type == "planning":
        thought = metadata.get("thought", "Planning next actions...")
        tools_planned = metadata.get("tools_planned", tools)  # Fallback to tools
        content = [
            Div(
                Span(f"🤔 Step {step_iteration} Planning", cls="font-semibold text-purple-600"),
                cls="flex items-center mb-2",
            ),
            Div(
                P(f"💭 {thought}", cls="text-gray-700"),
                cls="ml-4 mb-2",
            ),
        ]

        if tools_planned:
            content.append(
                Div(
                    P("🔧 Tools planned:", cls="font-medium text-gray-600 mb-1"),
                    *[
                        P(f"• {tool_name}", cls="ml-4 text-sm text-gray-600")
                        for tool_name in tools_planned
                    ],
                    cls="ml-4 mb-2",
                ),
            )

    elif event_type == "tool_execution_start":
        tool_predictions = metadata.get("tool_predictions", [])
        concurrent_execution = metadata.get("concurrent_execution", False)
        execution_mode = "concurrently" if concurrent_execution else "sequentially"

        content = [
            Div(
                Span(
                    f"⚡ Executing Tools ({execution_mode})",
                    cls="font-semibold text-orange-600",
                ),
                cls="flex items-center mb-2",
            ),
        ]

        # Show tools from predictions if available, otherwise use main tools field
        if tool_predictions:
            for tool_pred in tool_predictions:
                if isinstance(tool_pred, dict):
                    tool_name = tool_pred.get("tool_name", "Unknown")
                    reasoning = tool_pred.get("reasoning", "No reasoning provided")
                    content.append(
                        Div(
                            P(f"📦 {tool_name}", cls="font-mono text-sm font-bold"),
                            P(f"💡 {reasoning}", cls="text-sm text-gray-600 mt-1"),
                            cls="ml-4 mb-2 p-2 bg-orange-50 rounded",
                        ),
                    )
        elif tools:
            for tool_name in tools:
                content.append(
                    Div(
                        P(f"📦 {tool_name}", cls="font-mono text-sm font-bold"),
                        P("Executing...", cls="text-sm text-gray-600 mt-1"),
                        cls="ml-4 mb-2 p-2 bg-orange-50 rounded",
                    ),
                )

    elif event_type == "tool_result":
        tool_results = metadata.get("tool_results", [])
        content = [
            Div(
                Span("✅ Tool Execution Complete", cls="font-semibold text-green-600"),
                cls="flex items-center mb-2",
            ),
        ]

        # Show results if available, otherwise use main tools field
        if tool_results:
            for result in tool_results:
                if isinstance(result, dict):
                    tool_name = result.get("tool_name", "Unknown")
                    success = result.get("success", False)
                    result_data = result.get("result", result.get("error", "No result"))

                    bg_color = "bg-green-50" if success else "bg-red-50"
                    status_text = "✅ SUCCESS" if success else "❌ FAILED"
                    content.append(
                        Div(
                            P(f"📦 {tool_name}: {status_text}", cls="font-mono text-sm font-bold"),
                            P(
                                str(result_data)[:200] + (
                                    "..." if len(str(result_data)) > 200 else ""
                                ),
                                cls="text-sm text-gray-600 mt-1 font-mono",
                            ),
                            cls=f"ml-4 mb-2 p-2 {bg_color} rounded",
                        ),
                    )
        elif tools:
            # Fallback to showing tools from main field if no detailed results
            for tool_name in tools:
                content.append(
                    Div(
                        P(f"📦 {tool_name}: ✅ COMPLETED", cls="font-mono text-sm font-bold"),
                        P("Tool execution completed", cls="text-sm text-gray-600 mt-1"),
                        cls="ml-4 mb-2 p-2 bg-green-50 rounded",
                    ),
                )

    elif event_type == "iteration_complete":
        had_tools = metadata.get("had_tools", False)
        content = [
            Div(
                Span(f"✅ Step {step_iteration} Complete", cls="font-semibold text-green-600"),
                cls="flex items-center mb-2",
            ),
        ]

        # Only add thought if one was provided
        if metadata.get("thought"):
            content.append(
                Div(
                    P(f"💭 {metadata['thought']}", cls="text-gray-700"),
                    cls="ml-4 mb-2",
                ),
            )

        if had_tools:
            content.append(
                Div(
                    P("🔧 Tools were executed in this step", cls="text-sm text-green-600"),
                    cls="ml-4",
                ),
            )

    elif event_type == "synthesis_complete":
        total_steps = metadata.get("total_steps", 0)
        content = [
            Div(
                Span("🎯 Synthesis Complete", cls="font-semibold text-purple-600"),
                cls="flex items-center mb-2",
            ),
            Div(
                P(f"✅ Completed {total_steps} reasoning steps", cls="text-gray-700"),
                P("🔄 Generating final response...", cls="text-sm text-gray-500 mt-1"),
                cls="ml-4",
            ),
        ]

    elif event_type == "error":
        error_msg = reasoning_event.get("error", "Unknown error occurred")
        content = [
            Div(
                Span("❌ Error", cls="font-semibold text-red-600"),
                cls="flex items-center mb-2",
            ),
            Div(
                P(f"⚠️ {error_msg}", cls="text-red-700"),
                cls="ml-4 mb-2 p-2 bg-red-50 rounded",
            ),
        ]

    else:
        # Generic event with better formatting
        content = [
            Div(
                Span(f"📋 {event_type.upper()}", cls="font-semibold text-gray-600"),
                cls="flex items-center mb-2",
            ),
            Details(
                Summary("Show raw data", cls="text-sm text-gray-500 cursor-pointer"),
                Pre(Code(json.dumps(reasoning_event, indent=2), cls="text-xs")),
                cls="ml-4",
            ),
        ]

    return Div(
        *content,
        cls="border-l-2 border-gray-200 pl-3 py-2 mb-2",
    )

def power_user_panel() -> Div:
    """Power user settings panel for left side."""
    return Card(
        CardBody(
            Form(
                LabelTextArea(
                    "System Prompt",
                    id="system_prompt",
                    name="system_prompt",
                    placeholder="You are a helpful AI assistant...",
                    rows=8,
                    value="",
                ),
                Divider(),
                Grid(
                    Select(
                        Option("gpt-4o-mini", value="gpt-4o-mini", selected=True),
                        Option("gpt-4o", value="gpt-4o"),
                        id="model",
                        name="model",
                    ),
                    Br(),
                    LabelRange(
                        "Temperature",
                        id="temperature",
                        name="temperature",
                        min="0", max="2", step="0.1", value="0.2",
                        label_range=True,
                    ),
                    LabelRange(
                        "Max Tokens",
                        id="max_tokens",
                        name="max_tokens",
                        min="1", max="4000", step="1", value="1000",
                        label_range=True,
                    ),
                    cols=1,
                    cls="gap-4",
                ),
                id="power-user-form",
                cls="space-y-4",
            ),
        ),
        cls="h-full",
    )

def main_chat_interface() -> Div:
    """Main chat interface layout with split screen."""
    return Grid(
        # Left panel - Power user settings (1/3 width)
        Div(
            power_user_panel(),
            cls="h-screen overflow-y-auto p-4",
        ),

        # Right panel - Chat interface (2/3 width)
        Div(
            # Chat messages area with fixed height and scrolling
            Div(
                Div(
                    P("Start a conversation by typing a message below.",
                      cls=TextPresets.muted_sm + " text-center py-8"),
                    id="chat-messages",
                    cls="space-y-4 p-4",
                ),
                cls="bg-white border rounded-lg shadow-sm overflow-y-auto",
                style="height: calc(100vh - 180px); max-height: calc(100vh - 180px);",
            ),

            # Input form with loading indicator
            Div(
                # Loading indicator
                Div(
                    Loading(cls=LoadingT.dots),
                    P("AI is thinking...", cls=TextPresets.muted_sm + " ml-2"),
                    id="loading-indicator",
                    cls="htmx-indicator flex items-center justify-center py-2",
                ),
                # Input form
                Div(
                    Form(
                        DivLAligned(
                            Input(
                                name="message",
                                id="message_input",
                                placeholder="Ask me anything... (e.g., 'What's the weather like?')",  # noqa: E501
                                cls="flex-1",
                                required=True,
                            ),
                            Button(
                                UkIcon("send", cls="mr-2"),
                                "Send",
                                cls=ButtonT.primary,
                                type="submit",
                            ),
                            Button(
                                "Clear",
                                cls=ButtonT.secondary,
                                hx_post="/clear_chat",
                                hx_target="#chat-messages",
                                hx_swap="innerHTML",
                                hx_confirm="Are you sure you want to clear the chat?",
                            ),
                            cls="space-x-2",
                        ),
                        # HTMX attributes for real-time updates
                        hx_post="/send_message",
                        hx_target="#chat-messages",
                        hx_swap="beforeend",
                        hx_indicator="#loading-indicator",
                        hx_include="#power-user-form",  # Include power user settings
                    ),
                    cls="bg-white border rounded-lg shadow-sm p-4",
                ),
                cls="mt-4",
            ),

            cls="flex flex-col h-screen p-4",
        ),

        # Grid layout - 1/3 for power user, 2/3 for chat
        cols=3,
        cls="w-full",
        style="grid-template-columns: 1fr 2fr;",
    )

@rt("/")
def homepage():  # noqa: ANN201
    """Main page."""
    return Div(
        # Load HTMX SSE extension
        Script(src="https://unpkg.com/htmx.org@1.9.12/dist/ext/sse.js"),

        # Add custom styles and scripts
        Style("""
            .assistant-streaming-content::after {
                content: '▊';
                animation: blink 1s infinite;
            }
            @keyframes blink {
                0%, 50% { opacity: 1; }
                51%, 100% { opacity: 0; }
            }
            body {
                margin: 0;
                padding: 0;
                overflow: hidden;
            }
        """),

        Script("""
            // Auto-scroll chat messages
            function scrollToBottom() {
                // Target the scrollable container (parent of chat-messages)
                const chatMessages = document.getElementById('chat-messages');
                const scrollContainer = chatMessages ? chatMessages.parentElement : null;

                if (scrollContainer) {
                    scrollContainer.scrollTop = scrollContainer.scrollHeight;
                    // Force scroll with slight delay for dynamic content
                    setTimeout(() => {
                        scrollContainer.scrollTop = scrollContainer.scrollHeight;
                    }, 10);
                }
            }

            // Clear input after sending
            htmx.on('htmx:afterRequest', function(evt) {
                if (evt.detail.pathInfo.requestPath === '/send_message') {
                    document.getElementById('message_input').value = '';
                }
            });

            // Observe mutations to auto-scroll
            const observer = new MutationObserver(scrollToBottom);
            const chatMessages = document.getElementById('chat-messages');
            if (chatMessages) {
                observer.observe(chatMessages, {
                    childList: true,
                    subtree: true,
                    characterData: true,
                    attributes: true
                });
            }

            // Also scroll on window resize
            window.addEventListener('resize', scrollToBottom);
        """),

        main_chat_interface(),
    )

@rt("/send_message", methods=["POST"])
async def send_message(message: str, system_prompt: str = "", temperature: str = "0.2",  # noqa: ANN201
                      max_tokens: str = "1000", model: str = "gpt-4o-mini"):
    """Handle message sending with streaming response."""
    if not message.strip():
        return ""

    try:
        # Create user message and unique stream ID
        user_msg = ChatMessage("user", message.strip())
        user_msg_id = f"msg-user-{datetime.now().timestamp()}"
        stream_id = f"stream-{datetime.now().timestamp()}"
        ai_msg_id = f"msg-ai-{stream_id}"

        # Store user message in conversation history
        global conversations  # noqa: PLW0602
        conversations["default"].append(user_msg.to_dict())

        # Store stream parameters for the SSE endpoint
        stream_data = {
            "message": message.strip(),
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "model": model,
        }

        # Store in global dict
        global active_streams  # noqa: PLW0602
        active_streams[stream_id] = stream_data

        # Return user message and streaming AI placeholder
        return Div(
            chat_message_component(user_msg, user_msg_id),
            streaming_message_component(ai_msg_id),
            Script(f"""
                (function() {{
                    // Clear placeholder text with unique scope
                    const placeholder_{stream_id.replace('-', '_').replace('.', '_')} = document.querySelector('#chat-messages .text-center');
                    if (placeholder_{stream_id.replace('-', '_').replace('.', '_')}) placeholder_{stream_id.replace('-', '_').replace('.', '_')}.remove();

                    // Scroll to bottom
                    scrollToBottom();

                    // Set up EventSource manually for better control
                    const eventSource_{stream_id.replace('-', '_').replace('.', '_')} = new EventSource('/stream/{stream_id}');

                    eventSource_{stream_id.replace('-', '_').replace('.', '_')}.addEventListener('chunk', function(event) {{
                        // Show answer container if first chunk
                        const answerContainer = document.getElementById('{ai_msg_id}-answer-container');
                        if (answerContainer && answerContainer.classList.contains('hidden')) {{
                            answerContainer.classList.remove('hidden');
                        }}

                        const contentEl = document.getElementById('{ai_msg_id}-content');
                        if (contentEl) {{
                            contentEl.insertAdjacentHTML('beforeend', event.data);
                            scrollToBottom();
                        }}
                    }});

                    eventSource_{stream_id.replace('-', '_').replace('.', '_')}.addEventListener('reasoning_step', function(event) {{
                        // Show reasoning container if first step
                        const reasoningContainer = document.getElementById('{ai_msg_id}-reasoning-container');
                        if (reasoningContainer && reasoningContainer.classList.contains('hidden')) {{
                            reasoningContainer.classList.remove('hidden');
                        }}

                        // Add new reasoning step to the accordion
                        const reasoningSteps = document.getElementById('{ai_msg_id}-reasoning-steps');
                        if (reasoningSteps) {{
                            reasoningSteps.insertAdjacentHTML('beforeend', event.data);
                            scrollToBottom();
                        }}
                    }});

                    eventSource_{stream_id.replace('-', '_').replace('.', '_')}.addEventListener('reasoning_complete', function(event) {{
                        // Optional: Could add final styling or cleanup here
                        scrollToBottom();
                    }});

                    eventSource_{stream_id.replace('-', '_').replace('.', '_')}.addEventListener('complete', function(event) {{
                        const contentEl = document.getElementById('{ai_msg_id}-content');
                        if (contentEl) {{
                            contentEl.classList.remove('assistant-streaming-content');
                        }}
                        eventSource_{stream_id.replace('-', '_').replace('.', '_')}.close();
                    }});

                    eventSource_{stream_id.replace('-', '_').replace('.', '_')}.addEventListener('error', function(event) {{
                        eventSource_{stream_id.replace('-', '_').replace('.', '_')}.close();
                    }});

                    // Clean up on page unload
                    window.addEventListener('beforeunload', function() {{
                        eventSource_{stream_id.replace('-', '_').replace('.', '_')}.close();
                    }});
                }})();
            """),  # noqa: E501
        )

    except Exception as e:
        # Error handling
        user_msg = ChatMessage("user", message.strip())
        error_msg = ChatMessage("system", f"Error: {e!s}")

        return Div(
            chat_message_component(user_msg, f"error-user-{datetime.now().timestamp()}"),
            chat_message_component(error_msg, f"error-system-{datetime.now().timestamp()}"),
        )


@rt("/stream/{stream_id}")
async def stream_chat(stream_id: str):  # noqa: ANN201, PLR0915
    """SSE endpoint for streaming chat responses."""

    async def event_generator():  # noqa: ANN202, PLR0912, PLR0915
        try:
            # Get stream parameters
            global active_streams  # noqa: PLW0602
            if stream_id not in active_streams:
                yield "event: error\ndata: Stream not found\n\n"
                return

            stream_data = active_streams[stream_id]
            stream_data["message"]
            system_prompt = stream_data.get("system_prompt", "")
            temperature = float(stream_data.get("temperature", "0.2"))
            max_tokens = int(stream_data.get("max_tokens", "1000"))
            model = stream_data.get("model", "gpt-4o-mini")


            # Prepare request to reasoning API with conversation history
            messages = []
            if system_prompt and system_prompt.strip():
                messages.append({"role": "system", "content": system_prompt.strip()})

            # Add conversation history (excluding timestamps)
            global conversations  # noqa: PLW0602
            conversation_history = conversations.get("default", [])
            for msg in conversation_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

            request_payload = {
                "model": model,
                "messages": messages,
                "stream": True,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            headers = {
                "Content-Type": "application/json",
                "X-Session-ID": session_id,  # Session ID for tracing correlation
            }

            # Only add Authorization header if token is provided
            if REASONING_API_TOKEN:
                headers["Authorization"] = f"Bearer {REASONING_API_TOKEN}"

            reasoning_steps = []
            assistant_response = ""  # Track full response

            async with httpx.AsyncClient(timeout=60.0) as client:  # noqa: SIM117
                async with client.stream(
                    "POST",
                    f"{REASONING_API_URL}/v1/chat/completions",
                    headers=headers,
                    json=request_payload,
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data.strip() == "[DONE]":
                                break

                            try:
                                event = json.loads(data)

                                if "choices" in event:
                                    delta = event["choices"][0].get("delta", {})

                                    # Check for reasoning event in delta
                                    if "reasoning_event" in delta:
                                        # Reasoning event - stream immediately
                                        reasoning_event = delta["reasoning_event"]
                                        reasoning_steps.append(reasoning_event)

                                        # Create reasoning step HTML and stream it
                                        step_html = reasoning_step_component(
                                            reasoning_event,
                                            len(reasoning_steps),
                                        )
                                        yield f"event: reasoning_step\ndata: {step_html!s}\n\n"

                                    # Check for regular content (non-empty)
                                    elif delta.get("content"):
                                        content_chunk = delta["content"]
                                        assistant_response += content_chunk  # Track full response

                                        # Escape for HTML
                                        escaped_chunk = (content_chunk
                                                       .replace('&', '&amp;')
                                                       .replace('<', '&lt;')
                                                       .replace('>', '&gt;')
                                                       .replace('\n', '<br>'))

                                        # Send chunk directly as HTML
                                        yield f"event: chunk\ndata: {escaped_chunk}\n\n"

                            except json.JSONDecodeError:
                                continue

            # Finalize reasoning steps container if any steps were streamed
            if reasoning_steps:
                # Send signal to close/finalize the reasoning accordion
                yield "event: reasoning_complete\ndata: done\n\n"

            # Store assistant response in conversation history
            if assistant_response.strip():
                assistant_msg = ChatMessage("assistant", assistant_response.strip())
                conversations["default"].append(assistant_msg.to_dict())

            # Send completion signal
            yield "event: complete\ndata: done\n\n"

            # Clean up
            active_streams.pop(stream_id, None)

        except Exception as e:
            yield f"event: error\ndata: Error: {e!s}\n\n"
            # Clean up on error
            active_streams.pop(stream_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )

@rt("/clear_chat", methods=["POST"])
async def clear_chat():  # noqa: ANN201
    """Clear the chat messages."""
    global conversations  # noqa: PLW0602
    conversations["default"] = []  # Clear conversation history
    return P("Chat cleared. Start a new conversation!",
             cls=TextPresets.muted_sm + " text-center py-8")

@rt("/health")
async def health_check():  # noqa: ANN201
    """Health check endpoint."""
    return {"status": "healthy", "service": "reasoning-agent-web-client"}

if __name__ == "__main__":
    import uvicorn

    print(f"🚀 Starting Reasoning Agent Web Client on port {WEB_CLIENT_PORT}")
    print(f"📡 Connecting to API at: {REASONING_API_URL}")
    print(f"🌐 Web interface will be available at: http://localhost:{WEB_CLIENT_PORT}")
    print(f"Debug: Port = {WEB_CLIENT_PORT}, Host = 0.0.0.0")

    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=WEB_CLIENT_PORT,
            reload=False,  # Changed from True to avoid import string warning
        )
    except Exception as e:
        print(f"Error starting server: {e}")
        print("Try running: lsof -ti :8080 | xargs kill to kill any existing processes")
