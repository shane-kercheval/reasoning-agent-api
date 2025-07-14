#!/usr/bin/env python3
"""
MonsterUI Web Client for Reasoning Agent API.

A beautiful web interface for streaming conversations with collapsible reasoning steps
and power user controls. Built with MonsterUI and FastHTML.
"""

import os
import json
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
API_TOKEN = os.getenv("API_TOKEN", "token1")
WEB_CLIENT_PORT = int(os.getenv("WEB_CLIENT_PORT", 8080))

# Global state for conversations
conversations: dict[str, list[dict]] = {}
reasoning_events: dict[str, list[dict]] = {}
active_streams: dict[str, dict] = {}

class ChatMessage:
    """Represents a chat message."""

    def __init__(self, role: str, content: str, timestamp: datetime | None = None):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }

def format_timestamp(timestamp: datetime) -> str:
    """Format timestamp for display."""
    return timestamp.strftime("%H:%M:%S")

def chat_message_component(message: ChatMessage, message_id: str, reasoning_steps: list = None) -> Div:
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
                    *[P(line) if line and line.strip() else Br() for line in (message.content or "").split('\n')],
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
                cls="flex h-10 w-10 items-center justify-center rounded-full bg-green-500 text-white",
            ),
            # Message content
            Div(
                DivFullySpaced(
                    Strong("Assistant", cls="text-sm"),
                    Small(format_timestamp(timestamp), cls=TextPresets.muted_sm),
                ),
                Div(
                    Span(id=f"{message_id}-content", cls="assistant-streaming-content"),
                    cls="mt-2",
                ),
                cls="ml-3 flex-1",
            ),
        ),
        # Placeholder for reasoning steps
        Div(id=f"{message_id}-reasoning", cls="ml-14"),
        cls="max-w-4xl mr-auto mb-4 bg-green-50 border-green-200",
        id=message_id,
    )

def reasoning_step_component(event: dict, step_num: int) -> Div:
    """Render a reasoning step in an accordion."""
    event_type = event.get("type", "unknown")
    content = event.get("content", event)  # Some events have content wrapper, others don't

    if event_type == "reasoning" or "thought" in content:
        # Reasoning step
        thought = content.get("thought", "No thought provided")
        action = content.get("next_action", content.get("action", "unknown"))
        tools = content.get("tools_to_use", [])

        summary = f"🧠 Thinking: {thought[:60]}{'...' if len(thought) > 60 else ''}"

        details_content = [
            P(f"💭 Thought: {thought}", cls="mb-3"),
            P(f"⚡ Next Action: {action}", cls="mb-3"),
        ]

        if tools:
            details_content.append(P("🔧 Tools to use:", cls="mb-2 font-medium"))
            for i, tool in enumerate(tools):
                tool_info = Card(
                    P(f"📦 {tool.get('server_name', 'unknown')}.{tool.get('tool_name', 'unknown')}",
                      cls="font-mono text-sm font-bold"),
                    P(f"💡 Reasoning: {tool.get('reasoning', 'No reasoning provided')}",
                      cls="text-sm mt-2"),
                    P(f"⚙️ Arguments: {json.dumps(tool.get('arguments', {}), indent=2)}",
                      cls="text-xs mt-2 font-mono bg-gray-100 p-2 rounded"),
                    cls="mt-2 p-3 bg-blue-50 border-blue-200",
                )
                details_content.append(tool_info)

        return AccordionItem(summary, Div(*details_content))

    if event_type == "tool_result" or "tool_name" in content:
        # Tool execution result
        tool_name = content.get("tool_name", "Unknown Tool")
        result = content.get("result", content.get("content", "No result"))
        server_name = content.get("server_name", "")

        summary = f"🔧 Tool: {server_name}.{tool_name}" if server_name else f"🔧 Tool: {tool_name}"

        # Format result based on type
        if isinstance(result, dict):
            result_display = Pre(Code(json.dumps(result, indent=2), cls="text-sm"))
        elif isinstance(result, str) and len(result) > 200:
            result_display = Details(
                Summary("Show full result"),
                Pre(Code(result, cls="text-sm max-h-64 overflow-y-auto")),
            )
        else:
            result_display = P(str(result), cls="font-mono bg-gray-100 p-2 rounded text-sm")

        return AccordionItem(
            summary,
            Div(
                P("📤 Result:", cls="font-medium mb-2"),
                result_display,
            ),
        )

    if event_type == "error":
        # Error event
        error_msg = content.get("message", str(content))
        return AccordionItem(
            "❌ Error",
            Card(
                P(error_msg, cls="text-red-600"),
                cls="bg-red-50 border-red-200 p-3",
            ),
        )

    # Generic event
    return AccordionItem(
        f"📋 {event_type.title()}",
        Details(
            Summary("Show raw event data"),
            Pre(Code(json.dumps(event, indent=2), cls="text-sm")),
        ),
    )

def power_user_panel() -> Div:
    """Power user settings panel for left side."""
    return Card(
        CardBody(
            Form(
                LabelTextArea(
                    "System Prompt",
                    id="system_prompt",
                    placeholder="You are a helpful AI assistant...",
                    rows=8,
                    value="",
                ),
                Divider(),
                Grid(
                    LabelRange(
                        "Temperature",
                        id="temperature",
                        min="0", max="2", step="0.1", value="0.7",
                        label_range=True,
                    ),
                    LabelRange(
                        "Max Tokens",
                        id="max_tokens",
                        min="1", max="4000", step="1", value="1000",
                        label_range=True,
                    ),
                    cols=1,
                    cls="gap-4",
                ),
                P("Model: gpt-4o-mini", cls=TextPresets.muted_sm + " mt-4"),
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
            # Chat messages area
            Div(
                Div(
                    P("Start a conversation by typing a message below.",
                      cls=TextPresets.muted_sm + " text-center py-8"),
                    id="chat-messages",
                    cls="space-y-4 overflow-y-auto p-4 h-full",
                ),
                cls="flex-1 bg-white border rounded-lg shadow-sm mb-4 min-h-0",
                style="height: calc(100vh - 200px);",
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
                                placeholder="Ask me anything... (e.g., 'What's the weather like?')",
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
                    ),
                    cls="bg-white border rounded-lg shadow-sm p-4",
                ),
                cls="flex-shrink-0",
            ),

            cls="flex flex-col h-screen p-4",
        ),
        
        # Grid layout - 1/3 for power user, 2/3 for chat
        cols=3,
        cls="w-full",
        style="grid-template-columns: 1fr 2fr;",
    )

@rt("/")
def homepage():
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
            // Auto-scroll chat messages with debugging
            function scrollToBottom() {
                const chatMessages = document.getElementById('chat-messages');
                if (chatMessages) {
                    console.log('Scrolling - current:', chatMessages.scrollTop, 'max:', chatMessages.scrollHeight);
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                    // Force scroll with slight delay for dynamic content
                    setTimeout(() => {
                        chatMessages.scrollTop = chatMessages.scrollHeight;
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
            const observer = new MutationObserver(function(mutations) {
                console.log('DOM mutation detected, scrolling...');
                scrollToBottom();
            });
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
async def send_message(message: str, system_prompt: str = "", temperature: str = "0.7", 
                      max_tokens: str = "1000"):
    """Handle message sending with streaming response."""
    if not message.strip():
        return ""

    try:
        # Create user message and unique stream ID
        user_msg = ChatMessage("user", message.strip())
        user_msg_id = f"msg-user-{datetime.now().timestamp()}"
        stream_id = f"stream-{datetime.now().timestamp()}"
        ai_msg_id = f"msg-ai-{stream_id}"
        
        # Store stream parameters for the SSE endpoint
        stream_data = {
            "message": message.strip(),
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        print(f"[DEBUG] Creating stream {stream_id} with message: '{message.strip()}'")
        
        # Store in global dict
        global active_streams
        active_streams[stream_id] = stream_data
        print(f"[DEBUG] Active streams: {list(active_streams.keys())}")
        
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
                    console.log('Setting up EventSource for stream: {stream_id}');
                    const eventSource_{stream_id.replace('-', '_').replace('.', '_')} = new EventSource('/stream/{stream_id}');
                    
                    eventSource_{stream_id.replace('-', '_').replace('.', '_')}.addEventListener('open', function(event) {{
                        console.log('EventSource opened for {stream_id}');
                    }});
                    
                    eventSource_{stream_id.replace('-', '_').replace('.', '_')}.addEventListener('chunk', function(event) {{
                        console.log('Received chunk:', event.data);
                        const contentEl = document.getElementById('{ai_msg_id}-content');
                        if (contentEl) {{
                            contentEl.insertAdjacentHTML('beforeend', event.data);
                            scrollToBottom();
                        }}
                    }});
                    
                    eventSource_{stream_id.replace('-', '_').replace('.', '_')}.addEventListener('reasoning', function(event) {{
                        console.log('Received reasoning');
                        const reasoningEl = document.getElementById('{ai_msg_id}-reasoning');
                        if (reasoningEl) {{
                            reasoningEl.innerHTML = event.data;
                            scrollToBottom();
                        }}
                    }});
                    
                    eventSource_{stream_id.replace('-', '_').replace('.', '_')}.addEventListener('complete', function(event) {{
                        console.log('Stream complete for {stream_id}');
                        const contentEl = document.getElementById('{ai_msg_id}-content');
                        if (contentEl) {{
                            contentEl.classList.remove('assistant-streaming-content');
                        }}
                        eventSource_{stream_id.replace('-', '_').replace('.', '_')}.close();
                    }});
                    
                    eventSource_{stream_id.replace('-', '_').replace('.', '_')}.addEventListener('error', function(event) {{
                        console.error('SSE Error for {stream_id}:', event);
                        eventSource_{stream_id.replace('-', '_').replace('.', '_')}.close();
                    }});
                    
                    // Clean up on page unload
                    window.addEventListener('beforeunload', function() {{
                        eventSource_{stream_id.replace('-', '_').replace('.', '_')}.close();
                    }});
                }})();
            """)
        )

    except Exception as e:
        # Error handling
        user_msg = ChatMessage("user", message.strip())
        error_msg = ChatMessage("system", f"Error: {str(e)}")
        
        return Div(
            chat_message_component(user_msg, f"error-user-{datetime.now().timestamp()}"),
            chat_message_component(error_msg, f"error-system-{datetime.now().timestamp()}"),
        )


@rt("/stream/{stream_id}")
async def stream_chat(stream_id: str):
    """SSE endpoint for streaming chat responses."""
    
    async def event_generator():
        try:
            print(f"[DEBUG] Starting stream for {stream_id}")
            
            # Get stream parameters
            global active_streams
            if stream_id not in active_streams:
                print(f"[DEBUG] Stream {stream_id} not found in active_streams")
                yield f"event: error\ndata: Stream not found\n\n"
                return
                
            stream_data = active_streams[stream_id]
            message = stream_data["message"]
            system_prompt = stream_data.get("system_prompt", "")
            temperature = float(stream_data.get("temperature", "0.7"))
            max_tokens = int(stream_data.get("max_tokens", "1000"))
            
            print(f"[DEBUG] Stream data: message='{message}', temp={temperature}")
            
            ai_msg_id = f"msg-ai-{stream_id}"
            
            # Prepare request to reasoning API
            messages = []
            if system_prompt and system_prompt.strip():
                messages.append({"role": "system", "content": system_prompt.strip()})
            messages.append({"role": "user", "content": message})
            
            request_payload = {
                "model": "gpt-4o-mini",
                "messages": messages,
                "stream": True,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            headers = {
                "Authorization": f"Bearer {API_TOKEN}",
                "Content-Type": "application/json",
            }

            reasoning_steps = []
            
            async with httpx.AsyncClient(timeout=60.0) as client:
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
                                    # Regular completion chunk
                                    delta = event["choices"][0].get("delta", {})
                                    if "content" in delta and delta["content"] is not None:
                                        content_chunk = delta["content"]
                                        print(f"[DEBUG] Sending chunk: '{content_chunk}'")
                                        
                                        # Escape for HTML 
                                        escaped_chunk = (content_chunk
                                                       .replace('&', '&amp;')
                                                       .replace('<', '&lt;')
                                                       .replace('>', '&gt;')
                                                       .replace('\n', '<br>'))
                                        
                                        # Send chunk directly as HTML
                                        yield f"event: chunk\ndata: {escaped_chunk}\n\n"
                                else:
                                    # Reasoning event
                                    reasoning_steps.append(event)
                                    
                            except json.JSONDecodeError:
                                continue
            
            # Send reasoning steps if any (using separate SSE event)
            if reasoning_steps:
                print(f"[DEBUG] Sending {len(reasoning_steps)} reasoning steps")
                reasoning_accordion = Details(
                    Summary("🧠 View reasoning steps", cls="cursor-pointer text-sm text-gray-600 mt-3"),
                    Accordion(
                        *[reasoning_step_component(step, i+1) for i, step in enumerate(reasoning_steps)],
                        cls="mt-2",
                    ),
                )
                reasoning_html = str(reasoning_accordion)
                yield f"event: reasoning\ndata: {reasoning_html}\n\n"
            
            # Send completion signal
            print(f"[DEBUG] Sending completion for {stream_id}")
            yield f"event: complete\ndata: done\n\n"
            
            # Clean up
            if stream_id in active_streams:
                print(f"[DEBUG] Cleaning up stream {stream_id}")
                del active_streams[stream_id]
            
        except Exception as e:
            print(f"[DEBUG] Error in stream {stream_id}: {str(e)}")
            yield f"event: error\ndata: Error: {str(e)}\n\n"
            # Clean up on error
            if stream_id in active_streams:
                del active_streams[stream_id]
    
    return StreamingResponse(
        event_generator(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )

@rt("/clear_chat", methods=["POST"])
async def clear_chat():
    """Clear the chat messages."""
    return P("Chat cleared. Start a new conversation!",
             cls=TextPresets.muted_sm + " text-center py-8")

@rt("/health")
async def health_check():
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