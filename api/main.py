"""
FastAPI application for the Reasoning Agent API.

This module provides an OpenAI-compatible chat completion API that enhances requests
with reasoning capabilities. The API uses a streaming-only architecture for all responses.
"""


import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from uuid import UUID

import httpx
import litellm
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry import trace, context
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues
from opentelemetry.trace import set_span_in_context

# Note: You may see "Failed to detach context" errors logged to stderr
# This is a known OpenTelemetry issue with async generators and contextvars
# (see: https://github.com/open-telemetry/opentelemetry-python/issues/2606)
# The errors are logged but don't prevent responses from completing successfully.
from .openai_protocol import (
    OpenAIChatRequest,
    ModelsResponse,
    ModelInfo,
    ErrorResponse,
)
from .dependencies import (
    service_container,
    ToolsDependency,
    PromptManagerDependency,
    ConversationDBDependency,
)
from .auth import verify_token
from .config import settings
from .tracing import setup_tracing
from .request_router import determine_routing, RoutingMode
from .executors.passthrough import PassthroughExecutor
from .executors.reasoning_agent import ReasoningAgent


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:  # noqa: ARG001
    """
    Manage application lifespan events.

    NOTE: This runs once when the server starts and after it stops (yields control).
    """
    # Always initialize tracing (will be no-op if disabled)
    setup_tracing(
        enabled=settings.enable_tracing,
        project_name=settings.phoenix_project_name,
        endpoint=settings.phoenix_collector_endpoint,
        enable_console_export=settings.enable_console_tracing,
    )

    await service_container.initialize()
    try:
        yield
    finally:
        # SHUTDOWN: This runs once when server stops
        await service_container.cleanup()


app = FastAPI(
    title="Reasoning Agent API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get tracer and logger for request instrumentation
tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)


def extract_system_message(messages: list[dict[str, object]]) -> str | None:
    """
    Extract the first system message from the messages list.

    Args:
        messages: List of message dictionaries with 'role' and 'content' fields

    Returns:
        System message content if found, None otherwise

    Example:
        >>> messages = [
        ...     {"role": "system", "content": "You are a helpful assistant."},
        ...     {"role": "user", "content": "Hello"}
        ... ]
        >>> extract_system_message(messages)
        'You are a helpful assistant.'
    """
    system_msgs = [m for m in messages if m.get("role") == "system"]
    if not system_msgs:
        return None
    # Get content from first system message
    content = system_msgs[0].get("content")
    return str(content) if content is not None else None


def safe_detach_context(token: object) -> None:
    """
    Safely detach OpenTelemetry context token.

    Handles the case where the token was created in a different asyncio context,
    which can happen during concurrent request cancellation.
    """
    with suppress(ValueError):
        context.detach(token)


def span_cleanup(span: trace.Span, token: object) -> None:
    """
    End span and safely detach context token.

    This helper ensures we always clean up both the span and context together,
    preventing resource leaks and maintaining proper tracing hygiene.

    Only performs cleanup if the span is still recording, preventing
    duplicate cleanup in cases where the span was already ended.
    """
    if span.is_recording():
        span.end()
        safe_detach_context(token)


@app.get("/v1/models")
async def list_models(
    _: bool = Depends(verify_token),
) -> ModelsResponse:
    """
    List available models from LiteLLM proxy.

    Proxies to LiteLLM's /v1/models endpoint to provide dynamic model discovery.

    Requires authentication via bearer token.
    """
    try:
        # Call LiteLLM proxy's /v1/models endpoint to get available models
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.llm_base_url}/v1/models",
                headers={"Authorization": f"Bearer {settings.llm_api_key}"},
                timeout=5.0,
            )
            response.raise_for_status()

            # Parse LiteLLM response
            data = response.json()

            # Convert to our ModelsResponse format
            models_data = [
                ModelInfo(
                    id=model["id"],
                    created=model.get("created", int(time.time())),
                    owned_by=model.get("owned_by", "litellm"),
                )
                for model in data.get("data", [])
            ]

            return ModelsResponse(data=models_data)

    except httpx.HTTPStatusError as e:
        # Forward HTTP errors from LiteLLM
        logger.error(f"LiteLLM returned error status {e.response.status_code}: {e}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail={
                "error": {
                    "message": f"Failed to fetch models from LiteLLM: {e!s}",
                    "type": "upstream_error",
                },
            },
        )
    except httpx.RequestError as e:
        # Network/connection errors
        logger.error(f"Failed to connect to LiteLLM: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "message": f"LiteLLM service unavailable: {e!s}",
                    "type": "service_unavailable",
                },
            },
        )
    except Exception as e:
        # Unexpected errors
        logger.error(f"Unexpected error fetching models: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"Internal error while fetching models: {e!s}",
                    "type": "internal_server_error",
                },
            },
        )


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(  # noqa: PLR0915
    request: OpenAIChatRequest,
    tools: ToolsDependency,
    prompt_manager: PromptManagerDependency,
    conversation_db: ConversationDBDependency,
    http_request: Request,
    _: bool = Depends(verify_token),
) -> StreamingResponse:
    """
    OpenAI-compatible chat completions endpoint with intelligent request routing.

    ROUTING ARCHITECTURE:
    This endpoint implements routing to support three distinct execution paths:

    **A) Passthrough** (default): Direct OpenAI API call
      - No reasoning, no orchestration
      - Fast, low-latency responses
      - Default when no header provided (matches OpenAI experience)

    **B) Reasoning**: Single-loop reasoning agent
      - Accessible via `X-Routing-Mode: reasoning` header
      - Baseline for comparison
      - Manual selection only

    **C) Orchestration**: Multi-agent coordination via A2A protocol
      - Accessible via `X-Routing-Mode: orchestration` or auto-routing
      - Returns 501 stub until M3-M4 implementation

    TRACING ARCHITECTURE:
    This endpoint manages its own tracing instead of using middleware because:

    1. **Context Manager Incompatibility**: Standard `with tracer.start_as_current_span()`
       automatically ends spans when the context manager exits. For streaming responses,
       this happens immediately when StreamingResponse is returned, before the actual
       streaming completes.

    2. **Asynchronous Generator Consumption**: FastAPI's StreamingResponse consumes
       the generator asynchronously AFTER the endpoint function returns. Middleware-based
       tracing ends too early, causing "Setting attribute on ended span" warnings.

    3. **Manual Span Lifecycle Control**: We need the span to stay alive for the entire
       duration of streaming (potentially 10+ seconds), not just the function execution
       time (milliseconds). Only manual span management provides this control.

    4. **Output Attribute Timing**: The execution paths (passthrough/reasoning/orchestration)
       set output attributes on the parent span during streaming. The span must remain
       active to receive these attributes properly.

    Uses dependency injection for clean architecture and testability.
    Requires authentication via bearer token.
    """
    # Extract session ID from headers for tracing correlation
    session_id = http_request.headers.get("X-Session-ID")

    # CONVERSATION STORAGE: Extract conversation ID header for stateful/stateless mode detection
    conversation_header = http_request.headers.get("X-Conversation-ID")
    conversation_id: UUID | None = None
    is_new_conversation = False  # Track if this is a new conversation to avoid double-storing user messages
    messages_for_llm: list[dict[str, object]] = []

    # Check if conversation storage is available
    if conversation_header is not None and conversation_db is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "message": "Conversation storage is not available. The database service is not connected.",
                    "type": "service_unavailable",
                    "code": "conversation_storage_unavailable",
                },
            },
        )

    # Determine stateful vs stateless mode based on header presence
    if conversation_header is None:
        # STATELESS MODE: No header present, use messages as-is (current behavior)
        messages_for_llm = request.messages
        logger.debug("Stateless mode: No X-Conversation-ID header, using messages as-is")

    # STATEFUL MODE: Header present, manage conversation in database
    elif conversation_header == "" or conversation_header.lower() == "null":
        # NEW CONVERSATION: Create conversation in database
        system_msg = extract_system_message(request.messages)
        user_messages = [m for m in request.messages if m.get("role") != "system"]

        # Note: We'll create the conversation after routing decision to include routing_mode
        # For now, use messages as-is for routing decision
        messages_for_llm = (
            [{"role": "system", "content": system_msg}] if system_msg else []
        ) + user_messages

        logger.debug(f"New conversation mode: system_message={'present' if system_msg else 'absent'}")

    else:
        # CONTINUE EXISTING CONVERSATION: Load from database
        # Fail fast if system message present in continuation
        if any(m.get("role") == "system" for m in request.messages):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (
                            "System messages are not allowed when continuing a conversation. "
                            f"The system message for conversation {conversation_header} "
                            "was set during creation and cannot be changed."
                        ),
                        "type": "invalid_request_error",
                        "code": "system_message_in_continuation",
                    },
                },
            )

        # Parse and validate conversation UUID
        try:
            conversation_id = UUID(conversation_header)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": f"Invalid conversation ID format: {conversation_header}",
                        "type": "invalid_request_error",
                        "code": "invalid_conversation_id",
                    },
                },
            )

        # Load conversation from database
        try:
            conversation = await conversation_db.get_conversation(conversation_id)
        except Exception as e:
            logger.error(f"Failed to load conversation {conversation_id}: {e}")
            raise HTTPException(
                status_code=404,
                detail={
                    "error": {
                        "message": f"Conversation not found: {conversation_id}",
                        "type": "invalid_request_error",
                        "code": "conversation_not_found",
                    },
                },
            )

        # Build full message history: [system] + [history] + [new messages]
        history_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in conversation.messages
        ]
        messages_for_llm = (
            [{"role": "system", "content": conversation.system_message}, *history_messages, *request.messages]
        )

        logger.debug(
            f"Continuing conversation {conversation_id}: "
            f"{len(history_messages)} history + {len(request.messages)} new",
        )

    # Create span attributes
    span_attributes = {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
        "http.method": "POST",
        "http.url": str(http_request.url),
        "http.route": "/v1/chat/completions",
        "http.user_agent": http_request.headers.get("user-agent", ""),
    }

    # Add session ID to span if provided
    if session_id:
        span_attributes[SpanAttributes.SESSION_ID] = session_id

    # Add conversation context to span attributes
    if conversation_id:
        span_attributes["conversation.id"] = str(conversation_id)
        span_attributes["conversation.mode"] = "stateful"
    elif conversation_header is not None:
        # New conversation (header present but empty/null)
        span_attributes["conversation.mode"] = "stateful_new"
    else:
        span_attributes["conversation.mode"] = "stateless"

    # Use manual span management for streaming responses
    span = tracer.start_span("POST /v1/chat/completions", attributes=span_attributes)
    ctx = set_span_in_context(span)
    token = context.attach(ctx)

    # Set default success status - will only be overridden by errors
    span.set_attribute("http.status_code", 200)
    span.set_status(trace.Status(trace.StatusCode.OK))

    try:
        # ROUTING DECISION: Determine routing path (use messages_for_llm for accurate routing)
        # Create a modified request with conversation-aware messages
        routing_request = request.model_copy(update={"messages": messages_for_llm})
        routing_decision = await determine_routing(
            routing_request,
            headers=dict(http_request.headers),
        )

        # CONVERSATION CREATION: For new conversations, create in database after routing decision
        if conversation_header is not None and conversation_id is None:
            # This is a new conversation (header present but empty/null)
            system_msg = extract_system_message(request.messages)
            user_messages = [m for m in request.messages if m.get("role") != "system"]

            # Create conversation (system message only)
            conversation_id = await conversation_db.create_conversation(
                system_message=system_msg or "You are a helpful assistant.",
                title=None,
            )

            # Store user messages in new conversation
            if user_messages:
                await conversation_db.append_messages(conversation_id, user_messages)

            is_new_conversation = True  # User messages already stored

            logger.info(f"Created new conversation {conversation_id}")

            # Update span with conversation ID now that it's created
            span.set_attribute("conversation.id", str(conversation_id))

        # Log routing decision for observability
        logger.info(
            f"Routing decision: {routing_decision.routing_mode.value} "
            f"(source: {routing_decision.decision_source}, reason: {routing_decision.reason})",
        )

        # Add routing metadata to span
        span.set_attribute("routing.mode", routing_decision.routing_mode.value)
        span.set_attribute("routing.decision_source", routing_decision.decision_source)
        span.set_attribute("routing.reason", routing_decision.reason)

        # ROUTE A: PASSTHROUGH PATH - Direct OpenAI API call
        if routing_decision.routing_mode == RoutingMode.PASSTHROUGH:
            # Create request with conversation-aware messages for LLM call
            llm_request = request.model_copy(update={"messages": messages_for_llm})

            async def span_aware_stream():  # noqa: ANN202
                """
                Critical wrapper for streaming span lifecycle management with cancellation support.

                WHY THIS WRAPPER IS ESSENTIAL:

                1. **Span Lifecycle Synchronization**: Without this wrapper, the span would
                   end immediately when the endpoint function returns, but FastAPI's
                   StreamingResponse consumes the generator asynchronously afterward.

                2. **Generator Consumption Timing**: This wrapper ensures the span only
                   ends in the `finally` block AFTER the generator is fully consumed,
                   which happens when the client finishes reading all chunks.

                3. **Attribute Setting Window**: The execution paths set output attributes
                   on the parent span during generation. The span must stay alive until
                   after `parent_span.set_attribute(SpanAttributes.OUTPUT_VALUE, ...)`
                   is called, which happens during chunk generation.

                4. **Context Cleanup**: Properly detaches the OpenTelemetry context to
                   prevent memory leaks and context pollution.

                5. **Cancellation Support**: Checks for client disconnection before each yield
                   and raises CancelledError to stop the stream gracefully.

                Without this wrapper:
                - Span duration would be ~10ms (function execution time)
                - Output attributes would fail with "Setting attribute on ended span"
                - Tracing would not reflect actual streaming duration (~10+ seconds)

                With this wrapper:
                - Span duration matches actual streaming time
                - All attributes (input, metadata, output) appear on the same span
                - Clean context management and proper resource cleanup
                - Cancellation when client disconnects
                """
                assistant_content_buffer: list[str] = []

                try:
                    # Instantiate passthrough executor
                    executor = PassthroughExecutor()

                    # Use passthrough streaming with disconnection checking
                    async for chunk in executor.execute_stream(
                        llm_request,
                        parent_span=span,
                        check_disconnected=http_request.is_disconnected,
                    ):
                        # Buffer assistant content for storage (if stateful mode)
                        if conversation_id and chunk:
                            # Extract content from SSE chunk
                            # Chunk is a string like "data: {...}\n\n"
                            if isinstance(chunk, str) and "data: " in chunk:
                                import json
                                try:
                                    # Parse the JSON from SSE format
                                    data_line = chunk.strip().replace("data: ", "")
                                    if data_line and data_line != "[DONE]":
                                        chunk_data = json.loads(data_line)
                                        # Extract content delta
                                        if chunk_data.get("choices"):
                                            delta = chunk_data["choices"][0].get("delta", {})
                                            content = delta.get("content")
                                            if content:
                                                assistant_content_buffer.append(content)
                                except (json.JSONDecodeError, KeyError, IndexError):
                                    pass  # Skip malformed chunks

                        yield chunk

                    # STREAMING STORAGE: After streaming completes, store messages
                    if conversation_id and assistant_content_buffer:
                        full_content = "".join(assistant_content_buffer)
                        try:
                            # For NEW conversations: only store assistant response (user messages already stored)
                            # For CONTINUING conversations: store user messages + assistant response
                            if is_new_conversation:
                                # User messages already stored by create_conversation(), only store assistant
                                messages_to_store = [
                                    {"role": "assistant", "content": full_content},
                                ]
                            else:
                                # Continuing conversation: store user messages + assistant response
                                # Filter out system messages (already stored in conversation table)
                                user_messages = [m for m in request.messages if m.get("role") != "system"]
                                messages_to_store = [
                                    *user_messages,
                                    {"role": "assistant", "content": full_content},
                                ]
                            await conversation_db.append_messages(
                                conversation_id,
                                messages_to_store,
                            )
                            logger.debug(f"Stored {len(messages_to_store)} message(s) for conversation {conversation_id}")
                        except Exception as e:
                            logger.error(f"Failed to store messages for {conversation_id}: {e}", exc_info=True)

                            # Send storage failure indication in metadata chunk
                            import json
                            metadata_chunk = (
                                f"data: {json.dumps({'choices': [], 'metadata': {'storage_failed': True}})}\n\n"
                            )
                            yield metadata_chunk

                except asyncio.CancelledError:
                    # Cancellation is expected behavior - keep default OK status
                    span.set_attribute("http.cancelled", True)
                    span.set_attribute("cancellation.reason", "Client disconnected")
                    span_cleanup(span, token)
                    # Don't re-raise - let stream end gracefully
                    return
                except litellm.APIError as e:
                    # Forward LLM API errors
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    span_cleanup(span, token)
                    # Re-raise to be caught by StreamingResponse error handling
                    raise
                except Exception as e:
                    # Handle internal errors
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    span_cleanup(span, token)
                    # Re-raise to be caught by StreamingResponse error handling
                    raise
                finally:
                    # End span after streaming is complete (if not already ended)
                    span_cleanup(span, token)

            # Build response headers
            response_headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
            # Add conversation ID to response headers if stateful mode
            if conversation_id:
                response_headers["X-Conversation-ID"] = str(conversation_id)

            return StreamingResponse(
                span_aware_stream(),
                media_type="text/event-stream",
                headers=response_headers,
            )

        # ROUTE B: REASONING PATH - Single-loop reasoning agent
        if routing_decision.routing_mode == RoutingMode.REASONING:
            # Create request with conversation-aware messages for LLM call
            llm_request = request.model_copy(update={"messages": messages_for_llm})

            async def span_aware_reasoning_stream():  # noqa: ANN202
                """
                Wrapper for reasoning agent streaming with span lifecycle and
                disconnection support.

                Similar to passthrough streaming, but for reasoning agent path.
                """
                assistant_content_buffer: list[str] = []

                try:
                    # Instantiate reasoning agent with dependencies
                    reasoning_agent = ReasoningAgent(tools, prompt_manager)

                    async for chunk in reasoning_agent.execute_stream(
                        llm_request,
                        parent_span=span,
                        check_disconnected=http_request.is_disconnected,
                    ):
                        # Buffer assistant content for storage (if stateful mode)
                        if conversation_id and chunk:
                            if isinstance(chunk, str) and "data: " in chunk:
                                import json
                                try:
                                    data_line = chunk.strip().replace("data: ", "")
                                    if data_line and data_line != "[DONE]":
                                        chunk_data = json.loads(data_line)
                                        if chunk_data.get("choices"):
                                            delta = chunk_data["choices"][0].get("delta", {})
                                            content = delta.get("content")
                                            if content:
                                                assistant_content_buffer.append(content)
                                except (json.JSONDecodeError, KeyError, IndexError):
                                    pass

                        yield chunk

                    # STREAMING STORAGE: After streaming completes, store messages
                    if conversation_id and assistant_content_buffer:
                        full_content = "".join(assistant_content_buffer)
                        try:
                            # For NEW conversations: only store assistant response (user messages already stored)
                            # For CONTINUING conversations: store user messages + assistant response
                            if is_new_conversation:
                                # User messages already stored by create_conversation(), only store assistant
                                messages_to_store = [
                                    {"role": "assistant", "content": full_content},
                                ]
                            else:
                                # Continuing conversation: store user messages + assistant response
                                # Filter out system messages (already stored in conversation table)
                                user_messages = [m for m in request.messages if m.get("role") != "system"]
                                messages_to_store = [
                                    *user_messages,
                                    {"role": "assistant", "content": full_content},
                                ]
                            await conversation_db.append_messages(
                                conversation_id,
                                messages_to_store,
                            )
                            logger.debug(f"Stored {len(messages_to_store)} message(s) for conversation {conversation_id}")
                        except Exception as e:
                            logger.error(f"Failed to store assistant message for {conversation_id}: {e}", exc_info=True)

                            # Send storage failure indication
                            import json
                            metadata_chunk = (
                                f"data: {json.dumps({'choices': [], 'metadata': {'storage_failed': True}})}\n\n"
                            )
                            yield metadata_chunk

                except asyncio.CancelledError:
                    # Cancellation is expected behavior - keep default OK status
                    span.set_attribute("http.cancelled", True)
                    span.set_attribute("cancellation.reason", "Client disconnected")
                    span_cleanup(span, token)
                    # Don't re-raise - let stream end gracefully
                    return
                except litellm.APIError as e:
                    # Forward LLM API errors
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    span_cleanup(span, token)
                    # Re-raise to be caught by StreamingResponse error handling
                    raise
                except Exception as e:
                    # Handle internal errors
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    span_cleanup(span, token)
                    # Re-raise to be caught by StreamingResponse error handling
                    raise
                finally:
                    # End span after streaming is complete (if not already ended)
                    span_cleanup(span, token)

            # Build response headers
            response_headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
            # Add conversation ID to response headers if stateful mode
            if conversation_id:
                response_headers["X-Conversation-ID"] = str(conversation_id)

            return StreamingResponse(
                span_aware_reasoning_stream(),
                media_type="text/event-stream",
                headers=response_headers,
            )

        # ROUTE C: ORCHESTRATION PATH - Multi-agent coordination (501 stub until M3-M4)
        if routing_decision.routing_mode == RoutingMode.ORCHESTRATION:
            span_cleanup(span, token)
            raise HTTPException(
                status_code=501,
                detail={
                    "error": {
                        "message": (
                            "Multi-agent orchestration not yet implemented. "
                            "This feature will be available in Milestone 3-4. "
                            "For now, use X-Routing-Mode: passthrough or reasoning."
                        ),
                        "type": "not_implemented",
                        "code": "orchestration_not_ready",
                        "routing_decision": routing_decision.model_dump(mode='json'),
                    },
                },
            )

    except HTTPException:
        # Let FastAPI handle HTTPException naturally (e.g., 501 orchestration stub)
        # Don't wrap it in another error response
        raise
    except litellm.APIError as e:
        # Forward LLM API errors directly
        span.record_exception(e)
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
        span_cleanup(span, token)

        # LiteLLM APIError has status_code and message attributes
        raise HTTPException(
            status_code=e.status_code if hasattr(e, 'status_code') else 500,
            detail={"error": {"message": str(e), "type": "api_error"}},
        )
    except Exception as e:
        # Handle other internal errors
        span.record_exception(e)
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
        span_cleanup(span, token)

        error_response = ErrorResponse(
            error={
                "message": str(e),
                "type": "internal_server_error",
                "code": "500",
            },
        )
        raise HTTPException(
            status_code=500,
            detail=error_response.model_dump(),
        )


@app.get("/health")
async def health_check() -> dict[str, object]:
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/tools")
async def list_tools(
    tools: ToolsDependency,
    _: bool = Depends(verify_token),
) -> dict[str, list[str]]:
    """
    List available tools.

    Uses dependency injection to get available tools from MCP servers.
    Returns tool names grouped for compatibility.
    Requires authentication via bearer token.
    """
    try:
        if not tools:
            return {"tools": []}

        # Return list of tool names for compatibility
        tool_names = [tool.name for tool in tools]
        return {"tools": tool_names}

    except Exception as e:
        # Log error and re-raise for proper error response
        logger = logging.getLogger(__name__)
        logger.error(f"Error in list_tools: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
