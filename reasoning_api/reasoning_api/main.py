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
from typing import Annotated, Any
import httpx
import litellm
from fastapi import FastAPI, HTTPException, Depends, Request, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry import trace, context
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues
from opentelemetry.trace import set_span_in_context

# Note: OpenTelemetry context detach errors are filtered by logging_config.py
# This is a known OpenTelemetry issue with async generators and contextvars
# (see: https://github.com/open-telemetry/opentelemetry-python/issues/2606)
# The errors are harmless and filtered to reduce log noise.
from .openai_protocol import (
    OpenAIChatRequest,
    ModelsResponse,
    ModelInfo,
    generate_title_from_messages,
)
from .conversation_models import (
    ConversationListResponse,
    ConversationSummary,
    ConversationDetail,
    MessageResponse,
    UpdateConversationRequest,
    BranchConversationRequest,
    MessageSearchResponse,
    MessageSearchResultResponse,
)
from .dependencies import (
    service_container,
    ToolsDependency,
    PromptsDependency,
    PromptManagerDependency,
    ConversationDBDependency,
    ContextManagerDependency,
)
from .auth import verify_token
from .config import settings
from .tracing import setup_tracing
from .request_router import determine_routing, RoutingMode
from .logging_config import initialize_logging
from .executors.passthrough import PassthroughExecutor
from .executors.reasoning_agent import ReasoningAgent
from .conversation_utils import (
    parse_conversation_header,
    build_llm_messages,
    store_conversation_messages,
    ConversationMode,
    ConversationNotFoundError,
    InvalidConversationIDError,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:  # noqa: ARG001
    """
    Manage application lifespan events.

    NOTE: This runs once when the server starts and after it stops (yields control).
    """
    # Initialize logging and warning filters first (before any logs are generated)
    initialize_logging()

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
    expose_headers=["X-Conversation-ID"],  # Allow JavaScript to read conversation ID
)

# Get tracer and logger for request instrumentation
tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)


# Exception handlers for conversation errors
@app.exception_handler(ConversationNotFoundError)
async def conversation_not_found_handler(
        request: Request,  # noqa: ARG001
        exc: ConversationNotFoundError,
    ) -> JSONResponse:
    """Handle conversation not found error with 404 Not Found."""
    return JSONResponse(
        status_code=404,
        content={
            "error": {
                "message": str(exc),
                "type": "invalid_request_error",
                "code": "conversation_not_found",
            },
        },
    )


@app.exception_handler(InvalidConversationIDError)
async def invalid_conversation_id_handler(
        request: Request,  # noqa: ARG001
        exc: InvalidConversationIDError,
    ) -> JSONResponse:
    """Handle invalid conversation ID error with 400 Bad Request."""
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": str(exc),
                "type": "invalid_request_error",
                "code": "invalid_conversation_id",
            },
        },
    )




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


def create_executor_stream(
        executor: PassthroughExecutor | ReasoningAgent,
        llm_request: OpenAIChatRequest,
        request_messages: list[dict],
        conversation_id: UUID | None,
        conversation_db: ConversationDBDependency,
        span: trace.Span,
        token: object,
    ) -> AsyncGenerator[str]:
    """
    Create streaming generator for any executor.

    Handles streaming, storage, error handling, and span cleanup uniformly
    for all executor types (passthrough, reasoning, orchestration).

    Note: This is a generator factory (sync function returning async generator)
    to capture closure variables before FastAPI consumes the stream.
    """
    async def stream_with_lifecycle() -> AsyncGenerator[str]:
        """
        Wrapper managing span lifecycle and conversation storage.

        Critical for streaming: span must stay alive until generator is
        fully consumed by FastAPI's StreamingResponse (not just when
        endpoint returns). Otherwise output attributes fail with
        "Setting attribute on ended span" errors.
        """
        try:
            async for chunk in executor.execute_stream(llm_request):
                yield chunk

        except asyncio.CancelledError:
            span.set_attribute("http.cancelled", True)
            span.set_attribute("cancellation.reason", "Client disconnected")
            return
        except litellm.APIError as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise
        finally:
            # Save messages regardless of how streaming ended (success, cancel, error)
            if conversation_id:
                try:
                    await store_conversation_messages(
                        conversation_db=conversation_db,
                        conversation_id=conversation_id,
                        request_messages=request_messages,
                        response_content=executor.get_buffered_content(),
                        response_metadata=executor.get_metadata(),
                        reasoning_events=executor.get_reasoning_events(),
                    )
                    logger.debug(f"Stored conversation messages for {conversation_id}")
                except Exception as e:
                    logger.error(f"Failed to store messages for {conversation_id}: {e}", exc_info=True)  # noqa: E501

            # Cleanup span (always runs, even after return in except)
            span_cleanup(span, token)

    return stream_with_lifecycle()


@app.get("/v1/models")
async def list_models(
    _: bool = Depends(verify_token),
) -> ModelsResponse:
    """
    List available models from LiteLLM proxy.

    Proxies to LiteLLM's /v1/model/info endpoint to provide dynamic model discovery
    with detailed model capabilities and pricing information.

    Requires authentication via bearer token.
    """
    try:
        # Call LiteLLM proxy's /v1/model/info endpoint to get detailed model information
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.llm_base_url}/v1/model/info",
                headers={"Authorization": f"Bearer {settings.llm_api_key}"},
                timeout=5.0,
            )
            response.raise_for_status()

            # Parse LiteLLM response
            data = response.json()

            # Convert to our ModelsResponse format
            models_data = []
            for model in data.get("data", []):
                model_info = model.get("model_info", {})
                models_data.append(
                    ModelInfo(
                        id=model["model_name"],
                        created=int(time.time()),
                        owned_by=model_info.get("litellm_provider", "unknown"),
                        max_input_tokens=model_info.get("max_input_tokens"),
                        max_output_tokens=model_info.get("max_output_tokens"),
                        input_cost_per_token=model_info.get("input_cost_per_token"),
                        output_cost_per_token=model_info.get("output_cost_per_token"),
                        supports_reasoning=model_info.get("supports_reasoning"),
                        supports_response_schema=model_info.get("supports_response_schema"),
                        supports_vision=model_info.get("supports_vision"),
                        supports_function_calling=model_info.get("supports_function_calling"),
                        supports_web_search=model_info.get("supports_web_search"),
                    ),
                )

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
async def chat_completions(  # noqa: PLR0915, PLR0912
    request: OpenAIChatRequest,
    tools: ToolsDependency,
    prompt_manager: PromptManagerDependency,
    conversation_db: ConversationDBDependency,
    context_manager: ContextManagerDependency,
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
    session_id = http_request.headers.get("X-Session-ID")
    conversation_ctx = parse_conversation_header(http_request.headers.get("X-Conversation-ID"))

    if conversation_ctx.mode != ConversationMode.STATELESS and conversation_db is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "message": "Conversation storage is not available. The database service is not connected.",  # noqa: E501
                    "type": "service_unavailable",
                    "code": "conversation_storage_unavailable",
                },
            },
        )

    messages_for_llm = await build_llm_messages(
        request.messages,
        conversation_ctx,
        conversation_db,
    )
    conversation_id = conversation_ctx.conversation_id

    logger.debug(
        f"Conversation mode: {conversation_ctx.mode.value}, "
        f"messages: {len(messages_for_llm)}, "
        f"conversation_id: {conversation_id or 'none'}",
    )

    # Validate request: continuing conversations can have empty messages (for regeneration)
    # but new/stateless conversations must have at least one message
    if len(request.messages) == 0:
        if conversation_ctx.mode in (ConversationMode.NEW, ConversationMode.STATELESS):
            raise HTTPException(
                status_code=422,
                detail={
                    "error": {
                        "message": (
                            "Messages array cannot be empty for new or stateless conversations. "
                            "Empty messages are only allowed when continuing an existing "
                            "conversation (for regeneration)."
                        ),
                        "type": "invalid_request_error",
                        "code": "empty_messages_array",
                    },
                },
            )
        # For CONTINUING mode with empty messages: this is valid (regeneration)
        logger.info(
            f"Regeneration request for conversation {conversation_id}: "
            f"empty messages array, using conversation history only",
        )

    span_attributes = {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
        "http.method": "POST",
        "http.url": str(http_request.url),
        "http.route": "/v1/chat/completions",
        "http.user_agent": http_request.headers.get("user-agent", ""),
        "conversation.mode": conversation_ctx.mode.value,
    }
    if session_id:
        span_attributes[SpanAttributes.SESSION_ID] = session_id
    if conversation_id:
        span_attributes["conversation.id"] = str(conversation_id)

    span = tracer.start_span("POST /v1/chat/completions", attributes=span_attributes)
    ctx = set_span_in_context(span)
    token = context.attach(ctx)
    span.set_attribute("http.status_code", 200)
    span.set_status(trace.Status(trace.StatusCode.OK))

    try:
        routing_request = request.model_copy(update={"messages": messages_for_llm})
        routing_decision = await determine_routing(
            routing_request,
            headers=dict(http_request.headers),
        )
        if conversation_ctx.mode == ConversationMode.NEW:
            title = generate_title_from_messages(request.messages)
            conversation_id = await conversation_db.create_conversation(
                title=title,
            )
            logger.info(f"Created new conversation {conversation_id}")
            span.set_attribute("conversation.id", str(conversation_id))

        logger.info(
            f"Routing decision: {routing_decision.routing_mode.value} "
            f"(source: {routing_decision.decision_source}, reason: {routing_decision.reason})",
        )
        span.set_attribute("routing.mode", routing_decision.routing_mode.value)
        span.set_attribute("routing.decision_source", routing_decision.decision_source)
        span.set_attribute("routing.reason", routing_decision.reason)

        llm_request = request.model_copy(update={"messages": messages_for_llm})

        # Create executor based on routing decision
        if routing_decision.routing_mode == RoutingMode.PASSTHROUGH:
            executor = PassthroughExecutor(
                context_manager=context_manager,
                parent_span=span,
                check_disconnected=http_request.is_disconnected,
            )
        elif routing_decision.routing_mode == RoutingMode.REASONING:
            executor = ReasoningAgent(
                tools=tools,
                prompt_manager=prompt_manager,
                context_manager=context_manager,
                parent_span=span,
                check_disconnected=http_request.is_disconnected,
            )
        elif routing_decision.routing_mode == RoutingMode.ORCHESTRATION:
            span_cleanup(span, token)
            raise HTTPException(
                status_code=501,
                detail={
                    "error": {
                        "message": (
                            "Multi-agent orchestration not yet implemented. "
                            "This feature will be available in the future. "
                            "For now, use X-Routing-Mode: passthrough or reasoning."
                        ),
                        "type": "not_implemented",
                        "code": "orchestration_not_ready",
                        "routing_decision": routing_decision.model_dump(mode='json'),
                    },
                },
            )

        # Create streaming response (common for all non-orchestration routes)
        stream = create_executor_stream(
            executor=executor,
            llm_request=llm_request,
            request_messages=request.messages,
            conversation_id=conversation_id,
            conversation_db=conversation_db,
            span=span,
            token=token,
        )
        response_headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}
        if conversation_id:
            response_headers["X-Conversation-ID"] = str(conversation_id)
        return StreamingResponse(stream, media_type="text/event-stream", headers=response_headers)

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

        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": str(e),
                    "type": "internal_server_error",
                    "code": "500",
                },
            },
        )


@app.get("/v1/conversations", response_model=ConversationListResponse)
async def list_conversations(
    conversation_db: ConversationDBDependency,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    archive_filter: str = Query("active", pattern="^(active|archived|all)$", description="Filter by archive status"),  # noqa: E501
    _: bool = Depends(verify_token),
) -> ConversationListResponse:
    """
    List conversations with pagination.

    Returns conversations ordered by most recently updated (updated_at DESC).

    Query parameters:
    - **limit**: Results per page (1-100, default 50)
    - **offset**: Skip N results for pagination (default 0)
    - **archive_filter**: Filter by archive status:
        - "active" (default) - Only non-archived conversations
        - "archived" - Only archived conversations
        - "all" - All conversations

    Args:
        conversation_db: Database dependency for conversation operations
        limit: Maximum number of conversations to return (1-100, default: 50)
        offset: Number of conversations to skip (default: 0)
        archive_filter: Filter by archive status (default: "active")

    Returns:
        Paginated list of conversation summaries with total count

    Requires authentication via bearer token.
    """
    if conversation_db is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "message": "Conversation storage is not available",
                    "type": "service_unavailable",
                    "code": "conversation_storage_unavailable",
                },
            },
        )

    try:
        conversations, total = await conversation_db.list_conversations(
            limit=limit,
            offset=offset,
            archive_filter=archive_filter,
        )
    except ValueError as e:
        # Invalid archive_filter or other validation error
        raise HTTPException(
            status_code=422,
            detail={
                "error": {
                    "message": str(e),
                    "type": "invalid_request_error",
                    "code": "invalid_parameter",
                },
            },
        ) from e

    # Convert to response models
    summaries = [
        ConversationSummary(
            id=conv.id,
            title=conv.title,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            archived_at=conv.archived_at,
            message_count=conv.message_count or 0,
        )
        for conv in conversations
    ]

    return ConversationListResponse(
        conversations=summaries,
        total=total,
        limit=limit,
        offset=offset,
    )


@app.get("/v1/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: UUID,
    conversation_db: ConversationDBDependency,
    _: bool = Depends(verify_token),
) -> ConversationDetail:
    """
    Get a single conversation with all messages.

    Returns full conversation history ordered by sequence number.

    Args:
        conversation_id: UUID of the conversation to retrieve
        conversation_db: Database dependency for conversation operations

    Returns:
        Full conversation with all messages

    Raises:
        404: Conversation not found

    Requires authentication via bearer token.
    """
    if conversation_db is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "message": "Conversation storage is not available",
                    "type": "service_unavailable",
                    "code": "conversation_storage_unavailable",
                },
            },
        )

    try:
        conv = await conversation_db.get_conversation(conversation_id)
    except ValueError:
        # ConversationDB.get_conversation raises ValueError if not found
        raise ConversationNotFoundError(str(conversation_id))

    # Convert to response model
    messages = [
        MessageResponse(
            id=msg.id,
            conversation_id=msg.conversation_id,
            role=msg.role,
            content=msg.content,
            reasoning_events=msg.reasoning_events,
            metadata=msg.metadata,
            total_cost=msg.total_cost,
            created_at=msg.created_at,
            sequence_number=msg.sequence_number,
        )
        for msg in conv.messages
    ]
    return ConversationDetail(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        messages=messages,
    )


@app.delete("/v1/conversations/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: UUID,
    conversation_db: ConversationDBDependency,
    permanent: bool = False,
    _: bool = Depends(verify_token),
) -> None:
    """
    Delete a conversation (soft or hard delete).

    Soft delete (default): Sets archived_at timestamp. Archived conversations are
    excluded from list_conversations results but can still be retrieved by ID.

    Hard delete (permanent=true): Permanently removes the conversation and all
    associated messages from the database. This operation cannot be undone.

    Args:
        conversation_id: UUID of the conversation to delete
        conversation_db: Database dependency for conversation operations
        permanent: If true, permanently delete; if false, soft delete (default)

    Returns:
        204 No Content on success

    Raises:
        404: Conversation not found

    Requires authentication via bearer token.

    Examples:
        DELETE /v1/conversations/{id}              → Soft delete (default)
        DELETE /v1/conversations/{id}?permanent=true  → Hard delete
    """
    if conversation_db is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "message": "Conversation storage is not available",
                    "type": "service_unavailable",
                    "code": "conversation_storage_unavailable",
                },
            },
        )

    success = await conversation_db.delete_conversation(conversation_id, permanent)

    if not success:
        # Conversation not found (or already archived for soft delete)
        raise ConversationNotFoundError(str(conversation_id))


@app.delete("/v1/conversations/{conversation_id}/messages/{sequence_number}", status_code=204)
async def delete_messages(
    conversation_id: UUID,
    sequence_number: int,
    conversation_db: ConversationDBDependency,
    _: bool = Depends(verify_token),
) -> None:
    """
    Delete a message and all subsequent messages.

    Deletes the message at the specified sequence number and all messages that
    come after it in the conversation. This is useful for:
    - Removing unwanted exchanges
    - Truncating conversation before regeneration
    - Cleaning up conversation history

    Args:
        conversation_id: UUID of the conversation
        sequence_number: Sequence number of the message to delete (and all after)
        conversation_db: Database dependency for conversation operations

    Returns:
        204 No Content on success

    Raises:
        404: Conversation or message not found
        422: Invalid sequence number (negative)
        503: Database unavailable

    Requires authentication via bearer token.

    Examples:
        DELETE /v1/conversations/{id}/messages/5  → Deletes message 5 and all after
    """
    if conversation_db is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "message": "Conversation storage is not available",
                    "type": "service_unavailable",
                    "code": "conversation_storage_unavailable",
                },
            },
        )

    # Validate sequence_number is non-negative
    if sequence_number < 0:
        raise HTTPException(
            status_code=422,
            detail={
                "error": {
                    "message": f"Invalid sequence number: {sequence_number}. Must be >= 0.",
                    "type": "invalid_request_error",
                    "code": "invalid_sequence_number",
                },
            },
        )

    success = await conversation_db.delete_messages_from_sequence(
        conversation_id=conversation_id,
        from_sequence=sequence_number,
    )

    if not success:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": (
                        f"Message with sequence number {sequence_number} "
                        f"not found in conversation {conversation_id}"
                    ),
                    "type": "not_found_error",
                    "code": "message_not_found",
                },
            },
        )


@app.patch("/v1/conversations/{conversation_id}", response_model=ConversationSummary)
async def update_conversation(
    conversation_id: UUID,
    request: UpdateConversationRequest,
    conversation_db: ConversationDBDependency,
    _: bool = Depends(verify_token),
) -> ConversationSummary:
    """
    Update conversation title.

    Allows updating or clearing the conversation title. Empty strings and null values
    both clear the title. Titles are trimmed and limited to 200 characters.

    Args:
        conversation_id: UUID of the conversation to update
        request: Request body with title field
        conversation_db: Database dependency for conversation operations

    Returns:
        Updated conversation summary

    Raises:
        404: Conversation not found or archived
        422: Invalid title (exceeds 200 characters)

    Requires authentication via bearer token.
    """
    if conversation_db is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "message": "Conversation storage is not available",
                    "type": "service_unavailable",
                    "code": "conversation_storage_unavailable",
                },
            },
        )

    # Update conversation title
    success = await conversation_db.update_conversation_title(
        conversation_id=conversation_id,
        title=request.title,
    )

    if not success:
        # Conversation not found or archived
        raise ConversationNotFoundError(str(conversation_id))

    # Fetch updated conversation to return in response
    try:
        conv = await conversation_db.get_conversation(conversation_id)
    except ValueError:
        # Should not happen since we just updated it, but handle gracefully
        raise ConversationNotFoundError(str(conversation_id))

    return ConversationSummary(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        archived_at=conv.archived_at,
        message_count=conv.message_count or 0,
    )


@app.post("/v1/conversations/{conversation_id}/branch", response_model=ConversationDetail)
async def branch_conversation(
    conversation_id: UUID,
    request: BranchConversationRequest,
    conversation_db: ConversationDBDependency,
    _: bool = Depends(verify_token),
) -> ConversationDetail:
    """
    Branch a conversation at a specific message.

    Creates a new conversation by copying the source conversation up to and
    including the specified message sequence number. The new conversation:
    - Has a title like "Branch of {original_title}"
    - Contains all messages up to and including branch_at_sequence
    - Preserves user_id and metadata from source conversation
    - Is a completely independent conversation (changes don't affect source)

    This is useful for exploring different conversation paths or trying
    alternative responses without losing the original conversation history.

    Args:
        conversation_id: UUID of the source conversation to branch from
        request: Request body with branch_at_sequence parameter
        conversation_db: Database dependency for conversation operations

    Returns:
        Full ConversationDetail of the newly created branched conversation

    Raises:
        404: Source conversation or sequence number not found
        422: Invalid request parameters
        503: Database unavailable

    Requires authentication via bearer token.

    Example:
        POST /v1/conversations/{id}/branch
        Body: {"branch_at_sequence": 5}
        → Creates new conversation with messages 0-5 from source
    """
    if conversation_db is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "message": "Conversation storage is not available",
                    "type": "service_unavailable",
                    "code": "conversation_storage_unavailable",
                },
            },
        )

    try:
        # Branch conversation (creates new conversation with copied messages)
        branched_conv = await conversation_db.branch_conversation(
            source_conversation_id=conversation_id,
            branch_at_sequence=request.branch_at_sequence,
        )
    except ValueError as e:
        # Source conversation not found or sequence number doesn't exist
        error_msg = str(e)
        # Check for sequence number first (more specific)
        if "Sequence number" in error_msg:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": {
                        "message": error_msg,
                        "type": "not_found_error",
                        "code": "sequence_not_found",
                    },
                },
            )
        if "not found" in error_msg or "archived" in error_msg:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": {
                        "message": error_msg,
                        "type": "not_found_error",
                        "code": "conversation_not_found",
                    },
                },
            )
        # Other validation error
        raise HTTPException(
            status_code=422,
            detail={
                "error": {
                    "message": error_msg,
                    "type": "invalid_request_error",
                    "code": "invalid_parameter",
                },
            },
        )

    # Convert to response model (same as get_conversation endpoint)
    messages = [
        MessageResponse(
            id=msg.id,
            conversation_id=msg.conversation_id,
            role=msg.role,
            content=msg.content,
            reasoning_events=msg.reasoning_events,
            metadata=msg.metadata,
            total_cost=msg.total_cost,
            created_at=msg.created_at,
            sequence_number=msg.sequence_number,
        )
        for msg in branched_conv.messages
    ]
    return ConversationDetail(
        id=branched_conv.id,
        title=branched_conv.title,
        created_at=branched_conv.created_at,
        updated_at=branched_conv.updated_at,
        messages=messages,
    )


@app.get("/v1/messages/search", response_model=MessageSearchResponse)
async def search_messages(
    conversation_db: ConversationDBDependency,
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(50, ge=1, le=100, description="Results per page"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    archive_filter: str = Query("active", pattern="^(active|archived|all)$", description="Filter by archive status"),  # noqa: E501
    _: bool = Depends(verify_token),
) -> MessageSearchResponse:
    """
    Search messages across all conversations using full-text search.

    Searches message content using PostgreSQL full-text search with relevance ranking.
    Results are ordered by relevance score (best matches first) and then by recency.

    Query parameters:
    - **q**: Search phrase (required, case-insensitive, supports multi-word queries)
    - **limit**: Results per page (1-100, default 50)
    - **offset**: Skip N results for pagination (default 0)
    - **archive_filter**: Filter by archive status:
        - "active" (default) - Only search non-archived conversations
        - "archived" - Only search archived conversations
        - "all" - Search all conversations

    Returns:
    - **results**: List of matching messages with conversation context
    - **total**: Total number of matching messages (for pagination)
    - **limit**: Results per page (echoed from request)
    - **offset**: Skip count (echoed from request)
    - **query**: Search query (echoed from request)

    Each result includes:
    - Message content and metadata (role, created_at)
    - Conversation context (conversation_id, title, archived status)
    - Relevance score (higher = better match)
    - Highlighted snippet (shows matching text in context)

    Raises:
        503: Database unavailable
        422: Invalid parameters (e.g., invalid archive_filter)

    Requires authentication via bearer token.
    """
    if conversation_db is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "message": "Conversation storage is not available",
                    "type": "service_unavailable",
                    "code": "conversation_storage_unavailable",
                },
            },
        )

    try:
        # Execute search
        results, total = await conversation_db.search_messages(
            search_phrase=q,
            limit=limit,
            offset=offset,
            archive_filter=archive_filter,
        )

        # Convert database results to API response models
        result_responses = [
            MessageSearchResultResponse(
                message_id=r.message_id,
                conversation_id=r.conversation_id,
                conversation_title=r.conversation_title,
                role=r.role,
                content=r.content,
                snippet=r.snippet,
                relevance=r.relevance,
                created_at=r.created_at,
                archived=r.archived,
            )
            for r in results
        ]

        return MessageSearchResponse(
            results=result_responses,
            total=total,
            limit=limit,
            offset=offset,
            query=q,
        )

    except ValueError as e:
        # Invalid archive_filter or other validation error
        raise HTTPException(
            status_code=422,
            detail={
                "error": {
                    "message": str(e),
                    "type": "invalid_request_error",
                    "code": "invalid_parameter",
                },
            },
        ) from e


@app.get("/health")
async def health_check() -> dict[str, object]:
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/v1/mcp/tools")
async def list_mcp_tools(
    tools: ToolsDependency,
    _: bool = Depends(verify_token),
) -> dict[str, list[dict[str, object]]]:
    """
    List available MCP tools with metadata.

    Returns tool names, descriptions, and input schemas for discovery.
    Clients can use this to understand what tools are available.
    Requires authentication via bearer token.

    Returns:
        {"tools": [{"name": "...", "description": "...", "input_schema": {...}}, ...]}
    """
    try:
        return {
            "tools": [tool.to_dict() for tool in tools],
        }
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error listing MCP tools: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": str(e), "type": "internal_error"}},
        ) from e


@app.get("/v1/mcp/prompts")
async def list_mcp_prompts(
    prompts: PromptsDependency,
    _: bool = Depends(verify_token),
) -> dict[str, list[dict[str, object]]]:
    """
    List available MCP prompts with metadata.

    Returns prompt names, descriptions, and argument schemas for discovery.
    Clients can use this to understand what prompts are available.
    Requires authentication via bearer token.

    Returns:
        {"prompts": [{"name": "...", "description": "...", "arguments": [...]}, ...]}
    """
    try:
        return {
            "prompts": [prompt.to_dict() for prompt in prompts],
        }
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error listing MCP prompts: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": str(e), "type": "internal_error"}},
        ) from e


@app.post("/v1/mcp/prompts/{prompt_name}")
async def execute_mcp_prompt(
    prompt_name: str,
    arguments: dict[str, Any],
    prompts: PromptsDependency,
    _: bool = Depends(verify_token),
) -> dict[str, object]:
    """
    Execute an MCP prompt with the provided arguments.

    Calls the MCP server's get_prompt operation with the specified arguments
    and returns the rendered prompt messages ready for LLM consumption.
    Requires authentication via bearer token.

    Args:
        prompt_name: Name of the prompt to execute (e.g., "server__prompt_name")
        arguments: Dictionary of argument name/value pairs for the prompt
        prompts: Available prompts (injected dependency)

    Returns:
        {
            "description": "...",
            "messages": [{"role": "user", "content": "..."}, ...]
        }

    Raises:
        404: Prompt not found
        400: Invalid arguments
        500: Prompt execution failed
    """
    try:
        # Find prompt by name
        prompt = next((p for p in prompts if p.name == prompt_name), None)

        if prompt is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": {
                        "message": f"Prompt '{prompt_name}' not found",
                        "type": "not_found_error",
                    },
                },
            )

        # Execute the prompt with arguments
        result = await prompt(**arguments)

        if not result.success:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "message": result.error or "Prompt execution failed",
                        "type": "prompt_execution_error",
                    },
                },
            )

        return {
            "description": prompt.description,
            "messages": result.messages,
        }
    except HTTPException:
        raise
    except ValueError as e:
        # Validation errors (missing required arguments, unexpected arguments)
        logger = logging.getLogger(__name__)
        logger.warning(f"Invalid arguments for prompt '{prompt_name}': {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": str(e),
                    "type": "invalid_request_error",
                },
            },
        ) from e
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error executing MCP prompt '{prompt_name}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": str(e), "type": "internal_error"}},
        ) from e


@app.post("/v1/mcp/tools/{tool_name}")
async def execute_mcp_tool(
    tool_name: str,
    arguments: dict[str, Any],
    tools: ToolsDependency,
    _: bool = Depends(verify_token),
) -> dict[str, object]:
    """
    Execute an MCP tool with the provided arguments.

    Allows direct execution of individual MCP tools for testing, debugging,
    or client-side workflow orchestration. The tool is executed with the
    provided arguments and returns the result along with execution metadata.
    Requires authentication via bearer token.

    Args:
        tool_name: Name of the tool to execute (e.g., "server__tool_name")
        arguments: Dictionary of argument name/value pairs for the tool
        tools: Available tools (injected dependency)

    Returns:
        {
            "tool_name": "...",
            "success": true,
            "result": {...},
            "execution_time_ms": 123.45
        }

    Raises:
        404: Tool not found
        400: Invalid arguments
        500: Tool execution failed
    """
    try:
        # Find tool by name
        tool = next((t for t in tools if t.name == tool_name), None)

        if tool is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": {
                        "message": f"Tool '{tool_name}' not found",
                        "type": "not_found_error",
                    },
                },
            )

        # Execute the tool with arguments
        result = await tool(**arguments)

        if not result.success:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "message": result.error or "Tool execution failed",
                        "type": "tool_execution_error",
                    },
                },
            )

        return {
            "tool_name": result.tool_name,
            "success": result.success,
            "result": result.result,
            "execution_time_ms": result.execution_time_ms,
        }
    except HTTPException:
        raise
    except ValueError as e:
        # Validation errors (missing required arguments, unexpected arguments)
        logger.warning(f"Invalid arguments for tool '{tool_name}': {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": str(e),
                    "type": "invalid_request_error",
                },
            },
        ) from e
    except Exception as e:
        logger.error(f"Error executing MCP tool '{tool_name}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": str(e), "type": "internal_error"}},
        ) from e


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
