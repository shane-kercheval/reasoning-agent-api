"""
Integration test configuration with testcontainers.

NO MANUAL SETUP REQUIRED - testcontainers automatically:
1. Spins up PostgreSQL container when tests start
2. Runs Alembic migrations to create schema
3. Provides isolated database per test (transaction rollback)
4. Tears down container when tests finish

Just run: pytest tests/integration_tests/ -v
"""

import os
import json
import pytest
import pytest_asyncio
from collections.abc import AsyncGenerator
from testcontainers.postgres import PostgresContainer
from litellm import ModelResponse
from litellm.types.utils import StreamingChoices, Delta
from api.database import ConversationDB
from api.dependencies import service_container
from api.config import settings


@pytest.fixture(scope="session")
def postgres_container():
    """
    Start PostgreSQL container for entire test session.

    Session-scoped means:
    - Starts once when first integration test runs
    - Shared by all integration tests (fast!)
    - Tears down after all integration tests complete
    """
    with PostgresContainer(
        image="postgres:16",
        username="test_user",
        password="test_pass",
        dbname="test_reasoning",
        driver="asyncpg",  # Use asyncpg driver for async operations
    ) as postgres:
        # Get connection URL from container (testcontainers provides this)
        db_url = postgres.get_connection_url()
        yield db_url


@pytest.fixture(scope="session")
def postgres_engine(postgres_container: str) -> None:
    """
    Run Alembic migrations to create schema in test database.

    This is a session-scoped fixture that runs once for all tests.
    """
    import subprocess

    try:
        # Override REASONING_DATABASE_URL for migrations
        env = os.environ.copy()
        env['REASONING_DATABASE_URL'] = postgres_container  # Keep asyncpg URL!

        # Run migrations with environment override
        result = subprocess.run(
            ["uv", "run", "alembic", "upgrade", "head"],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"✓ Migrations applied successfully:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Migration failed:\nstdout: {e.stdout}\nstderr: {e.stderr}")
        raise


@pytest_asyncio.fixture(loop_scope="function")
async def conversation_db(postgres_container: str, postgres_engine):
    """
    Provide ConversationDB instance with transaction rollback isolation.

    Each test gets:
    - A clean database state
    - Real PostgreSQL behavior (catches real issues!)
    - Automatic cleanup via transaction rollback

    NO MANUAL CLEANUP CODE NEEDED - rollback handles everything!

    Note: loop_scope="function" ensures this fixture runs in the test's event loop,
    avoiding asyncpg's "Future attached to a different loop" errors.
    """
    import asyncpg

    # Ensure migrations have run (postgres_engine fixture runs them)
    # Convert SQLAlchemy URL to asyncpg format
    db_url = postgres_container.replace("postgresql+asyncpg://", "postgresql://")

    # Create asyncpg connection in the CURRENT event loop (the test's loop)
    conn = await asyncpg.connect(db_url)

    # Set up JSONB codec on this connection
    await conn.set_type_codec(
        "jsonb",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )

    # Begin a transaction (this will NEVER be committed)
    trans = conn.transaction()
    await trans.start()

    try:
        # Create ConversationDB with test connection (bypasses pool)
        db = ConversationDB(
            postgres_container,
            _test_connection=conn,
        )

        yield db

    finally:
        # Always rollback and close, even if test fails
        try:
            await trans.rollback()
        except Exception:
            pass  # Transaction might already be in error state

        try:
            await conn.close()
        except Exception:
            pass  # Connection might already be closed


@pytest_asyncio.fixture(scope="session", autouse=True)
async def initialize_service_container():
    """
    Initialize service container for integration tests.

    Note: ASGI transport doesn't trigger FastAPI lifespan events, so we
    manually initialize the service container here.

    IMPORTANT: Override LITELLM_API_KEY to use TEST key for all integration tests.
    This ensures test usage is tracked separately from dev/prod usage.
    """
    # Override settings.llm_api_key to use test key for stateful tests
    # Note: Stateless tests mock LiteLLM and don't need this
    if "LITELLM_TEST_KEY" in os.environ:
        settings.llm_api_key = os.environ["LITELLM_TEST_KEY"]
    else:
        # For stateless tests with mocked LiteLLM, use a dummy key
        settings.llm_api_key = "sk-test-dummy-key-for-mocked-tests"

    await service_container.initialize()

    yield

    # Cleanup - ignore event loop closed errors (common pytest-asyncio issue)
    try:
        await service_container.cleanup()
    except RuntimeError as e:
        if "Event loop is closed" not in str(e):
            raise


# =============================================================================
# Shared Mock Helpers for LiteLLM
# =============================================================================


def create_mock_litellm_response(content: str) -> ModelResponse:
    """
    Create a mock non-streaming LiteLLM response using actual litellm types.

    Use for JSON mode reasoning calls that expect a single response object.
    """
    from litellm.types.utils import Message, Choices, Usage

    return ModelResponse(
        id="test-id",
        choices=[Choices(
            index=0,
            message=Message(
                role="assistant",
                content=content,
            ),
            finish_reason="stop",
        )],
        created=1234567890,
        model="gpt-4o-mini",
        object="chat.completion",
        usage=Usage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        ),
    )


async def create_mock_litellm_stream(content: str) -> AsyncGenerator[ModelResponse]:
    """
    Create a mock LiteLLM streaming response using actual litellm types.

    Use for streaming calls that expect an async generator of chunks.
    """
    # First chunk - role
    yield ModelResponse(
        id="test-id",
        choices=[StreamingChoices(
            index=0,
            delta=Delta(role="assistant"),
            finish_reason=None,
        )],
        created=1234567890,
        model="gpt-4o-mini",
        object="chat.completion.chunk",
    )

    # Second chunk - content
    yield ModelResponse(
        id="test-id",
        choices=[StreamingChoices(
            index=0,
            delta=Delta(content=content),
            finish_reason=None,
        )],
        created=1234567890,
        model="gpt-4o-mini",
        object="chat.completion.chunk",
    )

    # Final chunk - finish
    yield ModelResponse(
        id="test-id",
        choices=[StreamingChoices(
            index=0,
            delta=Delta(),
            finish_reason="stop",
        )],
        created=1234567890,
        model="gpt-4o-mini",
        object="chat.completion.chunk",
    )


def create_smart_litellm_mock(
    streaming_content: str | None = None,
    reasoning_content: str | None = None,
):
    """
    Create a smart mock that returns the right type based on whether stream=True.

    This handles both streaming and non-streaming litellm calls and intelligently
    decides whether to use tools based on the user's message content.

    Args:
        streaming_content: Optional override for streaming responses
        reasoning_content: Optional override for JSON reasoning step calls
    """
    # Track what tools were used so we can generate appropriate streaming content
    last_tools_used = []

    def smart_mock(*args, **kwargs):
        nonlocal last_tools_used

        # Check if this is a streaming call
        if kwargs.get('stream', False):
            # Generate context-aware streaming content based on tools used
            if streaming_content is not None:
                content = streaming_content
            elif last_tools_used:
                # Generate content based on tools that were executed
                tool_name = last_tools_used[0].get('tool_name', '')
                arguments = last_tools_used[0].get('arguments', {})

                if tool_name in {'get_weather', 'weather_api'}:
                    location = arguments.get('location', 'Tokyo')
                    if tool_name == 'weather_api':
                        # MCP tool returns complete weather data including wind speed
                        content = f"Based on the tool results, the weather in {location} is 25°C and Partly cloudy with 65% humidity and wind speed of 10 km/h."
                    else:
                        # Fake tool format
                        content = f"Based on the tool results, the weather in {location} is 22°C and Sunny with 60% humidity."
                elif tool_name in {'search_web', 'search_database'}:
                    query = arguments.get('query', 'the topic')
                    content = f"Based on the search results for '{query}', I found several relevant resources with tutorials and documentation."
                elif tool_name == 'analyze_sentiment':
                    text = arguments.get('text', 'the text')
                    content = f"Based on sentiment analysis of '{text}', the sentiment is positive with high confidence."
                elif tool_name == 'calculator':
                    operation = arguments.get('operation', 'add')
                    a = arguments.get('a', 0)
                    b = arguments.get('b', 0)
                    if operation == 'add':
                        result = a + b
                    elif operation == 'subtract':
                        result = a - b
                    elif operation == 'multiply':
                        result = a * b
                    else:
                        result = "calculated"
                    content = f"Based on the calculation, {a} {operation} {b} equals {result}."
                elif tool_name == 'get_system_info':
                    content = "Based on the system info results, the platform is test_platform, version 1.0.0, and status is healthy."
                elif tool_name == 'process_text':
                    text = arguments.get('text', 'test text')
                    mode = arguments.get('mode', 'uppercase')
                    if mode == 'uppercase':
                        processed = text.upper()
                    elif mode == 'lowercase':
                        processed = text.lower()
                    elif mode == 'reverse':
                        processed = text[::-1]
                    else:
                        processed = text
                    content = f"Based on the text processing, '{text}' in {mode} mode is '{processed}'."
                else:
                    content = "Based on the tool results, here is the information you requested."
            else:
                content = "Here is the response to your question."

            return create_mock_litellm_stream(content)
        # Non-streaming call (likely JSON mode for reasoning)
        if kwargs.get('response_format', {}).get('type') == 'json_object':
            # Intelligent tool detection for reasoning steps
            if reasoning_content is not None:
                # Use provided reasoning content
                return create_mock_litellm_response(reasoning_content)
            # Auto-detect if tools should be used based on message content
            messages = kwargs.get('messages', [])
            user_message = ""
            for msg in messages:
                if isinstance(msg, dict) and msg.get('role') == 'user':
                    user_message = str(msg.get('content', '')).lower()
                    break

            # Check if message explicitly requests tool use
            should_use_tools = any(keyword in user_message for keyword in [
                'use the', 'using the', 'tool', 'get_weather', 'weather_api', 'search_web',
                'search_database', 'analyze_sentiment', 'calculator',
                'get_system_info', 'process_text',
            ])

            if should_use_tools:
                # Determine which tool based on keywords
                # tools_to_use must be list of ToolPrediction objects with tool_name, arguments, reasoning
                tools_to_use = []
                if 'weather_api' in user_message:
                    # MCP tool: weather_api
                    location = "Tokyo"
                    if 'paris' in user_message:
                        location = "Paris"
                    elif 'london' in user_message:
                        location = "London"
                    elif 'berlin' in user_message:
                        location = "Berlin"
                    tools_to_use = [{
                        "tool_name": "weather_api",
                        "arguments": {"location": location},
                        "reasoning": "Need to check current weather for the requested location",
                    }]
                elif 'weather' in user_message or 'get_weather' in user_message:
                    # Fake tool: get_weather
                    location = "Tokyo"
                    if 'paris' in user_message:
                        location = "Paris"
                    elif 'london' in user_message:
                        location = "London"
                    elif 'berlin' in user_message:
                        location = "Berlin"
                    tools_to_use = [{
                        "tool_name": "get_weather",
                        "arguments": {"location": location},
                        "reasoning": "Need to check current weather for the requested location",
                    }]
                elif 'calculator' in user_message:
                    # MCP tool: calculator
                    tools_to_use = [{
                        "tool_name": "calculator",
                        "arguments": {"operation": "add", "a": 5, "b": 3},
                        "reasoning": "Need to perform calculation",
                    }]
                elif 'search_database' in user_message:
                    # MCP tool: search_database
                    query = "test query"
                    # Try to extract query from quotes
                    import re
                    quoted_match = re.search(r"['\"]([^'\"]+)['\"]", user_message)
                    if quoted_match:
                        query = quoted_match.group(1)
                    elif 'python tutorials' in user_message:
                        query = "python tutorials"
                    elif 'python' in user_message:
                        query = "Python"
                    tools_to_use = [{
                        "tool_name": "search_database",
                        "arguments": {"query": query, "limit": 5},
                        "reasoning": "Need to search database for relevant information",
                    }]
                elif 'search' in user_message or 'search_web' in user_message:
                    # Fake tool: search_web
                    query = "renewable energy"  # default
                    if 'weather patterns' in user_message:
                        query = "weather patterns"
                    elif 'python programming' in user_message or 'python' in user_message:
                        query = "Python programming"
                    tools_to_use = [{
                        "tool_name": "search_web",
                        "arguments": {"query": query, "limit": 5},
                        "reasoning": "Need to search for relevant information",
                    }]
                elif 'sentiment' in user_message or 'analyze_sentiment' in user_message:
                    # Fake tool: analyze_sentiment
                    text = "This is good"
                    tools_to_use = [{
                        "tool_name": "analyze_sentiment",
                        "arguments": {"text": text},
                        "reasoning": "Need to analyze sentiment of the text",
                    }]
                elif 'get_system_info' in user_message:
                    # API MCP tool: get_system_info
                    tools_to_use = [{
                        "tool_name": "get_system_info",
                        "arguments": {},
                        "reasoning": "Need to get system information",
                    }]
                elif 'process_text' in user_message:
                    # API MCP tool: process_text
                    import re
                    text_match = re.search(r"['\"]([^'\"]+)['\"]", user_message)
                    text = text_match.group(1) if text_match else "test text"
                    mode = "uppercase"
                    if "lowercase" in user_message:
                        mode = "lowercase"
                    elif "reverse" in user_message:
                        mode = "reverse"
                    tools_to_use = [{
                        "tool_name": "process_text",
                        "arguments": {"text": text, "mode": mode},
                        "reasoning": "Need to process text",
                    }]

                # Store the tools we decided to use for streaming response generation
                last_tools_used = tools_to_use

                response_json = {
                    "thought": "I need to use a tool to answer this question",
                    "next_action": "use_tools",
                    "tools_to_use": tools_to_use,
                    "concurrent_execution": False,
                }
            else:
                # No tools needed
                last_tools_used = []
                response_json = {
                    "thought": "Processing request",
                    "next_action": "finished",
                    "tools_to_use": [],
                    "concurrent_execution": False,
                }

            import json
            return create_mock_litellm_response(json.dumps(response_json))
        return create_mock_litellm_response(streaming_content or "Response")

    return smart_mock
