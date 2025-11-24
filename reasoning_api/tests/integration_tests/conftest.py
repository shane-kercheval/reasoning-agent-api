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
import subprocess
import json
import pytest
import pytest_asyncio
import asyncpg
from collections.abc import AsyncGenerator
from testcontainers.postgres import PostgresContainer
from litellm import ModelResponse
from litellm.types.utils import StreamingChoices, Delta, Message, Choices, Usage
from reasoning_api.database import ConversationDB
from reasoning_api.dependencies import service_container
from reasoning_api.config import settings


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
    try:
        # Override REASONING_DATABASE_URL for migrations
        env = os.environ.copy()
        env['REASONING_DATABASE_URL'] = postgres_container  # Keep asyncpg URL!

        # Run migrations with environment override
        result = subprocess.run(
            ["uv", "run", "alembic", "-c", "reasoning_api/alembic.ini", "upgrade", "head"],
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
async def conversation_db(postgres_container: str, postgres_engine):  # noqa
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
        try:  # noqa: SIM105
            await trans.rollback()
        except Exception:
            pass  # Transaction might already be in error state

        try:  # noqa: SIM105
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


@pytest.fixture
def integration_workspace(tmp_path):
    """
    Create temporary directory for integration tests.

    Simulates Docker volume mount structure for tools-api integration tests.
    """
    # Create host and container paths
    host_workspace = tmp_path / "workspace"
    host_workspace.mkdir()

    # Simulate container path (for tools-api which uses /mnt/read_write)
    container_workspace = tmp_path / "mnt" / "read_write" / "workspace"
    container_workspace.mkdir(parents=True)

    return {
        "host_workspace": host_workspace,
        "container_workspace": container_workspace,
    }
