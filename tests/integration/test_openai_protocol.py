"""
OpenAI Protocol Validation Integration Tests.

CRITICAL: These tests validate that our OpenAI protocol abstractions
(builders, parsers, response structures) work correctly with the REAL OpenAI API.

This is essential because:
1. Proves our request builders create valid OpenAI requests
2. Proves our response parsers handle real OpenAI responses
3. Catches when OpenAI changes their API format
4. Ensures our mocks actually match reality

All tests require OPENAI_API_KEY and use real API calls.
"""

import os
import json
import pytest
from openai import AsyncOpenAI
from api.openai_protocol import (
    OpenAIRequestBuilder,
    OpenAIResponseBuilder,
    OpenAIStreamingResponseBuilder,
    OpenAIResponseParser,
)
from tests.utils.openai_test_helpers import (
    create_simple_chat_request,
    create_streaming_chunks,
)
from dotenv import load_dotenv
load_dotenv()


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Needs real OpenAI API key")
@pytest.mark.asyncio
class TestOpenAIRequestBuilderValidation:
    """Validate that our request builder creates requests OpenAI accepts."""

    @pytest.fixture
    def real_openai_client(self):
        """Real OpenAI client for testing."""
        return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def test_basic_chat_request_accepted_by_openai(self, real_openai_client: AsyncOpenAI):
        """Test that our basic request builder creates valid OpenAI requests."""
        # Build request using our builder
        request = (
            OpenAIRequestBuilder()
            .model("gpt-4o-mini")
            .message("user", "Say 'test' exactly")
            .temperature(0.1)
            .max_completion_tokens(10)
            .build()
        )

        # Send to real OpenAI - if builder is wrong, this will fail
        response = await real_openai_client.chat.completions.create(**request)

        # If we get here, our request format is correct
        assert response.id.startswith("chatcmpl-")
        assert response.model.startswith("gpt-")
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None

    async def test_json_mode_request_accepted_by_openai(self, real_openai_client: AsyncOpenAI):
        """Test that our JSON mode request builder works with real OpenAI."""
        request = (
            OpenAIRequestBuilder()
            .model("gpt-4o-mini")
            .message("user", "Return JSON with a 'greeting' field containing 'hello'")
            .json_mode()
            .temperature(0.1)
            .build()
        )

        response = await real_openai_client.chat.completions.create(**request)

        # Should get valid JSON back
        content = response.choices[0].message.content
        parsed_json = json.loads(content)  # Should not raise
        assert "greeting" in parsed_json
        assert "hello" in parsed_json["greeting"].lower()

    async def test_structured_output_request_accepted_by_openai(
        self, real_openai_client: AsyncOpenAI,
    ):
        """Test that our structured output request builder works with real OpenAI."""
        schema = {
            "name": "weather_response",
            "schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "temperature": {"type": "string"},
                    "condition": {"type": "string"},
                },
                "required": ["location", "temperature", "condition"],
                "additionalProperties": False,
            },
            "strict": True,
        }

        request = (
            OpenAIRequestBuilder()
            .model("gpt-4o-mini")
            .message("user", "Give me weather for Tokyo")
            .structured_output(schema)
            .temperature(0.1)
            .build()
        )

        response = await real_openai_client.chat.completions.create(**request)

        # Should get structured JSON matching our schema
        content = response.choices[0].message.content
        parsed = json.loads(content)
        assert "location" in parsed
        assert "temperature" in parsed
        assert "condition" in parsed

    async def test_streaming_request_accepted_by_openai(self, real_openai_client: AsyncOpenAI):
        """Test that our streaming request builder works with real OpenAI."""
        request = (
            OpenAIRequestBuilder()
            .model("gpt-4o-mini")
            .message("user", "Count to 3")
            .streaming()
            .temperature(0.1)
            .build()
        )

        # Should accept streaming request without error
        stream = await real_openai_client.chat.completions.create(**request)

        chunks_received = 0
        async for chunk in stream:
            chunks_received += 1
            # Basic validation that chunks have expected structure
            assert hasattr(chunk, 'id')
            assert hasattr(chunk, 'choices')
            if chunks_received > 10:  # Don't wait for full response
                break

        assert chunks_received > 0

    async def test_all_optional_parameters_accepted_by_openai(
        self, real_openai_client: AsyncOpenAI,
    ):
        """Test that all our optional parameters are accepted by OpenAI."""
        request = (
            OpenAIRequestBuilder()
            .model("gpt-4o-mini")
            .message("user", "Say hello")
            .temperature(0.5)
            .top_p(0.9)
            .max_completion_tokens(20)
            .frequency_penalty(0.1)
            .presence_penalty(0.1)
            .seed(12345)
            .stop(["END"])
            .user("test-user")
            .build()
        )

        # All parameters should be accepted
        response = await real_openai_client.chat.completions.create(**request)
        assert response.id.startswith("chatcmpl-")


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Needs real OpenAI API key")
@pytest.mark.asyncio
class TestOpenAIResponseParserValidation:
    """Validate that our parser handles real OpenAI responses correctly."""

    @pytest.fixture
    def real_openai_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def test_parser_handles_real_chat_response(self, real_openai_client: AsyncOpenAI):
        """Test that our parser can handle real OpenAI chat responses."""
        response = await real_openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'test'"}],
            max_completion_tokens=10,
        )

        # Convert real response to dict (as it comes from HTTP)
        response_dict = response.model_dump()

        # Our parser should handle it without errors
        parsed = OpenAIResponseParser.parse_chat_response(response_dict)

        # Verify our parser extracted the same data
        assert parsed.id == response.id
        assert parsed.model == response.model
        assert len(parsed.choices) == len(response.choices)
        assert parsed.choices[0].message.content == response.choices[0].message.content
        assert parsed.usage.total_tokens == response.usage.total_tokens

    async def test_parser_handles_real_streaming_response(self, real_openai_client: AsyncOpenAI):
        """Test that our streaming parser handles real OpenAI streaming."""
        stream = await real_openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Count to 3"}],
            stream=True,
            max_completion_tokens=20,
        )

        chunks_parsed = 0
        async for chunk in stream:
            # Convert to SSE format our parser expects
            chunk_dict = chunk.model_dump()
            sse_line = f"data: {json.dumps(chunk_dict)}\n\n"

            # Our parser should handle it
            parsed_chunk = OpenAIResponseParser.parse_streaming_chunk(sse_line)
            if parsed_chunk and not parsed_chunk.get("done"):
                chunks_parsed += 1

                # Verify basic structure
                assert "id" in parsed_chunk
                assert "choices" in parsed_chunk

            if chunks_parsed > 5:  # Don't process entire stream
                break

        assert chunks_parsed > 0

    async def test_parser_handles_real_json_mode_response(self, real_openai_client: AsyncOpenAI):
        """Test parser with real JSON mode responses."""
        response = await real_openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Return JSON with a 'test' field"}],
            response_format={"type": "json_object"},
        )

        response_dict = response.model_dump()
        parsed = OpenAIResponseParser.parse_chat_response(response_dict)
        # Should handle JSON mode response structure
        assert parsed.choices[0].message.content is not None
        # Content should be valid JSON
        json.loads(parsed.choices[0].message.content)


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Needs real OpenAI API key")
@pytest.mark.asyncio
class TestMockResponseStructureValidation:
    """Critical: Validate our mock responses match real OpenAI exactly."""

    @pytest.fixture
    def real_openai_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def test_mock_response_structure_matches_real_openai(
        self, real_openai_client: AsyncOpenAI,
    ):
        """CRITICAL: Verify our mock responses have identical structure to real OpenAI."""
        # Get real response
        real_response = await real_openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=10,
        )
        real_dict = real_response.model_dump()

        # Create mock response with same basic data
        mock_response = (
            OpenAIResponseBuilder()
            .id("chatcmpl-mock123")
            .model("gpt-4o-mini")
            .created(real_dict["created"])
            .choice(0, "assistant", "Hello!")
            .usage(10, 5)
            .system_fingerprint("fp_mock")
            .service_tier("default")
            .build()
        )

        # Structure should be identical (ignoring content values)
        assert set(mock_response.keys()) == set(real_dict.keys())
        assert mock_response["object"] == real_dict["object"]
        assert len(mock_response["choices"]) == len(real_dict["choices"])
        assert set(mock_response["choices"][0].keys()) == set(real_dict["choices"][0].keys())
        assert set(mock_response["usage"].keys()) == set(real_dict["usage"].keys())

    async def test_mock_streaming_structure_matches_real_openai(
        self, real_openai_client: AsyncOpenAI,
    ):
        """Verify our mock streaming chunks match real OpenAI streaming."""
        stream = await real_openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
            max_completion_tokens=5,
        )

        real_chunk = None
        async for chunk in stream:
            real_chunk = chunk.model_dump()
            break  # Just get first chunk

        # Create mock streaming chunk
        mock_stream = (
            OpenAIStreamingResponseBuilder()
            .chunk("chatcmpl-mock", "gpt-4o-mini", delta_content="Hi")
            .build()
        )

        # Parse mock chunk to compare structure
        mock_lines = mock_stream.strip().split('\n\n')
        mock_chunk_data = json.loads(mock_lines[0][6:])  # Remove "data: "

        # Structure should match
        assert set(mock_chunk_data.keys()) == set(real_chunk.keys())
        assert mock_chunk_data["object"] == real_chunk["object"]
        assert len(mock_chunk_data["choices"]) == len(real_chunk["choices"])


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Needs real OpenAI API key")
@pytest.mark.asyncio
class TestOpenAIAPIChangesDetection:
    """Tests that would catch if OpenAI changes their API format."""

    @pytest.fixture
    def real_openai_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def test_basic_response_fields_still_exist(self, real_openai_client: AsyncOpenAI):
        """Test that OpenAI still returns expected fields."""
        response = await real_openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
        )

        response_dict = response.model_dump()

        # Critical fields that our code depends on
        required_fields = ["id", "object", "created", "model", "choices", "usage"]
        for field in required_fields:
            assert field in response_dict, f"OpenAI removed required field: {field}"

        # Choice structure
        choice = response_dict["choices"][0]
        choice_fields = ["index", "message", "finish_reason"]
        for field in choice_fields:
            assert field in choice, f"OpenAI changed choice structure, missing: {field}"

        # Message structure
        message = choice["message"]
        message_fields = ["role", "content"]
        for field in message_fields:
            assert field in message, f"OpenAI changed message structure, missing: {field}"

    async def test_streaming_chunk_fields_still_exist(self, real_openai_client: AsyncOpenAI):
        """Test that OpenAI streaming chunks still have expected structure."""
        stream = await real_openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
            max_completion_tokens=5,
        )

        chunk_fields_found = set()
        async for chunk in stream:
            chunk_dict = chunk.model_dump()
            chunk_fields_found.update(chunk_dict.keys())

            if chunk_dict.get("choices"):
                choice = chunk_dict["choices"][0]
                if "delta" in choice:
                    break  # Found a content chunk

        # Critical fields for streaming
        required_streaming_fields = ["id", "object", "created", "model", "choices"]
        for field in required_streaming_fields:
            assert field in chunk_fields_found, f"OpenAI streaming missing field: {field}"


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Needs real OpenAI API key")
@pytest.mark.asyncio
class TestConvenienceFunctionValidation:
    """Test our convenience functions work with real OpenAI."""

    @pytest.fixture
    def real_openai_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def test_create_simple_chat_request_works(self, real_openai_client: AsyncOpenAI):
        """Test our convenience function creates valid requests."""
        request = create_simple_chat_request("gpt-4o-mini", "Say hello")

        # Should work with real OpenAI
        response = await real_openai_client.chat.completions.create(**request)
        assert "hello" in response.choices[0].message.content.lower()

    async def test_create_streaming_chunks_matches_real_format(self):
        """Test our convenience streaming function matches real format."""
        # This is harder to test directly, but we can validate the format
        chunks = create_streaming_chunks("chatcmpl-test", "gpt-4o-mini", "Hello world")

        lines = chunks.strip().split('\n\n')

        # Should end with [DONE]
        assert lines[-1] == "data: [DONE]"

        # Each chunk should be valid JSON
        for line in lines[:-1]:  # Exclude [DONE]
            assert line.startswith("data: ")
            chunk_data = json.loads(line[6:])
            assert "id" in chunk_data
            assert "object" in chunk_data
            assert chunk_data["object"] == "chat.completion.chunk"
