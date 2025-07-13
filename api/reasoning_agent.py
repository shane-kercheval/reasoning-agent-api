"""
Advanced reasoning agent that orchestrates OpenAI structured outputs with MCP tool execution.

This agent implements a sophisticated reasoning process that:
1. Uses OpenAI structured outputs to generate reasoning steps
2. Orchestrates parallel MCP tool execution when needed
3. Streams reasoning progress with enhanced metadata
4. Maintains full OpenAI API compatibility

The agent acts as the main orchestrator coordinating between prompts, reasoning,
tool execution, and response synthesis.
"""

import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from openai import AsyncOpenAI

from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    StreamChoice,
    Delta,
    Choice,
    ChatMessage,
    Usage,
)
from .mcp import MCPManager
from .prompt_manager import PromptManager
from .reasoning_models import (
    ReasoningStep,
    ReasoningAction,
    ToolRequest,
    ReasoningEvent,
    ReasoningEventType,
    ReasoningEventStatus,
    ToolResult,
)

logger = logging.getLogger(__name__)


class ReasoningAgent:
    """
    Advanced reasoning agent that orchestrates OpenAI structured outputs with MCP tools.

    This agent implements the complete reasoning workflow:
    1. Loads reasoning prompts from markdown files
    2. Uses OpenAI structured outputs to generate reasoning steps
    3. Orchestrates parallel MCP tool execution when needed
    4. Streams progress with reasoning_event metadata
    5. Synthesizes final responses
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        http_client: httpx.AsyncClient,
        mcp_manager: MCPManager,
        prompt_manager: PromptManager,
    ):
        """
        Initialize the reasoning agent.

        Args:
            base_url: Base URL for the OpenAI-compatible API
            api_key: OpenAI API key for authentication
            http_client: HTTP client for making requests
            mcp_manager: MCP server manager for tool execution
            prompt_manager: Prompt manager for loading reasoning prompts
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.http_client = http_client
        self.mcp_manager = mcp_manager
        self.prompt_manager = prompt_manager

        # Initialize OpenAI client for structured outputs
        self.openai_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
        )

        # Reasoning context
        self.max_reasoning_iterations = 10

        # Ensure auth header is set on http_client
        if 'Authorization' not in self.http_client.headers:
            self.http_client.headers['Authorization'] = f"Bearer {api_key}"

    async def process_chat_completion(
        self, request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """
        Process a non-streaming chat completion request with full reasoning.

        Args:
            request: The chat completion request

        Returns:
            OpenAI-compatible chat completion response with reasoning applied

        Raises:
            httpx.HTTPStatusError: If the OpenAI API returns an error
        """
        try:
            # Run the reasoning process
            reasoning_context = await self._execute_reasoning_process(request)
            # Generate final response using synthesis prompt
            return await self._synthesize_final_response(request, reasoning_context)
        except Exception as e:
            logger.error(f"Error in reasoning process: {e}")
            # Fallback to direct OpenAI call if reasoning fails
            return await self._fallback_to_openai(request, stream=False)

    async def _execute_reasoning_process(self, request: ChatCompletionRequest) -> dict[str, Any]:
        """Execute the full reasoning process with structured outputs and tool orchestration."""
        reasoning_context = {
            "steps": [],
            "tool_results": [],
            "final_thoughts": "",
            "user_request": request,
        }

        # Get reasoning system prompt
        system_prompt = await self.prompt_manager.get_prompt("reasoning_system")

        for iteration in range(self.max_reasoning_iterations):
            # Generate next reasoning step using structured outputs
            reasoning_step = await self._generate_reasoning_step(
                request,
                reasoning_context,
                system_prompt,
            )
            reasoning_context["steps"].append(reasoning_step)

            # Execute tools if needed
            if reasoning_step.tools_to_use:
                tool_results = await self._execute_tools(
                    reasoning_step.tools_to_use,
                    reasoning_step.parallel_execution,
                )
                reasoning_context["tool_results"].extend(tool_results)

            # Check if we should continue reasoning
            if reasoning_step.next_action == ReasoningAction.FINISHED:
                break

        return reasoning_context

    async def _generate_reasoning_step(
        self,
        request: ChatCompletionRequest,
        context: dict[str, Any],
        system_prompt: str,
    ) -> ReasoningStep:
        """Generate a single reasoning step using OpenAI structured outputs."""
        # Build conversation history for reasoning
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Original request: {request.messages[-1].content}"},
        ]

        # Add context from previous steps
        if context["steps"]:
            context_summary = "\n".join([
                f"Step {i+1}: {step.thought}"
                for i, step in enumerate(context["steps"])
            ])
            messages.append({"role": "assistant", "content": f"Previous reasoning:\n{context_summary}"})  # noqa: E501

        # Add tool results if available
        if context["tool_results"]:
            tool_summary = "\n".join([
                f"Tool {result.tool_name}: " +
                (f"SUCCESS - {result.result}" if result.success else f"FAILED - {result.error}")
                for result in context["tool_results"]
            ])
            messages.append({"role": "system", "content": f"Tool execution results:\n{tool_summary}"})  # noqa: E501

        # Get available tools from MCP manager
        available_tools = await self.mcp_manager.get_available_tools()
        tool_descriptions = "\n".join([
            f"- {tool['name']}: {tool.get('description', 'No description')}"
            for tool in available_tools
        ])
        messages.append({"role": "system", "content": f"Available tools:\n{tool_descriptions}"})

        # Add JSON schema instructions to the system prompt
        json_schema = ReasoningStep.model_json_schema()
        schema_instructions = f"""
You must respond with valid JSON that matches this exact schema:
{json.dumps(json_schema, indent=2)}

Your response must be valid JSON only, no other text.
"""
        # Update the last message to include schema instructions
        messages[-1]["content"] += schema_instructions

        # Request reasoning step using JSON mode
        try:
            response = await self.openai_client.chat.completions.create(
                model=request.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1,
            )

            if response.choices and response.choices[0].message.content:
                try:
                    json_response = json.loads(response.choices[0].message.content)
                    return ReasoningStep.model_validate(json_response)
                except (json.JSONDecodeError, ValueError) as parse_error:
                    logger.warning(f"Failed to parse JSON response: {parse_error}")
                    logger.warning(f"Raw response: {response.choices[0].message.content}")

            # Fallback - create a simple reasoning step
            logger.warning(f"Unexpected response format: {response}")
            return ReasoningStep(
                thought="Unable to generate structured reasoning step",
                next_action=ReasoningAction.FINISHED,
                tools_to_use=[],
                parallel_execution=False,
            )
        except Exception as e:
            logger.warning(f"JSON mode parsing failed: {e}")
            # Fallback - create a simple reasoning step
            return ReasoningStep(
                thought="Error in reasoning - proceeding to final answer",
                next_action=ReasoningAction.FINISHED,
                tools_to_use=[],
                parallel_execution=False,
            )

    async def _execute_tools(
            self,
            tool_requests: list[ToolRequest],
            parallel: bool = False,
        ) -> list[ToolResult]:
        """Execute tool requests through MCP manager."""
        if parallel:
            return await self.mcp_manager.execute_tools_parallel(tool_requests)
        results = []
        for tool_request in tool_requests:
            result = await self.mcp_manager.execute_tool(tool_request)
            results.append(result)
        return results

    async def _synthesize_final_response(
        self,
        request: ChatCompletionRequest,
        reasoning_context: dict[str, Any],
    ) -> ChatCompletionResponse:
        """Synthesize final response using reasoning context."""
        # Get synthesis prompt
        synthesis_prompt = await self.prompt_manager.get_prompt("final_answer")
        # Build synthesis messages
        messages = [
            {"role": "system", "content": synthesis_prompt},
            {"role": "user", "content": f"Original request: {request.messages[-1].content}"},
        ]
        # Add reasoning summary
        reasoning_summary = self._build_reasoning_summary(reasoning_context)
        messages.append({
            "role": "assistant",
            "content": f"My reasoning process:\n{reasoning_summary}",
        })
        # Generate final response
        response = await self.openai_client.chat.completions.create(
            model=request.model,
            messages=messages,
            temperature=request.temperature or 0.7,
            max_tokens=request.max_tokens,
        )
        # Convert OpenAI response to our Pydantic models

        choices = []
        for choice in response.choices:
            choices.append(Choice(
                index=choice.index,
                message=ChatMessage(
                    role=choice.message.role,
                    content=choice.message.content,
                ),
                finish_reason=choice.finish_reason,
            ))

        usage = None
        if response.usage:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return ChatCompletionResponse(
            id=response.id,
            created=response.created,
            model=response.model,
            choices=choices,
            usage=usage,
        )

    def _build_reasoning_summary(self, context: dict[str, Any]) -> str:
        """Build a summary of the reasoning process for final synthesis."""
        summary_parts = []

        for i, step in enumerate(context["steps"]):
            summary_parts.append(f"Step {i+1}: {step.thought}")
            if step.tools_to_use:
                tool_names = [tool.tool_name for tool in step.tools_to_use]
                summary_parts.append(f"  Used tools: {', '.join(tool_names)}")

        if context["tool_results"]:
            summary_parts.append("\nTool Results:")
            for result in context["tool_results"]:
                summary_parts.append(f"- {result.tool_name}: {result.result}")

        return "\n".join(summary_parts)

    async def _fallback_to_openai(
            self,
            request: ChatCompletionRequest,
            stream: bool = False,
        ) -> ChatCompletionResponse:
        """Fallback to direct OpenAI call when reasoning fails."""
        payload = request.model_dump(exclude_unset=True)
        payload['stream'] = stream

        response = await self.http_client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
        )
        response.raise_for_status()

        if stream:
            # Return streaming response (not implemented in this fallback)
            raise NotImplementedError("Streaming fallback not implemented")
        response_data = response.json()
        return ChatCompletionResponse(**response_data)

    async def process_chat_completion_stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncGenerator[str]:
        """
        Process a streaming chat completion request with reasoning steps.

        Args:
            request: The chat completion request

        Yields:
            Server-sent event formatted strings compatible with OpenAI streaming API

        Raises:
            httpx.HTTPStatusError: If the OpenAI API returns an error
        """
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        created = int(time.time())

        try:
            # Stream the reasoning process with reasoning_event metadata
            async for reasoning_chunk in self._stream_reasoning_process(
                request, completion_id, created,
            ):
                yield f"data: {reasoning_chunk}\n\n"

            # Stream the final synthesized response
            async for final_chunk in self._stream_final_response(
                request, completion_id, created,
            ):
                yield f"data: {final_chunk}\n\n"

        except Exception as e:
            logger.error(f"Error in streaming reasoning process: {e}")
            # Fallback to enhanced request streaming
            enhanced_request = await self._enhance_request(request)
            async for openai_chunk in self._stream_openai_response(
                enhanced_request, completion_id, created,
            ):
                yield f"data: {openai_chunk}\n\n"

        yield "data: [DONE]\n\n"

    async def _stream_reasoning_process(
        self,
        request: ChatCompletionRequest,
        completion_id: str,
        created: int,
    ) -> AsyncGenerator[str]:
        """Stream the reasoning process with reasoning_event metadata."""
        reasoning_context = {
            "steps": [],
            "tool_results": [],
            "final_thoughts": "",
            "user_request": request,
        }

        # Get reasoning system prompt
        system_prompt = await self.prompt_manager.get_prompt("reasoning_system")

        for iteration in range(self.max_reasoning_iterations):
            # Emit reasoning start event
            start_event = ReasoningEvent(
                type=ReasoningEventType.REASONING_STEP,
                step_id=str(iteration + 1),
                status=ReasoningEventStatus.IN_PROGRESS,
                metadata={"thought": f"Starting reasoning step {iteration + 1}..."},
            )

            chunk = ChatCompletionStreamResponse(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[
                    StreamChoice(
                        index=0,
                        delta=Delta(reasoning_event=start_event),
                        finish_reason=None,
                    ),
                ],
            )
            yield chunk.model_dump_json()

            # Generate reasoning step
            reasoning_step = await self._generate_reasoning_step(
                request,
                reasoning_context,
                system_prompt,
            )
            reasoning_context["steps"].append(reasoning_step)

            # Emit reasoning step content
            step_event = ReasoningEvent(
                type=ReasoningEventType.REASONING_STEP,
                step_id=str(iteration + 1),
                status=ReasoningEventStatus.COMPLETED,
                metadata={
                    "thought": reasoning_step.thought,
                    "tools_planned": [tool.tool_name for tool in reasoning_step.tools_to_use],
                },
            )

            chunk = ChatCompletionStreamResponse(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[
                    StreamChoice(
                        index=0,
                        delta=Delta(reasoning_event=step_event),
                        finish_reason=None,
                    ),
                ],
            )
            yield chunk.model_dump_json()

            # Execute tools if needed
            if reasoning_step.tools_to_use:
                # Emit tool execution start
                tool_start_event = ReasoningEvent(
                    type=ReasoningEventType.TOOL_EXECUTION,
                    step_id=f"{iteration + 1}-tools",
                    status=ReasoningEventStatus.IN_PROGRESS,
                    tools=[tool.tool_name for tool in reasoning_step.tools_to_use],
                )

                chunk = ChatCompletionStreamResponse(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    choices=[
                        StreamChoice(
                            index=0,
                            delta=Delta(reasoning_event=tool_start_event),
                            finish_reason=None,
                        ),
                    ],
                )
                yield chunk.model_dump_json()

                # Execute tools
                tool_results = await self._execute_tools(
                    reasoning_step.tools_to_use,
                    reasoning_step.parallel_execution,
                )
                reasoning_context["tool_results"].extend(tool_results)

                # Emit tool execution completed
                tool_complete_event = ReasoningEvent(
                    type=ReasoningEventType.TOOL_EXECUTION,
                    step_id=f"{iteration + 1}-tools",
                    status=ReasoningEventStatus.COMPLETED,
                    tools=[result.tool_name for result in tool_results],
                )

                chunk = ChatCompletionStreamResponse(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    choices=[
                        StreamChoice(
                            index=0,
                            delta=Delta(reasoning_event=tool_complete_event),
                            finish_reason=None,
                        ),
                    ],
                )
                yield chunk.model_dump_json()

            # Check if we should continue reasoning
            if reasoning_step.next_action == ReasoningAction.FINISHED:
                break

        # Emit reasoning completion
        complete_event = ReasoningEvent(
            type=ReasoningEventType.SYNTHESIS,
            step_id="final",
            status=ReasoningEventStatus.COMPLETED,
            metadata={"total_steps": len(reasoning_context["steps"])},
        )

        chunk = ChatCompletionStreamResponse(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[
                StreamChoice(
                    index=0,
                    delta=Delta(reasoning_event=complete_event),
                    finish_reason=None,
                ),
            ],
        )
        yield chunk.model_dump_json()

    async def _stream_final_response(
        self,
        request: ChatCompletionRequest,
        completion_id: str,
        created: int,
    ) -> AsyncGenerator[str]:
        """Stream the final synthesized response."""
        # Get synthesis prompt and build response
        synthesis_prompt = await self.prompt_manager.get_prompt("final_answer")

        # Build synthesis messages (simplified for streaming)
        messages = [
            {"role": "system", "content": synthesis_prompt},
        ] + [msg.model_dump() for msg in request.messages]

        # Stream synthesis response
        payload = {
            "model": request.model,
            "messages": messages,
            "stream": True,
            "temperature": request.temperature or 0.7,
        }

        async with self.http_client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=payload,
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix
                    if data == "[DONE]":
                        break
                    try:
                        # Parse and update with our completion_id and created timestamp
                        chunk_data = json.loads(data)
                        chunk_data["id"] = completion_id
                        chunk_data["created"] = created
                        yield json.dumps(chunk_data)
                    except json.JSONDecodeError:
                        # Skip malformed JSON
                        continue


    async def _stream_openai_response(
        self,
        request: ChatCompletionRequest,
        completion_id: str,
        created: int,
    ) -> AsyncGenerator[str]:
        """
        Stream response from OpenAI API.

        Args:
            request: The chat completion request
            completion_id: Unique completion ID to use
            created: Timestamp to use

        Yields:
            JSON strings representing OpenAI response chunks
        """
        payload = request.model_dump(exclude_unset=True)
        payload['stream'] = True

        async with self.http_client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=payload,
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix
                    if data == "[DONE]":
                        break
                    try:
                        # Parse and update with our completion_id and created timestamp
                        chunk_data = json.loads(data)
                        chunk_data["id"] = completion_id
                        chunk_data["created"] = created
                        yield json.dumps(chunk_data)
                    except json.JSONDecodeError:
                        # Skip malformed JSON
                        continue

    async def _enhance_request(
        self, request: ChatCompletionRequest,
    ) -> ChatCompletionRequest:
        """
        Enhance request with MCP tools if available.

        Args:
            request: Original chat completion request

        Returns:
            Enhanced request (currently just returns original)
        """
        # Placeholder for future MCP integration
        # For now, just return the original request
        return request
