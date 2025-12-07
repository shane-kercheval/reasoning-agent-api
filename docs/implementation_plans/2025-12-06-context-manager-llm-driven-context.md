# LLM-Driven Context Management

## Overview

This plan introduces intelligent, LLM-driven context management to replace the current naive truncation approach. Instead of blindly truncating large content or using recency-based message selection, we use an LLM to analyze all available information and make intelligent decisions about what to include and how to process it.

**Core Insight**: Give an LLM all artifacts (previews + metadata), the user's question, and the token budget. Let it return a prioritized list with processing strategies for each artifact.

**Problem Statement**:

1. **Greedy fit bug**: Current algorithm skips large messages, potentially creating incoherent context
2. **No relevance filtering**: Uses recency, not semantic relevance to the user's question
3. **Naive truncation**: `preview()` just truncates; doesn't intelligently summarize based on what's needed
4. **No caching**: Re-processes same content repeatedly
5. **No document support**: Can't attach documents to conversations
6. **Single model**: One model used for everything; no way to use cheaper models for internal tasks

**Target State**:

- Intelligent decision about when LLM planning is worth the overhead
- Single LLM call plans entire context assembly (using cheap/fast model)
- Each artifact gets appropriate processing: include full, summarize, RAG extract, or exclude
- Summaries are cached to avoid re-computation
- Documents can be attached to requests and flow through the same pipeline as tool results
- Configurable models for different internal tasks (planning, summarization)

---

## Milestone 1: Artifact Abstraction

### Goal
Create a unified `Artifact` model that represents any piece of information that may go into LLM context (tool results, documents, conversation messages). This abstraction enables the LLM context planner to treat all information sources uniformly.

### Success Criteria
- `Artifact` and `ArtifactType` models defined
- `ProcessingStrategy` and `ArtifactDecision` models defined for LLM output
- Helper function to extract artifacts from `ReasoningStepRecord` list
- Existing tests continue to pass

### Key Changes

**New file: `reasoning_api/reasoning_api/artifacts.py`**

```python
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Any

class ArtifactType(str, Enum):
    TOOL_RESULT = "tool_result"       # Ephemeral, conversation-scoped
    DOCUMENT = "document"              # Persistent, reusable across conversations

class ProcessingStrategy(str, Enum):
    INCLUDE_FULL = "include_full"     # Small, relevant → include as-is
    SUMMARIZE = "summarize"           # Large, relevant → LLM summarize
    RAG_EXTRACT = "rag_extract"       # Large, specific lookup needed → semantic search
    EXCLUDE = "exclude"               # Not relevant → omit

class Artifact(BaseModel):
    """A discrete piece of information that may go into LLM context."""

    artifact_id: str                  # Hash or unique ID for caching
    artifact_type: ArtifactType
    content: Any                      # Raw content (dict, list, str, etc.)
    size_chars: int                   # For quick size checks

    # Source tracking
    source: str                       # e.g., "step_1:web_scraper", "doc:quarterly_report"
    timestamp: datetime

    # For tool results: link back to reasoning step
    step_index: int | None = None
    tool_name: str | None = None

    # Metadata for decision-making
    metadata: dict[str, Any] = Field(default_factory=dict)

class ArtifactDecision(BaseModel):
    """LLM's decision for how to handle a single artifact."""

    artifact_id: str
    strategy: ProcessingStrategy
    priority: int                     # 1 = highest priority
    relevance_rationale: str          # Why relevant (or not)

    # Strategy-specific guidance (populated based on strategy)
    summarization_focus: str | None = None   # For SUMMARIZE: what to extract
    extraction_query: str | None = None      # For RAG_EXTRACT: what to search for

class ArtifactPlan(BaseModel):
    """LLM's complete plan for context assembly."""

    artifacts: list[ArtifactDecision]  # Ordered by priority (highest first)
    reasoning: str                      # Overall rationale for the plan
```

**Helper function to extract artifacts from step records:**

```python
def extract_artifacts_from_steps(
    step_records: list[ReasoningStepRecord],
) -> list[Artifact]:
    """Convert tool results from reasoning steps into Artifact format."""
    artifacts = []
    for record in step_records:
        for i, result in enumerate(record.tool_results):
            content = result.result if result.success else result.error
            content_str = json.dumps(content) if isinstance(content, (dict, list)) else str(content)

            artifacts.append(Artifact(
                artifact_id=f"step_{record.step_index}_tool_{i}_{result.tool_name}",
                artifact_type=ArtifactType.TOOL_RESULT,
                content=content,
                size_chars=len(content_str),
                source=f"step_{record.step_index}:{result.tool_name}",
                timestamp=record.timestamp,
                step_index=record.step_index,
                tool_name=result.tool_name,
                metadata={
                    "success": result.success,
                    "execution_time_ms": result.execution_time_ms,
                },
            ))
    return artifacts
```

### Testing Strategy
- Test `Artifact` instantiation with various content types (dict, list, str)
- Test `extract_artifacts_from_steps` with empty list, single step, multiple steps
- Test artifact_id generation is unique and deterministic
- Test size_chars calculation accuracy

### Dependencies
None - this is foundational.

### Risk Factors
- Need to decide on artifact_id format (affects caching later)
- Consider whether `content` should be stored as-is or always serialized

---

## Milestone 2: Document Attachment Support

### Goal
Enable documents to be attached to chat completion requests. Documents flow through the same pipeline as tool results but have different lifecycle (persistent vs. ephemeral).

### Success Criteria
- `DocumentAttachment` model added to `openai_protocol.py`
- `OpenAIChatRequest` extended with optional `documents` field
- Documents converted to `Artifact` format in request handling
- Passthrough and ReasoningAgent both receive document artifacts

### Key Changes

**Update `reasoning_api/reasoning_api/openai_protocol.py`:**

```python
class DocumentAttachment(BaseModel):
    """Document attached to a conversation for context."""

    model_config = ConfigDict(extra='allow')

    # Identity (at least one required)
    document_id: str | None = None      # Reference to stored doc (future: memory/DB)
    content: str | None = None          # Inline content

    # Metadata
    title: str | None = None
    source: str | None = None           # URL, filepath, etc.

    @model_validator(mode='after')
    def validate_content_source(self) -> 'DocumentAttachment':
        if not self.document_id and not self.content:
            raise ValueError("Either document_id or content must be provided")
        return self


class OpenAIChatRequest(BaseModel):
    # ... existing fields ...

    # Extensions (not in OpenAI spec)
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    documents: list[DocumentAttachment] | None = None  # NEW
```

**Helper to convert documents to artifacts:**

```python
def documents_to_artifacts(
    documents: list[DocumentAttachment],
) -> list[Artifact]:
    """Convert document attachments to Artifact format."""
    artifacts = []
    for i, doc in enumerate(documents):
        content = doc.content or ""  # Future: fetch by document_id
        artifacts.append(Artifact(
            artifact_id=doc.document_id or f"doc_inline_{i}_{hash(content)}",
            artifact_type=ArtifactType.DOCUMENT,
            content=content,
            size_chars=len(content),
            source=doc.source or doc.title or f"document_{i}",
            timestamp=datetime.now(UTC),
            metadata={"title": doc.title, "source": doc.source},
        ))
    return artifacts
```

### Testing Strategy
- Test `DocumentAttachment` validation (requires document_id OR content)
- Test `OpenAIChatRequest` with documents field (valid and invalid)
- Test `documents_to_artifacts` conversion
- Integration test: request with documents reaches executor

### Dependencies
- Milestone 1 (Artifact model)

### Risk Factors
- `document_id` lookup not implemented yet (just inline content for now)
- Need to coordinate with future memory/storage system for document persistence

---

## Milestone 3: Context Model Update

### Goal
Update the `Context` model to include documents alongside step_records. This unifies all context sources (conversation history, tool results, documents) into a single interface for the ContextManager.

### Success Criteria
- `Context` model includes `documents` field
- ContextManager's `_build_reasoning_context` handles documents
- Documents are formatted and included in LLM context (simple inclusion for now)

### Key Changes

**Update `reasoning_api/reasoning_api/context_manager.py`:**

```python
class Context(BaseModel):
    """Unified context for LLMs."""

    conversation_history: list[dict[str, str]]

    # Reasoning-specific
    step_records: list[ReasoningStepRecord] = Field(default_factory=list)
    goal: ContextGoal | None = None
    system_prompt_override: str | None = None

    # Document context (NEW)
    documents: list[Artifact] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}
```

**Update `_build_reasoning_context` to include documents:**

For this milestone, simply append documents to context (intelligent processing comes in Milestone 6):

```python
def _build_reasoning_context(self, model_name: str, context: Context):
    # ... existing logic for step_records ...

    # Add documents section if present
    if context.documents:
        doc_parts = []
        for doc in context.documents:
            title = doc.metadata.get("title", doc.source)
            doc_parts.append(f"### {title}\n\n{doc.content}")

        messages.append({
            "role": "user",
            "content": "## Attached Documents\n\n" + "\n\n---\n\n".join(doc_parts),
        })

    # ... rest of existing logic ...
```

### Testing Strategy
- Test `Context` with documents field populated
- Test `_build_reasoning_context` includes documents in output messages
- Test document formatting in context (title, content structure)
- Test with empty documents list (no change to output)

### Dependencies
- Milestone 1 (Artifact model)
- Milestone 2 (DocumentAttachment)

### Risk Factors
- Large documents will bloat context (addressed in Milestone 6)
- Document order in context may matter for some queries

---

## Milestone 4: Model Configuration

### Goal
Create a `ModelConfig` abstraction that separates the primary model (from request) from internal task models (context planning, summarization). This allows using cheap/fast models for internal operations while respecting the user's model choice for reasoning and synthesis.

### Success Criteria
- `ModelConfig` model defined with primary, context_planner, and summarizer fields
- Environment settings for default internal models
- `ModelConfig.from_request()` factory method
- Model info utilities for cost/capability lookup

### Key Changes

**New file: `reasoning_api/reasoning_api/model_config.py`**

```python
from pydantic import BaseModel
from litellm import get_model_info

class ModelConfig(BaseModel):
    """Configuration for which models to use for different tasks."""

    primary: str              # From request - used for reasoning steps + synthesis
    context_planner: str      # For planning context assembly (cheap/fast)
    summarizer: str           # For summarizing artifacts (cheap/fast)

    @classmethod
    def from_request(
        cls,
        request_model: str,
        context_planner_override: str | None = None,
        summarizer_override: str | None = None,
    ) -> "ModelConfig":
        """
        Create ModelConfig from request model with environment defaults.

        Args:
            request_model: The model from the user's request
            context_planner_override: Override for context planner (from env/settings)
            summarizer_override: Override for summarizer (from env/settings)
        """
        from reasoning_api.config import settings

        return cls(
            primary=request_model,
            context_planner=context_planner_override or settings.CONTEXT_PLANNER_MODEL,
            summarizer=summarizer_override or settings.SUMMARIZER_MODEL,
        )

    def get_primary_info(self) -> dict:
        """Get model info for the primary model."""
        return get_model_info(model=self.primary)

    def is_expensive_primary(self, threshold: float = 0.00001) -> bool:
        """
        Check if the primary model is expensive (affects context planning decisions).

        Args:
            threshold: Cost per token threshold (default catches o1, opus tier)
        """
        try:
            info = self.get_primary_info()
            input_cost = info.get('input_cost_per_token', 0) or 0
            return input_cost > threshold
        except Exception:
            return False  # Assume not expensive if we can't determine
```

**Update `reasoning_api/reasoning_api/config.py`** (or create if needed):

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # ... existing settings ...

    # Internal model configuration
    CONTEXT_PLANNER_MODEL: str = "gpt-4o-mini"
    SUMMARIZER_MODEL: str = "gpt-4o-mini"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
```

### Testing Strategy
- Test `ModelConfig.from_request()` with defaults
- Test `ModelConfig.from_request()` with overrides
- Test `is_expensive_primary()` with known expensive models (mock get_model_info)
- Test `is_expensive_primary()` with cheap models
- Test fallback when model info unavailable

### Dependencies
None - foundational for LLM planning.

### Risk Factors
- `get_model_info` may not have info for all models (need graceful fallback)
- Cost thresholds may need tuning based on real-world model pricing

---

## Milestone 5: Summary Cache

### Goal
Implement a caching layer for artifact summaries to avoid re-summarizing the same content. The cache key includes both the artifact content hash and the summarization focus (since the same artifact may need different summaries for different questions).

### Success Criteria
- `SummaryCache` abstraction with get/set operations
- In-memory backend implementation (production Redis backend is future work)
- Cache key = `artifact_id + focus_hash`
- Cache hit/miss tracking in metadata

### Key Changes

**New file: `reasoning_api/reasoning_api/summary_cache.py`**

```python
import hashlib
from abc import ABC, abstractmethod
from typing import Any

class CacheBackend(ABC):
    """Abstract cache backend interface."""

    @abstractmethod
    async def get(self, key: str) -> str | None:
        """Get cached value by key."""
        pass

    @abstractmethod
    async def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None:
        """Set cached value with optional TTL."""
        pass

class InMemoryCache(CacheBackend):
    """Simple in-memory cache for development/testing."""

    def __init__(self):
        self._cache: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self._cache.get(key)

    async def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None:
        # Note: TTL not implemented for in-memory (acceptable for dev)
        self._cache[key] = value

class SummaryCache:
    """
    Cache for artifact summaries.

    Key insight: Same artifact may need different summaries depending on
    what we're asking about it. Cache key = artifact_id + focus_hash.
    """

    def __init__(self, backend: CacheBackend):
        self.backend = backend
        self.hits = 0
        self.misses = 0

    def _cache_key(self, artifact_id: str, focus: str | None) -> str:
        focus_hash = hashlib.sha256((focus or "").encode()).hexdigest()[:16]
        return f"summary:{artifact_id}:{focus_hash}"

    async def get(self, artifact_id: str, focus: str | None = None) -> str | None:
        key = self._cache_key(artifact_id, focus)
        result = await self.backend.get(key)
        if result:
            self.hits += 1
        else:
            self.misses += 1
        return result

    async def set(
        self,
        artifact_id: str,
        summary: str,
        focus: str | None = None,
        ttl_seconds: int = 3600,
    ) -> None:
        key = self._cache_key(artifact_id, focus)
        await self.backend.set(key, summary, ttl_seconds)

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0.0,
        }
```

### Testing Strategy
- Test `InMemoryCache` get/set operations
- Test `SummaryCache` cache key generation (same artifact + different focus = different keys)
- Test cache hit/miss tracking
- Test stats calculation

### Dependencies
None - standalone component.

### Risk Factors
- In-memory cache doesn't survive restarts (acceptable for now)
- No cache eviction policy (memory could grow unbounded in long sessions)

---

## Milestone 6: LLM-Driven Context Planner

### Goal
Implement the core LLM-driven context planning. A single LLM call receives all artifacts (as previews), the user's query, the context goal, and token budget. It returns a prioritized plan specifying how to handle each artifact.

**Critical**: Include `should_use_llm_planning()` decision function to avoid unnecessary LLM calls when simple inclusion is sufficient.

### Success Criteria
- `should_use_llm_planning()` function that decides when LLM planning is worth the overhead
- `ArtifactProcessor` class with `plan_context` method
- LLM prompt that produces `ArtifactPlan` structured output
- `execute_plan` method that applies strategies (include_full, summarize, exclude)
- Integration with `SummaryCache` for summarization caching
- Uses `ModelConfig` for model selection

### Key Changes

**New file: `reasoning_api/reasoning_api/artifact_processor.py`**

```python
import json
import litellm
from litellm import token_counter

from .artifacts import Artifact, ArtifactPlan, ArtifactDecision, ProcessingStrategy
from .summary_cache import SummaryCache
from .model_config import ModelConfig
from .context_manager import ContextGoal, ContextUtilization
from .utils import preview


def should_use_llm_planning(
    artifacts: list[Artifact],
    conversation_history: list[dict],
    model_config: ModelConfig,
    utilization_strategy: ContextUtilization,
) -> bool:
    """
    Decide whether LLM planning is worth the overhead.

    Returns True when intelligent planning adds value:
    - Near or over token limit
    - Using expensive primary model (worth optimizing input)
    - Has large artifacts that may need summarization

    Returns False when simple inclusion is fine:
    - No artifacts to plan
    - Small total size
    - Everything fits easily with buffer
    """
    # Quick exit: nothing to plan
    if not artifacts:
        return False

    # Calculate total artifact size
    total_artifact_size = sum(a.size_chars for a in artifacts)

    # Quick exit: small enough to just include all
    if total_artifact_size < 5_000:
        return False

    # Get model info
    try:
        model_info = model_config.get_primary_info()
        max_tokens = model_info.get('max_input_tokens', 128_000)
    except Exception:
        max_tokens = 128_000  # Conservative default

    # Estimate current usage
    try:
        conv_tokens = token_counter(
            model=model_config.primary,
            messages=conversation_history,
        )
    except Exception:
        conv_tokens = sum(len(str(m.get('content', ''))) // 4 for m in conversation_history)

    artifact_tokens_est = total_artifact_size // 4
    total_est = conv_tokens + artifact_tokens_est

    # Calculate available based on utilization strategy
    utilization_pct = {
        ContextUtilization.LOW: 0.33,
        ContextUtilization.MEDIUM: 0.66,
        ContextUtilization.FULL: 1.0,
    }[utilization_strategy]
    available = int(max_tokens * utilization_pct)

    # Decision factors
    fits_easily = total_est < available * 0.7  # 30% buffer
    is_expensive = model_config.is_expensive_primary()
    has_large_artifacts = any(a.size_chars > 10_000 for a in artifacts)

    # Decision matrix:
    # - If fits easily AND cheap model AND no large artifacts → skip planning
    # - Otherwise → use planning
    if fits_easily and not is_expensive and not has_large_artifacts:
        return False

    return True


CONTEXT_PLANNER_PROMPT = """You are a context planner. Given artifacts and a user query, decide:
1. Which artifacts are relevant to answering the query
2. Priority order (most important first)
3. How to process each: include_full, summarize, rag_extract, or exclude

Consider:
- Token budget: {available_tokens} tokens available
- Goal: {goal_description}
- Relevance to the specific user query

Processing strategies:
- include_full: Small artifacts (<2000 chars) that are directly relevant
- summarize: Large artifacts that need key information extracted (provide summarization_focus)
- rag_extract: Large artifacts where specific facts are needed (provide extraction_query)
- exclude: Not relevant to the query

Return JSON matching this schema:
{schema}

Return artifacts in priority order. The executor will include artifacts until the token budget is exhausted."""


class ArtifactProcessor:
    """Uses LLM to plan and execute context assembly."""

    def __init__(
        self,
        summary_cache: SummaryCache,
        model_config: ModelConfig,
    ):
        self.summary_cache = summary_cache
        self.model_config = model_config

    async def plan_context(
        self,
        artifacts: list[Artifact],
        user_query: str,
        goal: ContextGoal,
        available_tokens: int,
    ) -> ArtifactPlan:
        """Get LLM's plan for handling each artifact."""

        if not artifacts:
            return ArtifactPlan(artifacts=[], reasoning="No artifacts to process")

        # Build lightweight previews for LLM
        artifact_previews = [
            {
                "id": a.artifact_id,
                "type": a.artifact_type.value,
                "source": a.source,
                "size_chars": a.size_chars,
                "preview": preview(a.content, max_str_len=500, max_items=5),
            }
            for a in artifacts
        ]

        goal_desc = (
            "deciding next reasoning step (be selective, focus on what's needed for planning)"
            if goal == ContextGoal.PLANNING
            else "generating final answer (include relevant details for comprehensive response)"
        )

        response = await litellm.acompletion(
            model=self.model_config.context_planner,  # Uses cheap model
            messages=[
                {
                    "role": "system",
                    "content": CONTEXT_PLANNER_PROMPT.format(
                        available_tokens=available_tokens,
                        goal_description=goal_desc,
                        schema=ArtifactPlan.model_json_schema(),
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps({
                        "user_query": user_query,
                        "artifacts": artifact_previews,
                    }),
                },
            ],
            response_format={"type": "json_object"},
        )

        return ArtifactPlan(**json.loads(response.choices[0].message.content))

    async def execute_plan(
        self,
        plan: ArtifactPlan,
        artifacts: dict[str, Artifact],
        token_budget: int,
    ) -> list[tuple[Artifact, str]]:
        """
        Execute the context plan, returning processed artifacts.

        Returns list of (artifact, processed_content) tuples in priority order,
        stopping when token budget would be exceeded.
        """
        results = []
        tokens_used = 0

        for decision in plan.artifacts:
            if decision.strategy == ProcessingStrategy.EXCLUDE:
                continue

            artifact = artifacts.get(decision.artifact_id)
            if not artifact:
                continue

            # Get or generate processed content
            processed = await self._apply_strategy(artifact, decision)

            # Estimate tokens (rough: 4 chars per token)
            est_tokens = len(processed) // 4
            if tokens_used + est_tokens > token_budget:
                break

            results.append((artifact, processed))
            tokens_used += est_tokens

        return results

    async def _apply_strategy(
        self,
        artifact: Artifact,
        decision: ArtifactDecision,
    ) -> str:
        """Apply the processing strategy to an artifact."""

        if decision.strategy == ProcessingStrategy.INCLUDE_FULL:
            return self._format_full(artifact)

        if decision.strategy == ProcessingStrategy.SUMMARIZE:
            # Check cache first
            cached = await self.summary_cache.get(
                artifact.artifact_id,
                decision.summarization_focus,
            )
            if cached:
                return cached

            # Generate summary
            summary = await self._summarize(artifact, decision.summarization_focus)

            # Cache it
            await self.summary_cache.set(
                artifact.artifact_id,
                summary,
                decision.summarization_focus,
            )
            return summary

        if decision.strategy == ProcessingStrategy.RAG_EXTRACT:
            # For now, fall back to summarize with extraction_query as focus
            # Full RAG implementation is future work
            return await self._apply_strategy(
                artifact,
                ArtifactDecision(
                    artifact_id=decision.artifact_id,
                    strategy=ProcessingStrategy.SUMMARIZE,
                    priority=decision.priority,
                    relevance_rationale=decision.relevance_rationale,
                    summarization_focus=decision.extraction_query,
                ),
            )

        return ""

    def _format_full(self, artifact: Artifact) -> str:
        """Format artifact for full inclusion."""
        if isinstance(artifact.content, (dict, list)):
            content = json.dumps(artifact.content, indent=2, ensure_ascii=False)
        else:
            content = str(artifact.content)
        return f"[{artifact.source}]\n{content}"

    async def _summarize(self, artifact: Artifact, focus: str | None) -> str:
        """Use LLM to summarize an artifact."""
        content = artifact.content
        if isinstance(content, (dict, list)):
            content = json.dumps(content, indent=2, ensure_ascii=False)

        focus_instruction = f"\n\nFocus on: {focus}" if focus else ""

        response = await litellm.acompletion(
            model=self.model_config.summarizer,  # Uses cheap model
            messages=[
                {
                    "role": "system",
                    "content": f"Summarize the following content concisely, preserving key facts and findings.{focus_instruction}",
                },
                {"role": "user", "content": str(content)},
            ],
        )

        return f"[{artifact.source}] (summarized)\n{response.choices[0].message.content}"
```

### Testing Strategy
- Test `should_use_llm_planning()`:
  - Returns False for empty artifacts
  - Returns False for small total size (<5K chars)
  - Returns False when everything fits easily with cheap model
  - Returns True when near/over limit
  - Returns True for expensive models
  - Returns True when large artifacts present
- Test `plan_context` with various artifact combinations
- Test `execute_plan` respects token budget and stops appropriately
- Test `_apply_strategy` for each strategy type
- Test cache integration (hit vs miss paths)
- Mock LLM responses for deterministic testing

### Dependencies
- Milestone 1 (Artifact models)
- Milestone 4 (ModelConfig)
- Milestone 5 (SummaryCache)

### Risk Factors
- LLM may not always produce valid JSON (need error handling)
- Token estimation is rough (4 chars/token) - may need refinement
- RAG_EXTRACT falls back to summarize for now (full RAG is future work)

---

## Milestone 7: Integrate ArtifactProcessor with ContextManager

### Goal
Wire the `ArtifactProcessor` into `ContextManager` so that reasoning context is built using intelligent LLM-driven planning when appropriate, with fallback to simple inclusion otherwise.

### Success Criteria
- `ContextManager` accepts `ModelConfig` and optional `ArtifactProcessor` dependency
- Uses `should_use_llm_planning()` to decide approach
- When LLM planning is used, delegates to `ArtifactProcessor`
- Falls back to current behavior when LLM planning not needed
- Metadata includes artifact processing stats (cache hits, planning used, etc.)

### Key Changes

**Update `reasoning_api/reasoning_api/context_manager.py`:**

```python
class ContextManager:
    def __init__(
        self,
        context_utilization: ContextUtilization = ContextUtilization.FULL,
        model_config: ModelConfig | None = None,
        artifact_processor: ArtifactProcessor | None = None,
    ):
        self.context_utilization = context_utilization
        self.model_config = model_config
        self.artifact_processor = artifact_processor

    async def __call__(
        self,
        model_name: str,
        context: Context,
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """
        Manage context for the given model.

        Now async to support LLM-driven artifact processing.
        """
        # Create model config if not provided
        if self.model_config is None:
            self.model_config = ModelConfig.from_request(model_name)

        if context.step_records or context.goal or context.documents:
            return await self._build_reasoning_context(model_name, context)

        return self._apply_token_limits(model_name, context)

    async def _build_reasoning_context(
        self,
        model_name: str,
        context: Context,
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Build reasoning context with optional LLM-driven processing."""

        # Gather all artifacts (tool results + documents)
        artifacts = extract_artifacts_from_steps(context.step_records)
        artifacts.extend(context.documents)

        # Decide whether to use LLM planning
        use_llm_planning = (
            self.artifact_processor is not None
            and should_use_llm_planning(
                artifacts=artifacts,
                conversation_history=context.conversation_history,
                model_config=self.model_config,
                utilization_strategy=self.context_utilization,
            )
        )

        if use_llm_planning:
            return await self._build_with_llm_planning(context, artifacts)

        # Fallback to simple inclusion
        return self._build_reasoning_context_simple(model_name, context)

    async def _build_with_llm_planning(
        self,
        context: Context,
        artifacts: list[Artifact],
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Build context using LLM-driven artifact processing."""

        # Calculate available tokens for artifacts
        max_tokens = self._get_max_tokens(self.model_config.primary)
        reserved = self._estimate_base_tokens(context)
        available_for_artifacts = max_tokens - reserved - 500  # buffer

        user_query = self._extract_user_query(context)
        goal = context.goal or ContextGoal.SYNTHESIS

        # Get LLM's plan
        plan = await self.artifact_processor.plan_context(
            artifacts=artifacts,
            user_query=user_query,
            goal=goal,
            available_tokens=available_for_artifacts,
        )

        # Execute the plan
        artifact_map = {a.artifact_id: a for a in artifacts}
        processed = await self.artifact_processor.execute_plan(
            plan=plan,
            artifacts=artifact_map,
            token_budget=available_for_artifacts,
        )

        # Assemble final context
        return self._assemble_with_processed_artifacts(context, processed, plan)

    def _build_reasoning_context_simple(
        self,
        model_name: str,
        context: Context,
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Build context with simple inclusion (no LLM planning)."""
        # This is the existing _build_reasoning_context logic
        # Include all artifacts directly using preview() for large ones
        ...
```

**Note**: The `__call__` method signature changes from sync to async since artifact processing involves LLM calls. This is a breaking change but acceptable per guidelines.

### Testing Strategy
- Test with artifact_processor=None (always uses simple approach)
- Test with artifact_processor but should_use_llm_planning returns False
- Test with artifact_processor and should_use_llm_planning returns True
- Test metadata includes `llm_planning_used` flag
- Test metadata includes cache stats when LLM planning used
- Integration test: full reasoning flow with LLM-driven context
- Test error handling when LLM planning fails (should fall back to simple)

### Dependencies
- Milestone 4 (ModelConfig)
- Milestone 6 (ArtifactProcessor with should_use_llm_planning)
- Milestone 3 (Context with documents)

### Risk Factors
- `__call__` becoming async is a breaking change - all callers need updates
- Need to handle case where LLM planning fails (fall back to simple)

---

## Milestone 8: Fix Greedy Fit Bug

### Goal
Replace the current "greedy fit" algorithm that skips large messages (creating incoherent context) with a smarter approach that maintains conversation coherence.

### Success Criteria
- Never skip a message and continue with older ones
- When hitting token limit, stop (don't skip and continue)
- Conversation context remains coherent
- Clear metadata when messages are excluded

### Key Changes

The current bug (documented in `context_manager.py:181-189`):

```python
# BUG: "greedy fit" creates incoherent conversation history.
# Current behavior: skips messages that don't fit, continues including older ones.
# Example: [msg1, msg2, msg3_huge, msg4] with msg3 too large → includes [msg1, msg2, msg4]
# Problem: msg4 may reference msg3's content, but msg3 is missing
```

**Fix approach - stop at first exclusion:**

```python
def _apply_token_limits(
    self,
    model_name: str,
    context: Context,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Apply token limits while maintaining conversation coherence."""

    # ... existing setup code ...

    # Work backwards through non-system messages
    # STOP at first message that doesn't fit (don't skip and continue)
    for msg in reversed(non_system_messages):
        msg_tokens = token_counter(model=model_name, messages=[msg])

        if total_tokens_used + msg_tokens > max_input_tokens:
            # STOP HERE - don't skip and continue
            # This maintains conversation coherence
            messages_excluded = len(non_system_messages) - messages_included
            break

        # Insert after system messages to maintain order
        final_messages.insert(len(system_messages), msg)
        total_tokens_used += msg_tokens
        messages_included += 1

    # ... rest of method ...
```

**Alternative enhancement (future)**: Instead of just stopping, summarize the excluded messages into a "context summary" message. This preserves some information while staying within limits.

### Testing Strategy
- Test that skipping never happens (stop at first exclusion)
- Test with conversation where middle message is huge
- Test metadata correctly reports excluded count
- Test conversation coherence (included messages are contiguous from end)

### Dependencies
None - fixes existing code.

### Risk Factors
- May exclude more messages than current approach (tradeoff for coherence)
- Users may need to adjust conversation length expectations

---

## Milestone 9: Wire Everything Together

### Goal
Update executors (PassthroughExecutor, ReasoningAgent) and dependency injection to use the enhanced ContextManager with artifact processing, model configuration, and document support.

### Success Criteria
- Both executors accept and pass through documents
- `ModelConfig` created from request and settings
- `ArtifactProcessor` instantiated with proper dependencies
- End-to-end test: request with documents → intelligent context → response
- Cache statistics exposed in response metadata

### Key Changes

**Update dependency injection in `api/dependencies.py`:**

```python
from reasoning_api.model_config import ModelConfig
from reasoning_api.summary_cache import SummaryCache, InMemoryCache
from reasoning_api.artifact_processor import ArtifactProcessor

def get_model_config(request: OpenAIChatRequest) -> ModelConfig:
    """Create ModelConfig from request with environment defaults."""
    return ModelConfig.from_request(request.model)

def get_artifact_processor(
    model_config: ModelConfig = Depends(get_model_config),
) -> ArtifactProcessor:
    """Get ArtifactProcessor with cache backend."""
    cache_backend = InMemoryCache()  # Future: Redis
    summary_cache = SummaryCache(cache_backend)
    return ArtifactProcessor(
        summary_cache=summary_cache,
        model_config=model_config,
    )

def get_context_manager(
    request: Request,
    model_config: ModelConfig = Depends(get_model_config),
    artifact_processor: ArtifactProcessor = Depends(get_artifact_processor),
) -> ContextManager:
    """Get ContextManager with all dependencies."""
    header_value = request.headers.get(CONTEXT_UTILIZATION_HEADER)
    utilization = parse_context_utilization_header(header_value)
    return ContextManager(
        context_utilization=utilization,
        model_config=model_config,
        artifact_processor=artifact_processor,
    )
```

**Update executor initialization to handle documents:**

```python
# In request handling (main.py or similar)
documents = []
if request.documents:
    documents = documents_to_artifacts(request.documents)

executor = ReasoningAgent(
    tools=tools,
    prompt_manager=prompt_manager,
    context_manager=context_manager,
    documents=documents,
)
```

**Update response metadata to include context stats:**

```python
# In executor, after context building
context_metadata = {
    "llm_planning_used": True,  # or False
    "artifacts_processed": 5,
    "cache_stats": artifact_processor.summary_cache.stats(),
}
```

### Testing Strategy
- Integration test: request with documents → response references document content
- Integration test: reasoning with tool results → intelligent summarization when appropriate
- Integration test: small request → skips LLM planning (check metadata)
- Integration test: large request with expensive model → uses LLM planning
- Test cache stats appear in response metadata
- Test passthrough executor with documents
- Test reasoning agent with documents

### Dependencies
- All previous milestones

### Risk Factors
- Breaking changes to executor signatures
- Need to update all executor instantiation sites
- Async ContextManager requires updating all callers

---

## Implementation Notes

### Breaking Changes (Acceptable)

1. `ContextManager.__call__` becomes async (artifact processing involves LLM calls)
2. Executor constructors gain new parameters (documents, model_config, etc.)
3. `Context` model gains new fields

### Files Changed Summary

| File | Changes |
|------|---------|
| `reasoning_api/artifacts.py` | NEW - Artifact, ArtifactDecision, ArtifactPlan models |
| `reasoning_api/model_config.py` | NEW - ModelConfig for model selection |
| `reasoning_api/summary_cache.py` | NEW - SummaryCache and backends |
| `reasoning_api/artifact_processor.py` | NEW - LLM-driven context planning + should_use_llm_planning |
| `reasoning_api/config.py` | Add CONTEXT_PLANNER_MODEL, SUMMARIZER_MODEL settings |
| `reasoning_api/openai_protocol.py` | Add DocumentAttachment, update OpenAIChatRequest |
| `reasoning_api/context_manager.py` | Update Context model, integrate ArtifactProcessor, fix greedy fit |
| `api/dependencies.py` | Add ModelConfig, ArtifactProcessor dependencies |
| `api/executors/*.py` | Accept documents, use updated ContextManager (now async) |

### Future Work (Out of Scope)

1. **Redis cache backend**: Production-ready caching with TTL
2. **Full RAG implementation**: Vector embeddings and semantic search for RAG_EXTRACT strategy
3. **Document storage**: Persistent document storage with `document_id` lookup
4. **Conversation unit grouping**: Group related messages (user + assistant pairs) before applying limits
5. **Summarize-to-fit**: Instead of excluding, summarize oversized messages to fit
6. **Per-request model overrides**: Allow `model_config_override` in request body

### Testing Guidelines

- Mock LLM calls in unit tests for determinism
- Integration tests should use real LLM calls (marked appropriately)
- Test cache hit/miss paths explicitly
- Test error handling for malformed LLM responses
- Test with various artifact sizes (small, medium, huge)
- Test `should_use_llm_planning()` decision boundaries

---

## Decision Matrix: When LLM Planning Is Used

| Condition | LLM Planning? | Rationale |
|-----------|---------------|-----------|
| No artifacts | No | Nothing to plan |
| Total artifact size < 5K chars | No | Just include everything |
| Everything fits with 30% buffer + cheap model + no large artifacts | No | Simple inclusion is fine |
| Near or over token limit | Yes | Need to be selective |
| Expensive primary model (o1, opus tier) | Yes | Minimize input tokens to save cost |
| Any artifact > 10K chars | Yes | May need summarization |

---

## Questions to Resolve

Before implementing, clarify:

1. **Cost threshold**: Is `0.00001` per input token the right threshold for "expensive"? (Currently catches o1, opus)

2. **Cache TTL**: What's the appropriate TTL for summaries? Session-scoped (1 hour)? Longer for documents?

3. **Token estimation**: The rough 4 chars/token estimate may be inaccurate. Should we use `litellm.token_counter` for precision (at cost of performance)?

4. **RAG priority**: Is the RAG_EXTRACT → SUMMARIZE fallback acceptable, or should RAG be fully implemented in this plan?

5. **Artifact size thresholds**: Are the thresholds (5K total, 10K per artifact) reasonable starting points?
