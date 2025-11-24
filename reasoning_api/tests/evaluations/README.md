# Evaluations

This directory contains evaluations for the ReasoningAgent using the [flex-evals](https://github.com/shane-kercheval/flex-evals) framework.

## Overview

These evaluations test the real reasoning agent with actual LLM API calls (via LiteLLM proxy) to ensure it works correctly most of the time despite LLM variability. Unlike unit tests that mock responses, evaluations test the full system with real LLM calls and use statistical thresholds to account for variability.

**Key Features**:
- Tests flow through LiteLLM proxy (same path as production)
- Uses separate virtual key for evaluation tracking (`LITELLM_EVAL_KEY`)
- All requests traced in Phoenix for observability
- Statistical thresholds account for LLM non-determinism

## Prerequisites

Before running evaluations:

1. **Setup environment**:
   ```bash
   # Copy environment file
   cp .env.dev.example .env

   # Set required variables in .env:
   # - OPENAI_API_KEY (real OpenAI key for LiteLLM)
   # - LITELLM_MASTER_KEY
   # - LITELLM_POSTGRES_PASSWORD
   # - PHOENIX_POSTGRES_PASSWORD
   ```

2. **Start services and setup LiteLLM**:
   ```bash
   # Start all services (including LiteLLM)
   make docker_up

   # Generate virtual keys
   make litellm_setup

   # Copy keys to .env:
   # LITELLM_API_KEY=sk-...
   # LITELLM_EVAL_KEY=sk-...  # Separate key for evaluation tracking
   ```

3. **Verify setup**:
   ```bash
   # Check services are running
   docker compose ps

   # Verify LiteLLM is healthy
   curl http://localhost:4000/health/readiness
   ```

## Running Evaluations

### Run all evaluations:
```bash
# With services running:
make evaluations

# Or directly:
python tests/evaluations/run_evaluations.py
```

### Run specific evaluation:
```bash
python tests/evaluations/run_evaluations.py -k test_weather_tool_usage
```

### Run with verbose output:
```bash
python tests/evaluations/run_evaluations.py -v -s
```

### Direct pytest usage:
```bash
pytest tests/evaluations/test_eval_reasoning_agent.py -v
```

**Note**: All evaluations use `LITELLM_EVAL_KEY` for separate usage tracking in the LiteLLM dashboard.

## Evaluation Types

### 1. Single Tool Usage
- `test_weather_tool_usage`: Tests weather tool calls
- `test_calculator_tool_usage`: Tests calculator tool calls  
- `test_search_tool_usage`: Tests search tool calls

### 2. Multi-Tool Usage
- `test_multiple_tool_usage`: Tests using multiple tools in one request

### 3. No Tools Needed
- `test_no_tool_needed`: Tests answering questions without tools

### 4. Streaming
- `test_streaming_with_tools`: Tests streaming responses with tools

### 5. Custom Validation
- `test_weather_with_custom_check`: Uses custom validation logic

## Configuration

Each evaluation is configured with:
- **test_cases**: List of inputs to test
- **checks**: Validation criteria (e.g., ContainsCheck for required phrases)
- **samples**: Number of times to run each test case
- **success_threshold**: Minimum success rate (e.g., 0.8 = 80%)

## Example Results

```
PASSED test_weather_tool_usage[TestCase(input='What's the weather in Tokyo? Use the get_weather tool.')] - 4/5 (80.0%) ✓
PASSED test_calculator_tool_usage[TestCase(input='Calculate 15 + 27 using the calculate tool.')] - 5/5 (100.0%) ✓
FAILED test_multiple_tool_usage[TestCase(input='What's the weather in Tokyo and calculate 25 + 15?')] - 2/5 (40.0%) ✗
```

## Understanding Results

- ✓ **PASSED**: Success rate met or exceeded the threshold
- ✗ **FAILED**: Success rate below the threshold
- Individual run results show which specific samples passed/failed

## Customizing Evaluations

### Adding New Test Cases
```python
@evaluate(
    test_cases=[
        TestCase(input="Your test input here"),
        TestCase(input="Another test input", metadata={"key": "value"}),
    ],
    checks=[
        ContainsCheck(
            text="$.output.value",
            phrases=["expected", "phrases"],
            case_sensitive=False,
        ),
    ],
    samples=5,
    success_threshold=0.8,
)
async def test_my_evaluation(test_case: TestCase, reasoning_agent_with_tools: ReasoningAgent) -> str:
    # Your test logic here
    pass
```

### Custom Validation
```python
def custom_check(test_case: TestCase, output: Any) -> bool:
    # Your custom validation logic
    return True  # or False

@evaluate(
    test_cases=[...],
    checks=[custom_check],
    samples=3,
    success_threshold=0.8,
)
async def test_with_custom_validation(...):
    # Test implementation
    pass
```

## Best Practices

1. **Use appropriate thresholds**: 
   - Simple factual questions: 0.9+ (90%+)
   - Tool usage: 0.8+ (80%+) 
   - Complex multi-tool: 0.7+ (70%+)

2. **Choose sample sizes wisely**:
   - Quick feedback: 3-5 samples
   - Reliable results: 10+ samples

3. **Use lower temperature**: 0.3 or lower for more consistent results

4. **Test different scenarios**:
   - Single tool usage
   - Multi-tool usage  
   - No tools needed
   - Edge cases and error handling

## Integration with CI/CD

Add to your CI pipeline:

```yaml
- name: Run Evaluations
  run: |
    # Start LiteLLM stack (testcontainers handles this automatically)
    export OPENAI_API_KEY=${{ secrets.OPENAI_API_KEY }}
    export LITELLM_API_KEY=${{ secrets.LITELLM_EVAL_KEY }}
    python tests/evaluations/run_evaluations.py
  continue-on-error: true  # Don't fail CI if evals fail
```

**CI/CD Notes**:
- Evaluations use testcontainers to automatically start LiteLLM stack
- No manual Docker setup required in CI
- Virtual keys can be pre-generated and stored as secrets
- Consider using `continue-on-error: true` since LLM evaluations can be flaky
- Track trends rather than blocking deployments on evaluation failures

## Observability and Tracking

### LiteLLM Dashboard

View evaluation usage and costs in the LiteLLM dashboard:

1. **Open dashboard**: http://localhost:4000
2. **Filter by virtual key**: Select `LITELLM_EVAL_KEY`
3. **View metrics**:
   - Total evaluation requests
   - Token usage and costs
   - Model distribution
   - Error rates

### Phoenix Traces

Analyze evaluation traces in Phoenix:

1. **Open Phoenix**: http://localhost:6006
2. **Filter by component**: `reasoning-agent` or `evaluation`
3. **Analyze patterns**:
   - Latency distribution
   - Tool usage patterns
   - Reasoning step quality
   - Error patterns

### Separate Tracking

Evaluations use a dedicated virtual key (`LITELLM_EVAL_KEY`) to:
- Separate evaluation costs from development/production
- Track evaluation usage independently
- Analyze evaluation-specific patterns
- Budget evaluation costs separately (if needed)

This enables clear separation between:
- `LITELLM_API_KEY`: Development and production traffic
- `LITELLM_EVAL_KEY`: Evaluation and testing traffic