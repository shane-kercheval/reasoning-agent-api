# JSON Mode vs Structured Outputs: Key Differences

## Problem: Why `dict[str, Any]` Breaks Structured Outputs

When using `dict[str, Any]` in a Pydantic model, the generated JSON schema includes:
```json
{
  "properties": {
    "arguments": {
      "type": "object",
      "additionalProperties": true  // ← OpenAI rejects this!
    }
  }
}
```

**OpenAI's Structured Outputs Requirement:**
- ALL objects must have `"additionalProperties": false`
- ALL keys must be pre-defined in the schema
- This ensures deterministic, strictly-validated outputs

**Why This Fails:**
- `dict[str, Any]` needs arbitrary keys (unknown at schema definition time)
- Setting `additionalProperties: false` would make the dict unusable
- There's no way to make `dict[str, Any]` compatible with structured outputs

## Solution: Use JSON Mode Instead

### Structured Outputs (Pydantic Model)
```python
# ✅ Sends schema + field descriptions to OpenAI
response_format=EventsList

# What the model receives:
# - Complete JSON schema with all field types
# - All Field(description="...") text
# - Strict validation enforced by OpenAI
# - 100% guaranteed to match schema

# ❌ Limitations:
# - Cannot use dict[str, Any] (additionalProperties: true rejected)
# - All object keys must be pre-defined
# - Less flexible for dynamic data structures
```

### JSON Mode (Raw JSON Object)
```python
# ✅ Works with dict[str, Any]
response_format={"type": "json_object"}

# What the model receives:
# - NOTHING! No schema at all!
# - Only knows it must return valid JSON

# ✅ You MUST embed schema in system prompt:
system_prompt = f"""
Your response MUST conform to this schema:

```json
{ReasoningStep.schema_for_prompt()}
```
"""

# ⚠️ Validation:
# - OpenAI doesn't validate against schema
# - You validate manually: ReasoningStep.model_validate(json_response)
# - ~99% accuracy if prompt is clear (not 100% guaranteed)
```

## Key Takeaway

| Feature | Structured Outputs | JSON Mode |
|---------|-------------------|-----------|
| **Schema validation** | OpenAI enforces | Manual (client-side) |
| **Field descriptions** | Automatic (in schema) | Manual (in prompt) |
| **`dict[str, Any]` support** | ❌ No (`additionalProperties: true` rejected) | ✅ Yes |
| **Accuracy** | 100% guaranteed | ~99% (depends on prompt clarity) |
| **Use when** | Schema is fully static | Need flexible/dynamic structures |

## Implementation in This Codebase

1. **`ToolPrediction.arguments: dict[str, Any]`** requires JSON mode
2. **Schema embedded in prompts** via `{{reasoning_schema}}` placeholder
3. **Manual validation** after receiving response: `ReasoningStep.model_validate(json_response)`
4. **Field descriptions preserved** by including full schema in system prompt

## Example Comparison

### Works with Structured Outputs ✅
```python
class CalendarEvent(BaseModel):
    name: str  # Fixed type
    date: str  # Fixed type
    participants: list[str]  # Fixed array type

# Schema: All types known, no additionalProperties needed
```

### Requires JSON Mode ❌ (Structured Outputs)
```python
class ToolPrediction(BaseModel):
    tool_name: str
    arguments: dict[str, Any]  # ← Dynamic keys, arbitrary values!

# Schema: "additionalProperties": true generated
# OpenAI rejects: "must be false"
```

## Testing the Difference

```bash
# View the actual schema sent in prompts
uv run python << 'EOF'
from api.reasoning_models import ReasoningStep
print(ReasoningStep.schema_for_prompt())
EOF

# Check if Field descriptions are present in schema
# Look for: "description": "Exact name of the tool..."
```
