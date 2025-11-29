"""Tests for PromptTemplate class."""

from pathlib import Path

import pytest

from tools_api.services.prompts.template import PromptTemplate


@pytest.mark.asyncio
async def test_basic_variable_substitution() -> None:
    """Test basic variable substitution with {{ name }}."""
    template = PromptTemplate(
        name="greeting",
        description="A greeting prompt",
        template="Hello, {{ name }}!",
        arguments=[{"name": "name", "required": True, "description": "Name to greet"}],
    )

    result = await template.render(name="Alice")
    assert result == "Hello, Alice!"


@pytest.mark.asyncio
async def test_conditional_blocks() -> None:
    """Test conditional blocks with {% if var %}...{% endif %} and optional args."""
    template = PromptTemplate(
        name="review",
        description="Code review prompt",
        template="Review the code.{% if focus %}\nFocus on: {{ focus }}{% endif %}",
        arguments=[
            {"name": "focus", "required": False, "description": "Focus areas"},
        ],
    )

    # With focus provided
    result_with = await template.render(focus="security")
    assert "Focus on: security" in result_with

    # Without focus - optional args auto-populated with None, so {% if focus %} is False
    result_without = await template.render()
    assert "Focus on:" not in result_without
    assert result_without == "Review the code."


@pytest.mark.asyncio
async def test_missing_required_argument_raises_error() -> None:
    """Test that missing required argument raises ValueError with argument and prompt name."""
    template = PromptTemplate(
        name="greeting",
        description="A greeting prompt",
        template="Hello, {{ name }}!",
        arguments=[{"name": "name", "required": True, "description": "Name to greet"}],
    )

    with pytest.raises(ValueError) as exc_info:
        await template.render()

    error_msg = str(exc_info.value)
    assert "name" in error_msg
    assert "greeting" in error_msg


@pytest.mark.asyncio
async def test_missing_required_argument_includes_source_path() -> None:
    """Test that missing required argument includes source path in error if provided."""
    template = PromptTemplate(
        name="greeting",
        description="A greeting prompt",
        template="Hello, {{ name }}!",
        arguments=[{"name": "name", "required": True, "description": "Name to greet"}],
        source_path=Path("/prompts/greeting.md"),
    )

    with pytest.raises(ValueError) as exc_info:
        await template.render()

    error_msg = str(exc_info.value)
    assert "/prompts/greeting.md" in error_msg


@pytest.mark.asyncio
async def test_optional_argument_provided_works() -> None:
    """Test that optional argument provided works correctly."""
    template = PromptTemplate(
        name="greeting",
        description="A greeting prompt",
        template="Hello{{ name }}!",
        arguments=[{"name": "name", "required": False, "description": "Name to greet"}],
    )

    result = await template.render(name=" Alice")
    assert result == "Hello Alice!"


@pytest.mark.asyncio
async def test_all_properties_return_correct_values() -> None:
    """Test that all properties return correct values."""
    template = PromptTemplate(
        name="test_prompt",
        description="Test description",
        template="Template content",
        arguments=[{"name": "arg1", "required": True, "description": "Argument 1"}],
        category="test_category",
        tags=["tag1", "tag2"],
        source_path=Path("/test/path.md"),
    )

    assert template.name == "test_prompt"
    assert template.description == "Test description"
    assert template.arguments == [
        {"name": "arg1", "required": True, "description": "Argument 1"},
    ]
    assert template.category == "test_category"
    assert template.tags == ["tag1", "tag2"]
    assert template.source_path == Path("/test/path.md")


@pytest.mark.asyncio
async def test_call_wrapper_returns_prompt_result() -> None:
    """Test __call__ wrapper returns PromptResult with correct structure."""
    template = PromptTemplate(
        name="test",
        description="Test",
        template="Hello, {{ name }}!",
        arguments=[{"name": "name", "required": True, "description": "Name"}],
    )

    result = await template(name="Alice")
    assert result.success is True
    assert result.content == "Hello, Alice!"
    assert result.error is None


@pytest.mark.asyncio
async def test_call_wrapper_handles_errors() -> None:
    """Test __call__ wrapper handles errors and returns PromptResult with error."""
    template = PromptTemplate(
        name="test",
        description="Test",
        template="Hello, {{ name }}!",
        arguments=[{"name": "name", "required": True, "description": "Name"}],
    )

    result = await template()  # Missing required argument
    assert result.success is False
    assert result.content == ""
    assert result.error is not None
    assert "name" in result.error


@pytest.mark.asyncio
async def test_empty_template_renders_to_empty_string() -> None:
    """Test that empty template renders to empty string."""
    template = PromptTemplate(
        name="empty",
        description="Empty prompt",
        template="",
    )

    result = await template.render()
    assert result == ""


@pytest.mark.asyncio
async def test_template_with_no_variables_renders_as_is() -> None:
    """Test that template with no variables renders as-is."""
    template = PromptTemplate(
        name="static",
        description="Static prompt",
        template="This is a static prompt with no variables.",
    )

    result = await template.render()
    assert result == "This is a static prompt with no variables."


@pytest.mark.asyncio
async def test_unicode_content_in_template_and_arguments() -> None:
    """Test unicode content in template and arguments."""
    template = PromptTemplate(
        name="unicode",
        description="Unicode test",
        template="Hello, {{ name }}! Your emoji is {{ emoji }}.",
        arguments=[
            {"name": "name", "required": True, "description": "Name"},
            {"name": "emoji", "required": True, "description": "Emoji"},
        ],
    )

    result = await template.render(name="Muller", emoji="\U0001F389")
    assert "Muller" in result
    assert "\U0001F389" in result


@pytest.mark.asyncio
async def test_undefined_variable_in_template_raises_error() -> None:
    """Test that undefined variable in template raises UndefinedError (fail fast)."""
    from jinja2.exceptions import UndefinedError

    template = PromptTemplate(
        name="test",
        description="Test",
        template="Value: {{ undefined_var }}",
    )

    with pytest.raises(UndefinedError):
        await template.render()


@pytest.mark.asyncio
async def test_default_values_for_optional_fields() -> None:
    """Test that optional constructor fields have correct defaults."""
    template = PromptTemplate(
        name="minimal",
        description="Minimal prompt",
        template="Hello",
    )

    assert template.arguments == []
    assert template.category is None
    assert template.tags == []
    assert template.source_path is None


@pytest.mark.asyncio
async def test_for_loop_in_template() -> None:
    """Test {% for %} loop in template."""
    template = PromptTemplate(
        name="list",
        description="List prompt",
        template="Items:{% for item in items %}\n- {{ item }}{% endfor %}",
        arguments=[{"name": "items", "required": True, "description": "List of items"}],
    )

    result = await template.render(items=["apple", "banana", "cherry"])
    assert "- apple" in result
    assert "- banana" in result
    assert "- cherry" in result


@pytest.mark.asyncio
async def test_nested_variables() -> None:
    """Test accessing nested object properties."""
    template = PromptTemplate(
        name="nested",
        description="Nested prompt",
        template="User: {{ user.name }}, Age: {{ user.age }}",
        arguments=[{"name": "user", "required": True, "description": "User object"}],
    )

    result = await template.render(user={"name": "Alice", "age": 30})
    assert "Alice" in result
    assert "30" in result


@pytest.mark.asyncio
async def test_multiple_required_arguments() -> None:
    """Test multiple required arguments."""
    template = PromptTemplate(
        name="multi",
        description="Multi-arg prompt",
        template="{{ greeting }}, {{ name }}! Welcome to {{ place }}.",
        arguments=[
            {"name": "greeting", "required": True, "description": "Greeting"},
            {"name": "name", "required": True, "description": "Name"},
            {"name": "place", "required": True, "description": "Place"},
        ],
    )

    # All present
    result = await template.render(greeting="Hello", name="Bob", place="Paris")
    assert result == "Hello, Bob! Welcome to Paris."

    # Missing one
    with pytest.raises(ValueError) as exc_info:
        await template.render(greeting="Hello", name="Bob")
    assert "place" in str(exc_info.value)


@pytest.mark.asyncio
async def test_unknown_argument_raises_error() -> None:
    """Test that passing unknown arguments raises ValueError (catches typos)."""
    template = PromptTemplate(
        name="greeting",
        description="A greeting prompt",
        template="Hello, {{ name }}!",
        arguments=[{"name": "name", "required": True, "description": "Name to greet"}],
    )

    with pytest.raises(ValueError) as exc_info:
        await template.render(name="Alice", nme="typo")  # 'nme' is a typo

    error_msg = str(exc_info.value)
    assert "nme" in error_msg
    assert "Unknown argument" in error_msg
    assert "greeting" in error_msg


@pytest.mark.asyncio
async def test_unknown_argument_includes_valid_args_in_error() -> None:
    """Test that unknown argument error includes list of valid arguments."""
    template = PromptTemplate(
        name="test",
        description="Test",
        template="{{ foo }} {{ bar }}",
        arguments=[
            {"name": "foo", "required": True, "description": "Foo"},
            {"name": "bar", "required": True, "description": "Bar"},
        ],
    )

    with pytest.raises(ValueError) as exc_info:
        await template.render(foo="a", bar="b", baz="typo")

    error_msg = str(exc_info.value)
    assert "foo" in error_msg
    assert "bar" in error_msg


@pytest.mark.asyncio
async def test_optional_arg_omitted_auto_populated_with_none() -> None:
    """Test that optional arguments omitted are auto-populated with None."""
    template = PromptTemplate(
        name="test",
        description="Test",
        template="Value: {% if opt %}{{ opt }}{% else %}default{% endif %}",
        arguments=[
            {"name": "opt", "required": False, "description": "Optional value"},
        ],
    )

    # Omitting optional arg should work (auto-populated with None)
    result = await template.render()
    assert result == "Value: default"

    # Providing optional arg should use provided value
    result_with = await template.render(opt="provided")
    assert result_with == "Value: provided"


@pytest.mark.asyncio
async def test_template_with_no_args_rejects_any_kwargs() -> None:
    """Test that templates with no defined args reject any passed kwargs."""
    template = PromptTemplate(
        name="static",
        description="Static prompt",
        template="Hello world",
    )

    with pytest.raises(ValueError) as exc_info:
        await template.render(unexpected="value")

    assert "Unknown argument" in str(exc_info.value)
