# Python Code Generation Guidelines:

**CODE STYLE & FORMATTING:**
- Follow PEP 8 style guide strictly
- Use 4 spaces for indentation (no tabs)
- Use single spaces 
- Use clear, descriptive variable and function names that convey purpose
- Use single quotes for internal/code strings ('dict_key', 'variable_value')
- Use double quotes for user-facing strings ("Error: Invalid input", "Welcome!")
- DO NOT place imports inside functions or methods; only place them at the top of the file

**TYPE ANNOTATIONS:**
- Include type hints for all function parameters and return values
- Use modern type hint syntax, for example:
  - `list[str]` instead of `List[str]`
  - `dict[str, int]` instead of `Dict[str, int]`
  - `int | None` instead of `Optional[int]`
  - `str | int` instead of `Union[str, int]`

**DOCUMENTATION:**
- Use docstrings when creating files, classes, and functions.
- Use docstrings in the format below when creating functions:
    ```python
    def function_name(param: str, value: int = 0) -> bool:
        """Brief one-line description.
        
        Longer description if needed, explaining the function's purpose,
        behavior, or important implementation details.
        
        Args:
            param:
	            Description of the parameter.
            value:
	            Description with default value noted.
            
        Returns:
            Description of return value and its type/structure.
            
        Raises:
            ValueError: When input validation fails.
            TypeError: When incorrect types are passed.
            
        Example:
            >>> result = function_name("test", 42)
            >>> print(result)
            True
        """
    ```
- Include usage examples for non-trivial functions
- Document complex algorithms or business logic

**CODE COMMENTS:**
- Add comments to code only when they clarify logic or business rules
- Skip comments that simply restate what the code obviously does
- Don't add meta-commentary about code changes

**TESTING GUIDELINES:**
- Use `pytest` for the test framework
- Follow naming convention: `test__<function_name>__<scenario>`
- Use descriptive assertion messages
- Write tests that verify behavior and business logic, not implementation details (i.e. what the code does (its observable behavior) rather than how it does it (the internal mechanics))
- Focus on what could realistically break or fail
- Test edge cases, error conditions, and user workflows
- Use descriptive test names that explain the scenario being tested
- Skip trivial or 'low value' tests (simple getters/setters, obvious assignments)

**OUTPUT FORMAT:**
- Provide a brief overview explaining what the code does
- Highlight any important design decisions or trade-offs
- Discuss any concerns or considerations with the implementation
- No commentary on routine code changes unless specifically noteworthy
