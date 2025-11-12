## execute_database_query

### Description
Execute a SQL query against the database with support for parameterized queries, transactions, and result formatting. Returns query results as structured data with metadata about execution time and row counts.

### Parameters
#### Required
- `query` (string): SQL query to execute (supports parameterized queries with $1, $2, etc.)

#### Optional
- `parameters` (list[Any] - Default: `[]`): Optional list of parameters to bind to the query
- `timeout_seconds` (number - Default: `30`): Maximum execution time before query is cancelled
- `return_format` (string - Default: `"dict"`): Format for results: 'dict', 'tuple', or 'dataframe'
- `use_transaction` (boolean - Default: `False`): Whether to wrap query in a transaction

---

## filesystem_operation

### Description
Perform various filesystem operations including reading, writing, moving, copying, and deleting files. Supports both text and binary modes with automatic encoding detection and path validation.

### Parameters
#### Required
- `operation` (string): Operation type: 'read', 'write', 'move', 'copy', 'delete', 'stat'
- `path` (string): Target file or directory path (supports absolute and relative paths)

#### Optional
- `content` (str | bytes | None): Content to write (required for 'write' operation)
- `destination` (str | None): Destination path for 'move' or 'copy' operations
- `encoding` (str | None - Default: `"utf-8"`): Text encoding for read/write operations (default: 'utf-8')
- `create_dirs` (boolean - Default: `True`): Automatically create parent directories if they don't exist
- `follow_symlinks` (boolean - Default: `True`): Whether to follow symbolic links during operations

---

## http_api_request

### Description
Make HTTP API requests with full control over headers, authentication, request body, timeouts, and retry behavior. Supports all HTTP methods and returns structured response data with status codes and headers.

### Parameters
#### Required
- `url` (string): Target URL (must be valid HTTP/HTTPS endpoint)

#### Optional
- `method` (string - Default: `"GET"`): HTTP method: GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS
- `headers` (dict[str, str] - Default: `{}`): HTTP headers as key-value pairs (e.g., {'Authorization': 'Bearer token'})
- `body` (dict[str, Any] | str | bytes | None): Request body (automatically serialized based on Content-Type header)
- `query_params` (dict[str, str | int | bool] - Default: `{}`): URL query parameters to append to the URL
- `timeout_seconds` (number - Default: `30.0`): Request timeout in seconds (applies to both connect and read)
- `retry_config` (dict[str, int] - Default: `{'max_attempts': 3, 'backoff_factor': 2}`): Retry configuration with keys: 'max_attempts', 'backoff_factor', 'retry_on_status' (comma-separated status codes)
- `verify_ssl` (boolean - Default: `True`): Whether to verify SSL certificates (disable for self-signed certs)
- `follow_redirects` (boolean - Default: `True`): Automatically follow HTTP redirects (3xx status codes)

---

## transform_data

### Description
Transform and manipulate structured data with support for filtering, mapping, aggregation, and joins. Works with lists, dicts, and nested structures to produce clean, formatted output.

### Parameters
#### Required
- `data` (list[dict[str, Any]] | dict[str, Any]): Input data to transform (supports both list of objects and single object)
- `operations` (list[dict[str, str | int | float]]): Ordered list of transformation operations to apply. Each operation has 'type' and operation-specific parameters.

#### Optional
- `output_format` (string - Default: `"json"`): Output format: 'json', 'csv', 'xml', 'yaml', or 'pretty'
- `filter_nulls` (boolean - Default: `False`): Remove null/None values from output
- `sort_keys` (boolean - Default: `False`): Sort object keys alphabetically in output

---

## system_health_check

### Description
Perform a comprehensive health check of all system components including database connectivity, external API availability, disk space, memory usage, and service status. Returns detailed diagnostics and recommendations.

### Parameters
No parameters required.