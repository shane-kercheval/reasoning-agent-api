# Reasoning Agent Web Client

A beautiful MonsterUI-powered web interface for the Reasoning Agent API. Features streaming conversations, collapsible reasoning steps, and power user controls.

## Features

- **🎨 Beautiful UI**: Built with MonsterUI for modern, responsive design
- **💬 Real-time Chat**: Streaming conversations with the reasoning agent
- **🧠 Reasoning Steps**: Collapsible tree view of AI thinking process
- **⚙️ Power User Mode**: Advanced settings for system prompts and parameters
- **📱 Responsive**: Works on desktop, tablet, and mobile
- **🔒 Secure**: Token-based authentication with your API

## Quick Start

### Prerequisites

- Python 3.13+
- uv package manager
- Running Reasoning Agent API

### Local Development

1. **Install dependencies**:
   ```bash
   cd web-client
   uv sync
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

3. **Start the web client**:
   ```bash
   uv run python main.py
   ```

4. **Access the interface**:
   - Open http://localhost:8080 in your browser
   - Make sure your API is running on http://localhost:8000

## Configuration

### Environment Variables

- `REASONING_API_URL`: URL of your reasoning agent API
- `API_TOKEN`: Authentication token (must match API_TOKENS in your API)
- `WEB_CLIENT_PORT`: Port for web client (development only)

### Example Configuration

```bash
# Development
REASONING_API_URL=http://localhost:8000
API_TOKEN=token1
WEB_CLIENT_PORT=8080

# Production (Render)
REASONING_API_URL=https://your-api.onrender.com
API_TOKEN=token1
```

## Deployment

### Render Deployment

1. **Create new Web Service** on Render
2. **Connect your repository**
3. **Configure service**:
   - **Root Directory**: `web-client`
   - **Build Command**: `uv sync`
   - **Start Command**: `uv run python main.py`
4. **Set environment variables**:
   - `REASONING_API_URL`: URL of your deployed API service
   - `API_TOKEN`: Your authentication token

## Architecture

```
User Browser ←→ MonsterUI Web Client ←→ Reasoning Agent API
   (Port 80)        (Port 8080)           (Port 8000)
```

The web client acts as a frontend that:
- Serves HTML pages to browsers
- Makes HTTP requests to your reasoning agent API
- Handles real-time streaming and UI updates
- Manages user interactions and settings

## UI Components

### Chat Interface
- **Message bubbles**: Styled chat messages with timestamps
- **Streaming responses**: Real-time message updates
- **Multi-line support**: Proper formatting for long messages

### Reasoning Steps
- **Collapsible accordions**: Expandable reasoning process view
- **Tool execution details**: Shows MCP tool calls and results
- **Error handling**: Clear display of any issues

### Power User Settings
- **System prompt editor**: Custom system prompts
- **Parameter controls**: Temperature, max tokens, etc.
- **Stream toggle**: Enable/disable streaming

## Development

### Project Structure
```
web-client/
├── main.py              # Main FastHTML application
├── pyproject.toml       # Dependencies and configuration
├── .env.example         # Environment variables template
└── README.md           # This file
```

### Adding Features

The web client is built with:
- **FastHTML**: Python-based web framework
- **MonsterUI**: Beautiful UI components
- **HTMX**: Real-time frontend interactions

To add new features:
1. Add UI components using MonsterUI
2. Create FastHTML routes for functionality
3. Use HTMX for dynamic updates

## Troubleshooting

### Common Issues

**"Connection refused" errors**:
- Check that your reasoning agent API is running
- Verify the `REASONING_API_URL` in your `.env`

**Authentication errors**:
- Ensure `API_TOKEN` matches one in your API's `API_TOKENS`
- Check that `REQUIRE_AUTH=true` in your API if using authentication

**UI not updating**:
- Check browser console for JavaScript errors
- Verify HTMX is working properly

### Debug Mode

Set environment variable for more verbose output:
```bash
DEBUG=true uv run python main.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with the reasoning agent API
5. Submit a pull request
