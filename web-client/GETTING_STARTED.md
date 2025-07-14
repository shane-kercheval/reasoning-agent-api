# Getting Started with the Web Client

This guide walks you through setting up and running the MonsterUI web client for the first time.

## Prerequisites

✅ **Reasoning Agent API running** on http://localhost:8000  
✅ **Python 3.13+** installed  
✅ **uv package manager** installed  

## Step-by-Step Setup

### 1. Install Dependencies

From the **web-client** directory:

```bash
cd web-client
uv sync
```

### 2. Configure Environment

```bash
# Copy the example configuration
cp .env.example .env

# Edit .env file (optional - defaults work for local development)
# REASONING_API_URL=http://localhost:8000
# API_TOKEN=token1
# WEB_CLIENT_PORT=8080
```

### 3. Start the Web Client

```bash
# Start the web client
uv run python main.py
```

You should see:
```
🚀 Starting Reasoning Agent Web Client on port 8080
📡 Connecting to API at: http://localhost:8000
🌐 Web interface will be available at: http://localhost:8080
```

### 4. Test the Interface

1. **Open your browser** to http://localhost:8080
2. **Type a message** like "Hello, can you help me?"
3. **Click Send** and watch the AI respond with reasoning steps!

## Troubleshooting

### Common Issues

**"Connection refused"**:
- ✅ Make sure your reasoning agent API is running on port 8000
- ✅ Run `make api` from the main project directory

**"Authentication failed"**:
- ✅ Check that your API has `API_TOKENS=token1,token2,token3` in its `.env`
- ✅ Verify `REQUIRE_AUTH=true` in your API configuration

**"Module not found"**:
- ✅ Run `uv sync` in the web-client directory
- ✅ Check that you're in the correct directory

**Port already in use**:
- ✅ Change `WEB_CLIENT_PORT=8081` in your `.env`
- ✅ Or kill the existing process: `lsof -ti :8080 | xargs kill`

### Debug Mode

For more verbose output:

```bash
# Set debug environment variable
DEBUG=true uv run python main.py
```

## Using with Makefile

From the **main project directory**, you can use:

```bash
# Start API (terminal 1)
make api

# Start web client (terminal 2) 
make web_client
```

## Features to Try

Once running, try these features:

1. **💬 Basic Chat**: Ask simple questions
2. **🧠 Reasoning Steps**: Watch the AI think through complex problems
3. **🔧 Tool Usage**: Ask about weather, web searches (if MCP servers configured)
4. **⚙️ Power User Mode**: Click the settings button to customize prompts
5. **📱 Mobile**: Try it on your phone/tablet

## What's Happening

When you send a message:

1. **Web Client** (port 8080) receives your message
2. **Web Client** sends HTTP request to **API** (port 8000)
3. **API** processes with reasoning and streams back events
4. **Web Client** displays chat and reasoning steps in real-time

## Next Steps

- **Deploy to Render**: See main README for deployment instructions
- **Customize UI**: Explore MonsterUI components in `main.py`
- **Add Features**: Extend the web client with new functionality
- **MCP Tools**: Configure additional MCP servers for more capabilities

Happy chatting! 🤖✨
