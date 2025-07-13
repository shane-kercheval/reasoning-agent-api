#!/usr/bin/env python3
"""
Mock MCP Server for Demo Purposes.

This creates a simple HTTP-based MCP server that provides demo tools
for the reasoning agent demonstration. It simulates real MCP tools
without requiring external services.

Usage:
    python demo_mcp_server.py --port 8001

The server provides these tools:
- web_search: Simulates web search capabilities
- weather_api: Simulates weather data retrieval
- calculator: Simulates mathematical calculations
- data_lookup: Simulates database/API lookups
"""

import json
import random
import time
import argparse
import uvicorn
from datetime import datetime
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Demo MCP Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock data for demonstrations
MOCK_DATA = {
    "populations": {
        "tokyo": 37400000,
        "new york city": 8400000,
        "new york": 8400000,
        "nyc": 8400000,
        "london": 9500000,
        "paris": 2200000,
        "los angeles": 3900000,
        "chicago": 2700000,
    },
    "weather": {
        "tokyo": {"temp": 22, "condition": "partly cloudy", "humidity": 65},
        "new york": {"temp": 18, "condition": "sunny", "humidity": 58},
        "london": {"temp": 15, "condition": "rainy", "humidity": 78},
        "paris": {"temp": 19, "condition": "overcast", "humidity": 72},
        "los angeles": {"temp": 25, "condition": "sunny", "humidity": 45},
    },
    "attractions": {
        "paris": [
            {"name": "Eiffel Tower", "rating": 4.6, "cost": "€25"},
            {"name": "Louvre Museum", "rating": 4.7, "cost": "€17"},
            {"name": "Notre-Dame Cathedral", "rating": 4.5, "cost": "Free"},
            {"name": "Arc de Triomphe", "rating": 4.4, "cost": "€13"},
            {"name": "Montmartre & Sacré-Cœur", "rating": 4.6, "cost": "Free"},
        ],
        "tokyo": [
            {"name": "Senso-ji Temple", "rating": 4.3, "cost": "Free"},
            {"name": "Tokyo Skytree", "rating": 4.2, "cost": "¥2100"},
            {"name": "Meiji Shrine", "rating": 4.4, "cost": "Free"},
            {"name": "Tsukiji Outer Market", "rating": 4.3, "cost": "Variable"},
        ],
    },
}

class MCPServer:
    """Mock MCP server implementation."""

    def __init__(self):
        self.tools = {
            "web_search": {
                "name": "web_search",
                "description": "Search the web for information on any topic",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "max_results": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            },
            "weather_api": {
                "name": "weather_api",
                "description": "Get current weather information for a city",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                    },
                    "required": ["city"],
                },
            },
            "calculator": {
                "name": "calculator",
                "description": "Perform mathematical calculations",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Mathematical expression"},
                        "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide", "evaluate"]},  # noqa: E501
                    },
                    "required": ["expression"],
                },
            },
            "data_lookup": {
                "name": "data_lookup",
                "description": "Look up factual data from various sources",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "enum": ["population", "geography", "economics", "attractions"]},  # noqa: E501
                        "query": {"type": "string", "description": "What to look up"},
                    },
                    "required": ["category", "query"],
                },
            },
        }

    def simulate_delay(self, min_ms: int = 100, max_ms: int = 800) -> None:
        """Simulate realistic API response time."""
        delay = random.uniform(min_ms, max_ms) / 1000
        time.sleep(delay)

    def web_search(self, query: str, max_results: int = 5) -> dict[str, Any]:
        """Simulate web search results."""
        self.simulate_delay(200, 1000)

        # Generate mock search results based on query
        query_lower = query.lower()
        results = []

        if "population" in query_lower:
            for city, pop in MOCK_DATA["populations"].items():
                if city in query_lower:
                    results.append({
                        "title": f"{city.title()} Population 2024",
                        "snippet": f"The current population of {city.title()} is approximately {pop:,} people.",  # noqa: E501
                        "url": f"https://worldpopulation.org/{city.replace(' ', '-')}",
                        "source": "World Population Review",
                    })

        elif "weather" in query_lower:
            for city, weather in MOCK_DATA["weather"].items():
                if city in query_lower:
                    results.append({
                        "title": f"Current Weather in {city.title()}",
                        "snippet": f"Temperature: {weather['temp']}°C, Condition: {weather['condition']}, Humidity: {weather['humidity']}%",  # noqa: E501
                        "url": f"https://weather.com/{city.replace(' ', '-')}",
                        "source": "Weather.com",
                    })

        elif "attractions" in query_lower or "tourist" in query_lower or "visit" in query_lower:
            for city, attractions in MOCK_DATA["attractions"].items():
                if city in query_lower:
                    for attraction in attractions[:3]:  # Top 3
                        results.append({
                            "title": f"{attraction['name']} - {city.title()}",
                            "snippet": f"Rating: {attraction['rating']}/5, Cost: {attraction['cost']}",  # noqa: E501
                            "url": f"https://tripadvisor.com/{attraction['name'].replace(' ', '-').lower()}",  # noqa: E501
                            "source": "TripAdvisor",
                        })

        # Generic results if no specific match
        if not results:
            results = [
                {
                    "title": f"Search Results for: {query}",
                    "snippet": f"Comprehensive information about {query} from reliable sources.",
                    "url": f"https://example.com/search?q={query.replace(' ', '+')}",
                    "source": "Demo Search Engine",
                },
            ]

        return {
            "results": results[:max_results],
            "total_found": len(results),
            "search_time_ms": random.randint(50, 200),
            "query": query,
        }

    def weather_api(self, city: str) -> dict[str, Any]:
        """Simulate weather API call."""
        self.simulate_delay(100, 400)

        city_lower = city.lower()
        weather_data = MOCK_DATA["weather"].get(city_lower)

        if weather_data:
            return {
                "city": city.title(),
                "temperature": weather_data["temp"],
                "condition": weather_data["condition"],
                "humidity": weather_data["humidity"],
                "timestamp": datetime.now().isoformat(),
                "source": "Demo Weather API",
            }
        # Generate random weather for unknown cities
        return {
            "city": city.title(),
            "temperature": random.randint(10, 30),
            "condition": random.choice(["sunny", "cloudy", "rainy", "partly cloudy"]),
            "humidity": random.randint(40, 80),
            "timestamp": datetime.now().isoformat(),
            "source": "Demo Weather API (estimated)",
        }

    def calculator(self, expression: str, operation: str = "evaluate") -> dict[str, Any]:
        """Simulate calculator operations."""
        self.simulate_delay(50, 150)

        try:
            # Simple expression evaluation (be careful in real implementations!)
            if operation == "evaluate":
                # Only allow basic math operations for safety
                allowed_chars = set("0123456789+-*/.() ")
                if all(c in allowed_chars for c in expression):
                    result = eval(expression)
                else:
                    raise ValueError("Invalid characters in expression")
            else:
                result = f"Operation '{operation}' not implemented in demo"

            return {
                "expression": expression,
                "result": result,
                "operation": operation,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "expression": expression,
                "error": str(e),
                "operation": operation,
                "timestamp": datetime.now().isoformat(),
            }

    def data_lookup(self, category: str, query: str) -> dict[str, Any]:
        """Simulate data lookup from various sources."""
        self.simulate_delay(150, 600)

        query_lower = query.lower()

        if category == "population":
            for city, pop in MOCK_DATA["populations"].items():
                if city in query_lower:
                    return {
                        "category": category,
                        "query": query,
                        "result": {
                            "city": city.title(),
                            "population": pop,
                            "year": 2024,
                            "source": "World Population Database",
                        },
                    }

        elif category == "attractions":
            for city, attractions in MOCK_DATA["attractions"].items():
                if city in query_lower:
                    return {
                        "category": category,
                        "query": query,
                        "result": {
                            "city": city.title(),
                            "attractions": attractions,
                            "source": "Tourism Database",
                        },
                    }

        # Generic response
        return {
            "category": category,
            "query": query,
            "result": f"Demo data for {query} in category {category}",
            "source": "Demo Database",
        }

# Global server instance
mcp_server = MCPServer()

@app.get("/health")
async def health() -> dict[str, Any]:
    """Health check endpoint."""
    return {"status": "healthy", "server": "demo-mcp", "timestamp": datetime.now().isoformat()}

@app.post("/tools/list")
async def list_tools() -> dict[str, Any]:
    """List available tools (MCP format)."""
    return {
        "tools": list(mcp_server.tools.values()),
    }

@app.post("/tools/call")
async def call_tool(request: dict[str, Any]) -> dict[str, Any]:
    """Call a specific tool (MCP format)."""
    tool_name = request.get("name")
    arguments = request.get("arguments", {})

    if tool_name not in mcp_server.tools:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

    try:
        # Call the appropriate tool method
        if tool_name == "web_search":
            result = mcp_server.web_search(**arguments)
        elif tool_name == "weather_api":
            result = mcp_server.weather_api(**arguments)
        elif tool_name == "calculator":
            result = mcp_server.calculator(**arguments)
        elif tool_name == "data_lookup":
            result = mcp_server.data_lookup(**arguments)
        else:
            raise HTTPException(status_code=400, detail=f"Tool '{tool_name}' not implemented")

        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result, indent=2),
                },
            ],
            "isError": False,
        }

    except Exception as e:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Error executing tool '{tool_name}': {e!s}",
                },
            ],
            "isError": True,
        }

@app.get("/")
async def root() -> dict[str, Any]:
    """Root endpoint with server info."""
    return {
        "server": "Demo MCP Server",
        "version": "1.0.0",
        "tools": len(mcp_server.tools),
        "available_tools": list(mcp_server.tools.keys()),
        "description": "Mock MCP server for reasoning agent demonstration",
    }

def main() -> None:
    """Run the demo MCP server."""
    parser = argparse.ArgumentParser(description="Demo MCP Server")
    parser.add_argument("--port", type=int, default=8001, help="Port to run server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()

    print(f"🚀 Starting Demo MCP Server on {args.host}:{args.port}")
    print(f"📋 Available tools: {', '.join(mcp_server.tools.keys())}")
    print(f"🔗 Health check: http://{args.host}:{args.port}/health")
    print(f"🛠️  Tools endpoint: http://{args.host}:{args.port}/tools/list")

    uvicorn.run(
        "demo_mcp_server:app",
        host=args.host,
        port=args.port,
        reload=False,
        access_log=True,
    )

if __name__ == "__main__":
    main()
