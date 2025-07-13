"""
Production-ready MCP Server with fake tools for demos and testing.

This server provides realistic fake tools that can be deployed to cloud platforms
like Render, Railway, or any hosting service. It's designed for demonstrations
and integration testing with the reasoning agent.

Available tools:
- get_weather: Get weather information for any location (fake data)
- search_web: Search the web and return results (fake data)
- get_stock_price: Get current stock price information (fake data)
- analyze_text: Analyze text sentiment and extract insights (fake data)
- translate_text: Translate text between languages (fake data)

Usage:
    # Local development
    uv run python mcp_servers/fake_server.py
    
    # Production deployment
    python mcp_servers/fake_server.py

The server will start on port specified by PORT environment variable (default: 8000)
and be accessible at http://localhost:PORT/mcp/
"""

import json
import os
import random
from datetime import datetime, timedelta
from typing import Literal

from fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("demo-tools-server")


@mcp.tool
async def get_weather(
    location: str,
    units: Literal["celsius", "fahrenheit"] = "celsius"
) -> dict:
    """Get current weather information for any location worldwide."""
    
    # Generate realistic fake weather data
    conditions = ["Clear", "Sunny", "Partly Cloudy", "Cloudy", "Light Rain", "Rain", "Snow", "Windy"]
    wind_directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    
    if units == "fahrenheit":
        temp = random.randint(32, 95)
        feels_like = temp + random.randint(-5, 10)
        temp_unit = "°F"
        wind_speed_unit = "mph"
        wind_speed = random.randint(3, 25)
    else:
        temp = random.randint(0, 35)
        feels_like = temp + random.randint(-3, 5)
        temp_unit = "°C"
        wind_speed_unit = "km/h"
        wind_speed = random.randint(5, 40)
    
    return {
        "location": location,
        "current": {
            "temperature": f"{temp}{temp_unit}",
            "feels_like": f"{feels_like}{temp_unit}",
            "condition": random.choice(conditions),
            "humidity": f"{random.randint(30, 90)}%",
            "wind": f"{wind_speed} {wind_speed_unit} {random.choice(wind_directions)}",
            "pressure": f"{random.randint(995, 1035)} hPa",
            "visibility": f"{random.randint(5, 15)} km",
            "uv_index": random.randint(1, 10)
        },
        "forecast": [
            {
                "date": (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d"),
                "day": (datetime.now() + timedelta(days=i)).strftime("%A"),
                "high": f"{temp + random.randint(-5, 8)}{temp_unit}",
                "low": f"{temp - random.randint(5, 15)}{temp_unit}",
                "condition": random.choice(conditions),
                "precipitation": f"{random.randint(0, 80)}%"
            }
            for i in range(1, 4)
        ],
        "last_updated": datetime.now().isoformat(),
        "source": "Demo Weather API",
        "server": "demo-tools-server"
    }


@mcp.tool
async def search_web(
    query: str,
    max_results: int = 5
) -> dict:
    """Search the web and return relevant results."""
    
    # Generate realistic fake search results
    domains = ["wikipedia.org", "github.com", "stackoverflow.com", "medium.com", "reddit.com", "news.com"]
    
    results = []
    for i in range(min(max_results, 10)):
        domain = random.choice(domains)
        results.append({
            "title": f"Everything you need to know about {query} - Complete Guide",
            "url": f"https://{domain}/{query.lower().replace(' ', '-')}-{i+1}",
            "snippet": f"Comprehensive information about {query}. This article covers the latest developments, best practices, and expert insights on {query}. Learn from industry professionals...",
            "published": (datetime.now() - timedelta(days=random.randint(1, 365))).strftime("%Y-%m-%d"),
            "source": domain.split('.')[0].title(),
            "relevance_score": round(random.uniform(0.7, 0.99), 2)
        })
    
    return {
        "query": query,
        "results": results,
        "total_results": len(results),
        "search_time": f"{random.uniform(0.1, 0.8):.2f}s",
        "timestamp": datetime.now().isoformat(),
        "server": "demo-tools-server"
    }


@mcp.tool
async def get_stock_price(
    symbol: str,
    include_chart: bool = False
) -> dict:
    """Get current stock price and market information."""
    
    # Generate realistic fake stock data
    base_price = random.uniform(50, 500)
    change_percent = random.uniform(-5, 5)
    change_amount = base_price * (change_percent / 100)
    
    result = {
        "symbol": symbol.upper(),
        "current_price": round(base_price, 2),
        "change": round(change_amount, 2),
        "change_percent": round(change_percent, 2),
        "previous_close": round(base_price - change_amount, 2),
        "day_high": round(base_price + random.uniform(0, 10), 2),
        "day_low": round(base_price - random.uniform(0, 10), 2),
        "volume": random.randint(1000000, 50000000),
        "market_cap": f"{random.uniform(1, 1000):.1f}B",
        "pe_ratio": round(random.uniform(10, 30), 1),
        "last_updated": datetime.now().isoformat(),
        "market_status": "OPEN" if 9 <= datetime.now().hour <= 16 else "CLOSED",
        "server": "demo-tools-server"
    }
    
    if include_chart:
        # Generate fake historical data points
        chart_data = []
        for i in range(30):  # 30 days of data
            date = datetime.now() - timedelta(days=i)
            price = base_price + random.uniform(-20, 20)
            chart_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "price": round(price, 2)
            })
        result["chart_data"] = list(reversed(chart_data))
    
    return result


@mcp.tool
async def analyze_text(
    text: str,
    analysis_type: Literal["sentiment", "keywords", "summary", "all"] = "all"
) -> dict:
    """Analyze text for sentiment, keywords, and generate summary."""
    
    # Generate fake analysis results
    sentiments = ["positive", "negative", "neutral"]
    sentiment = random.choice(sentiments)
    
    # Generate fake keywords based on text length
    word_count = len(text.split())
    fake_keywords = [
        "technology", "innovation", "business", "strategy", "growth", "development",
        "market", "customer", "product", "service", "quality", "performance",
        "solution", "opportunity", "challenge", "success", "improvement"
    ]
    keywords = random.sample(fake_keywords, min(5, word_count // 10 + 1))
    
    result = {
        "text_length": len(text),
        "word_count": word_count,
        "timestamp": datetime.now().isoformat(),
        "server": "demo-tools-server"
    }
    
    if analysis_type in ["sentiment", "all"]:
        result["sentiment"] = {
            "label": sentiment,
            "confidence": round(random.uniform(0.7, 0.95), 2),
            "positive_score": round(random.uniform(0, 1), 2),
            "negative_score": round(random.uniform(0, 1), 2),
            "neutral_score": round(random.uniform(0, 1), 2)
        }
    
    if analysis_type in ["keywords", "all"]:
        result["keywords"] = [
            {"word": keyword, "relevance": round(random.uniform(0.5, 1.0), 2)}
            for keyword in keywords
        ]
    
    if analysis_type in ["summary", "all"]:
        result["summary"] = f"This text discusses {', '.join(keywords[:3])} with a {sentiment} tone. The content covers key aspects of the topic with {word_count} words providing detailed insights."
    
    return result


@mcp.tool
async def translate_text(
    text: str,
    target_language: str,
    source_language: str = "auto"
) -> dict:
    """Translate text between languages."""
    
    # Language code mapping for realistic responses
    language_names = {
        "en": "English",
        "es": "Spanish", 
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "ru": "Russian",
        "ja": "Japanese",
        "ko": "Korean",
        "zh": "Chinese",
        "ar": "Arabic"
    }
    
    # Generate fake translation (in real implementation, this would call a translation API)
    fake_translations = {
        "hello": {"es": "hola", "fr": "bonjour", "de": "hallo", "it": "ciao"},
        "thank you": {"es": "gracias", "fr": "merci", "de": "danke", "it": "grazie"},
        "goodbye": {"es": "adiós", "fr": "au revoir", "de": "auf wiedersehen", "it": "arrivederci"}
    }
    
    # Simple fake translation logic
    lower_text = text.lower()
    if lower_text in fake_translations and target_language in fake_translations[lower_text]:
        translated = fake_translations[lower_text][target_language]
    else:
        # For demo purposes, just add language indicator
        translated = f"[{target_language.upper()}] {text}"
    
    detected_language = source_language if source_language != "auto" else "en"
    
    return {
        "original_text": text,
        "translated_text": translated,
        "source_language": {
            "code": detected_language,
            "name": language_names.get(detected_language, "Unknown"),
            "confidence": round(random.uniform(0.8, 0.99), 2) if source_language == "auto" else 1.0
        },
        "target_language": {
            "code": target_language,
            "name": language_names.get(target_language, "Unknown")
        },
        "timestamp": datetime.now().isoformat(),
        "server": "demo-tools-server"
    }


def get_server_instance():
    """Get the FastMCP server instance for in-memory testing."""
    return mcp


if __name__ == "__main__":
    # Get port from environment variable (for deployment platforms like Render)
    # Default to 8001 to avoid conflict with reasoning agent API on 8000
    port = int(os.getenv("PORT", 8001))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"Starting Demo MCP Server on {host}:{port}")
    print("Available tools:")
    print("  - get_weather: Get weather information")
    print("  - search_web: Search the web")
    print("  - get_stock_price: Get stock prices")
    print("  - analyze_text: Analyze text sentiment and keywords")
    print("  - translate_text: Translate between languages")
    print(f"Server accessible at: http://{host}:{port}/mcp/")
    print(f"Health check: http://{host}:{port}/")
    
    # Run as HTTP server
    mcp.run(transport="http", host=host, port=port)