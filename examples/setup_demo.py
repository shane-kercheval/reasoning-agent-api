#!/usr/bin/env python3
"""
Setup script for the Reasoning Agent Demo.

This script sets up everything needed to run the reasoning agent demo:
1. Installs required dependencies
2. Checks environment configuration
3. Provides instructions for running the demo

Usage:
    python setup_demo.py
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"📦 {description}...")
    try:
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False

def check_dependency(command: str, name: str) -> bool:
    """Check if a dependency is available."""
    try:
        subprocess.run(command, shell=True, check=True, capture_output=True)
        print(f"✅ {name} is available")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ {name} is not available")
        return False

def main() -> bool:
    """Main setup function."""
    print("🚀 Setting up Reasoning Agent Demo")
    print("=" * 50)

    # Check if we're in the right directory
    if not Path("api").exists() or not Path("pyproject.toml").exists():
        print("❌ Error: Please run this script from the project root directory")
        print("   Expected files: api/, pyproject.toml")
        return False

    print("\n📋 Checking system dependencies...")

    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 11):
        print(f"❌ Python 3.11+ required, found {python_version.major}.{python_version.minor}")
        return False
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")

    # Check for uv
    if not check_dependency("uv --version", "uv package manager"):
        print("\n📥 Installing uv package manager...")
        if not run_command("curl -LsSf https://astral.sh/uv/install.sh | sh", "Installing uv"):
            print("❌ Failed to install uv. Please install manually: https://docs.astral.sh/uv/")
            return False

    print("\n📦 Installing demo dependencies...")

    # Install rich for beautiful console output
    success = run_command("uv add rich", "Installing rich for demo UI")
    if not success:
        print("⚠️  Warning: Failed to install rich. Demo will work but with less pretty output")

    # Install project dependencies
    success = run_command("uv sync", "Installing project dependencies")
    if not success:
        print("❌ Failed to install project dependencies")
        return False

    print("\n🔧 Setting up demo configuration...")

    # Copy demo config if it doesn't exist
    demo_config = Path("config/demo_mcp_servers.yaml")
    main_config = Path("config/mcp_servers.yaml")

    if demo_config.exists() and not main_config.exists():
        run_command(f"cp {demo_config} {main_config}", "Setting up MCP configuration")

    print("\n🔍 Checking environment...")

    # Check for OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print("✅ OPENAI_API_KEY is set")
    else:
        print("⚠️  OPENAI_API_KEY is not set")
        print("   Set it with: export OPENAI_API_KEY=your_key_here")
        print("   The demo will not work without this!")

    print("\n🎯 Demo Setup Complete!")
    print("=" * 50)

    print("\n📖 How to run the demo:")
    print("1. Start the main reasoning agent server:")
    print("   uv run python -m api.main")
    print("")
    print("2. (Optional) Start the MCP server for tool demos:")
    print("   uv run python mcp_server/server.py")
    print("")
    print("3. Run the interactive demo:")
    print("   uv run python examples/demo_reasoning_agent.py")
    print("")

    if not openai_key:
        print("⚠️  Remember to set OPENAI_API_KEY before running the demo!")

    print("\n🎉 Ready to demo the sophisticated reasoning agent!")

    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error during setup: {e}")
        sys.exit(1)
