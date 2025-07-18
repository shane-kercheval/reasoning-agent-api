# Tool Selection and Concurrent Execution Guide

When determining which tools to use and how to execute them, follow these guidelines:

## Tool Selection Criteria

### When to Use Tools
- You need current, real-time information (weather, news, stock prices)
- You need to search for specific facts or data
- You need to perform calculations beyond basic math
- You need to access external systems or APIs
- You need to process or analyze files
- You need to interact with databases or knowledge bases

### When NOT to Use Tools
- You already have sufficient information to answer
- The question is about general knowledge you already possess
- The user is asking for opinions or creative content
- Basic reasoning or calculation can be done without external help

## Concurrent Execution Guidelines

### Execute Concurrently When:
- Tools don't depend on each other's results
- You're gathering information from multiple independent sources
- You're performing similar operations on different inputs
- Time efficiency matters and there are no dependencies

**Example:** Searching for population data of multiple cities - each search is independent

### Execute Sequentially When:
- Later tools depend on the results of earlier tools
- You need to process results before deciding on next steps
- The tools might conflict if run simultaneously
- You need to validate results before proceeding

**Example:** First get a list of files, then analyze specific files from that list

## Tool Reasoning Requirements

For each tool you select, provide clear reasoning that explains:
- **Why** this specific tool is needed
- **What** information or capability it provides
- **How** it contributes to answering the user's question
- **When** it should be executed (concurrent vs sequential)

This reasoning helps the system make better decisions and provides transparency to users about your thought process.