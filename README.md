# Nebius Assignment 3 — Customer Service Data Analyst Agent

**Student:** Daniel Marques Vieira

## Overview

A LangGraph ReAct agent that answers structured, unstructured, and out-of-scope questions about the [Bitext Customer Service dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset). Tools are also exposed via a FastMCP server.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  LangGraph Graph                │
│                                                 │
│  user input → [router_node]                     │
│                    │                            │
│         ┌──────────┼──────────┐                 │
│         ▼          ▼          ▼                 │
│   structured  unstructured  out-of-scope        │
│         │          │          │                 │
│         └──────────┘          │                 │
│                │              ▼                 │
│         [react_agent]   [decline_node]          │
│                │                                │
│       [SqliteSaver checkpoint]                  │
└─────────────────────────────────────────────────┘
```

### Model Choice
- **Main agent:** `TODO: pick from Nebius Token Factory` — used for reasoning, tool-calling, and summarization.
- **Router:** `TODO: smaller/cheaper model` — lightweight classifier for structured / unstructured / out-of-scope.

> Justify your choices here once decided.

### Tools
| Tool | Description |
|------|-------------|
| `list_categories` | Returns all unique categories in the dataset |
| `list_intents` | Returns all unique intents, optionally filtered by category |
| `count_rows` | Counts rows matching optional category / intent filters |
| `get_distribution` | Returns intent distribution within a category |
| `get_examples` | Returns N sample rows for a given category / intent |
| `summarize_category` | Summarises agent responses for a given category (unstructured) |

## Setup

### 1. Clone & install
```bash
git clone https://github.com/dmvtmn/nebius-AIEng-assignment3.git
cd nebius-AIEng-assignment3
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment variables
```bash
cp .env.example .env
# Fill in NEBIUS_API_KEY
```

### 3. Download dataset
```bash
python scripts/download_data.py
```

## Running the CLI
```bash
# Start a new session
python main.py

# Resume a named session (persists across restarts)
python main.py --session my_session
```

The CLI prints each reasoning step (tool calls + observations) before the final answer.

## MCP Server

### Starting the server
```bash
python mcp_server.py          # stdio mode
python mcp_server.py --sse    # SSE mode on port 8000
```

### Connecting a client (stdio)
```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List tools
            tools = await session.list_tools()
            print("Available tools:", [t.name for t in tools.tools])

            # Call tool
            result = await session.call_tool("list_categories", {})
            print("Categories:", result.content)

if __name__ == "__main__":
    asyncio.run(main())
```

### Connecting via SSE
```python
import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

async def main():
    # Connect to the SSE endpoint
    async with sse_client("http://localhost:8000/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List tools
            tools = await session.list_tools()
            print("Available tools:", [t.name for t in tools.tools])

            # Call tool
            result = await session.call_tool("list_categories", {})
            print("Categories:", result.content)

if __name__ == "__main__":
    asyncio.run(main())
```

### Claude Desktop config
```json
{
  "mcpServers": {
    "bitext-analyst": {
      "command": "python",
      "args": ["/path/to/nebius-AIEng-assignment3/mcp_server.py"]
    }
  }
}
```

## User Profile
The agent extracts and persists facts about users in a user profile, separate from standard message history.

- **What it stores**: Distilled facts like name, frequent topics, and preferences. It does not store full message history.
- **Where it lives**: `profiles/{session_id}.json`
- **How to test it**:
    1. `python main.py --session me`
    2. Say "My name is Daniel and I'm mostly interested in refund data"
    3. Ask "What do you remember about me?"
    4. Exit and restart: `python main.py --session me`
    5. Ask "What do you remember about me?" again — profile persists

## Debugging
Enable verbose logging:
```bash
LANGCHAIN_VERBOSE=true python main.py
```
Or open the graph in **LangGraph Studio** (free, no config needed beyond `langgraph.json`).

## Testing Memory

1. python main.py --session grader_test
2. Ask: "Show me 3 examples from the REFUND category"
3. Ask: "Show me 3 more"  (must use context from previous turn)
4. Ask: "How many complaints did we get?"
5. Ask: "What about refunds?"  (must resolve "refunds" from history)
6. Exit and restart: python main.py --session grader_test
7. Ask: "What did we just discuss?"  (must restore conversation)
