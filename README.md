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

## Running the MCP Server
```bash
python mcp_server.py
```

### Connecting a client to a tool
```python
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
            result = await session.call_tool("list_categories", {})
            print(result)
```

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