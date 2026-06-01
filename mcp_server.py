"""FastMCP server exposing a subset of agent tools as MCP endpoints."""
from __future__ import annotations

import argparse
from typing import Optional

from fastmcp import FastMCP

from agent.tools import (
    list_categories as _list_categories,
    list_intents as _list_intents,
    count_rows as _count_rows,
    get_distribution as _get_distribution,
    get_examples as _get_examples,
)

mcp = FastMCP("bitext-customer-service-analyst")

@mcp.tool()
def list_categories() -> list[str]:
    """Return all unique top-level categories in the Bitext Customer Service dataset.
    Use this to understand what broad topics are covered before drilling deeper.
    """
    return _list_categories.invoke({})

@mcp.tool()
def list_intents(category: str) -> list[str]:
    """Return unique intent labels, optionally filtered to a single category.
    Use to discover which specific intents exist before counting or sampling.
    """
    return _list_intents.invoke({"category": category})

@mcp.tool()
def count_rows(category: Optional[str] = None, intent: Optional[str] = None) -> int:
    """Count dataset rows matching optional category and/or intent filters.
    Pass category='REFUND' to count all refund rows, or add intent='get_refund'
    to narrow further. Omit both to count all rows.
    """
    return _count_rows.invoke({"category": category, "intent": intent})

@mcp.tool()
def get_distribution(category: str) -> dict[str, int]:
    """Return a {intent: count} distribution for a given category.
    Use to understand the breakdown of intents within a category.
    """
    return _get_distribution.invoke({"category": category})

@mcp.tool()
def get_examples(intent: str, n: int = 3) -> list[dict]:
    """Return N example rows (instruction + response) from the dataset.
    Filter by intent. Returns at most 20 rows.
    """
    return _get_examples.invoke({"intent": intent, "n": n})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sse", action="store_true", help="Run SSE transport on port 8000")
    args = parser.parse_args()

    if args.sse:
        mcp.run(transport="sse", host="0.0.0.0", port=8000)
    else:
        mcp.run(transport="stdio")
