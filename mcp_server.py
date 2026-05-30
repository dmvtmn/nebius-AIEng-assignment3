"""FastMCP server exposing a subset of agent tools as MCP endpoints."""
from __future__ import annotations

from typing import Optional

from fastmcp import FastMCP

from data.loader import load_dataset_df

mcp = FastMCP("bitext-customer-service-analyst")

_df = None


def _get_df():
    global _df
    if _df is None:
        _df = load_dataset_df()
    return _df


@mcp.tool
def list_categories() -> list[str]:
    """Return all unique top-level categories in the Bitext Customer Service dataset.
    Use this to understand what broad topics are covered before drilling deeper.
    """
    df = _get_df()
    return sorted(df["category"].unique().tolist())


@mcp.tool
def count_rows(category: Optional[str] = None, intent: Optional[str] = None) -> int:
    """Count dataset rows matching optional category and/or intent filters.
    Pass category='REFUND' to count all refund rows, or add intent='get_refund'
    to narrow further. Omit both to count all rows.
    """
    df = _get_df()
    if category:
        df = df[df["category"].str.lower() == category.lower()]
    if intent:
        df = df[df["intent"].str.lower() == intent.lower()]
    return len(df)


@mcp.tool
def get_examples(
    category: Optional[str] = None,
    intent: Optional[str] = None,
    n: int = 5,
) -> list[dict]:
    """Return N example rows (instruction + response) from the dataset.
    Filter by category and/or intent. Returns at most 20 rows.
    """
    df = _get_df()
    if category:
        df = df[df["category"].str.lower() == category.lower()]
    if intent:
        df = df[df["intent"].str.lower() == intent.lower()]
    sample = df.sample(min(n, 20, len(df)))
    return sample[["instruction", "response", "intent", "category"]].to_dict(orient="records")


@mcp.tool
def get_distribution(category: str) -> dict[str, int]:
    """Return a {intent: count} distribution for a given category.
    Use to understand the breakdown of intents within a category.
    """
    df = _get_df()
    filtered = df[df["category"].str.lower() == category.lower()]
    return filtered["intent"].value_counts().to_dict()


@mcp.tool
def list_intents(category: Optional[str] = None) -> list[str]:
    """Return unique intent labels, optionally filtered to a single category."""
    df = _get_df()
    if category:
        df = df[df["category"].str.lower() == category.lower()]
    return sorted(df["intent"].unique().tolist())


if __name__ == "__main__":
    mcp.run()
