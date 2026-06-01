"""Dataset analysis tools exposed to the LangGraph ReAct agent."""
from __future__ import annotations

from typing import Optional

import pandas as pd
from langchain_core.tools import tool
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, Field

from data.loader import load_dataset_df
from agent.profile import load_profile

_df: Optional[pd.DataFrame] = None


def _get_df() -> pd.DataFrame:
    """Lazy-load the dataset once and cache it in-process."""
    global _df
    if _df is None:
        _df = load_dataset_df()
    return _df


# ---------------------------------------------------------------------------
# Tool input schemas (Pydantic)
# ---------------------------------------------------------------------------

class CategoryFilter(BaseModel):
    category: Optional[str] = Field(None, description="Category name to filter by (case-insensitive). Leave empty for all categories.")


class RequiredCategoryFilter(BaseModel):
    category: str = Field(..., description="Category name to filter by (case-insensitive). Required.")


class IntentFilter(BaseModel):
    category: Optional[str] = Field(None, description="Category name to filter by (case-insensitive).")
    intent: Optional[str] = Field(None, description="Intent name to filter by (case-insensitive).")


class ExampleArgs(BaseModel):
    category: Optional[str] = Field(None, description="Category to sample from. Leave empty for all categories.")
    intent: Optional[str] = Field(None, description="Intent to sample from. Leave empty for all intents.")
    n: int = Field(5, description="Number of examples to return (default 5, max 20).")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool(args_schema=CategoryFilter)
def list_categories(category: Optional[str] = None) -> list[str]:
    """Return all unique top-level categories present in the Bitext dataset.
    Use this first to understand what broad topics the data covers.
    Do not filter by category here — it is ignored; all categories are returned.
    """
    df = _get_df()
    return sorted(df["category"].unique().tolist())


@tool(args_schema=CategoryFilter)
def list_intents(category: Optional[str] = None) -> list[str]:
    """Return unique intent labels, optionally filtered to a single category.
    Use to discover which specific intents exist before counting or sampling.
    """
    df = _get_df()
    if category:
        df = df[df["category"].str.lower() == category.lower()]
    return sorted(df["intent"].unique().tolist())


@tool(args_schema=IntentFilter)
def count_rows(category: Optional[str] = None, intent: Optional[str] = None) -> int:
    """Count dataset rows matching optional category and/or intent filters.
    Use to answer 'how many X?' questions. Chain with list_intents when
    the user mentions an intent by natural language (e.g. 'refund requests').
    """
    df = _get_df()
    if category:
        df = df[df["category"].str.lower() == category.lower()]
    if intent:
        df = df[df["intent"].str.lower() == intent.lower()]
    return len(df)


@tool(args_schema=RequiredCategoryFilter)
def get_distribution(category: str) -> dict[str, int]:
    """Return a {intent: count} distribution for a given category.
    Use when asked about the breakdown or distribution of intents within a category.
    """
    df = _get_df()
    filtered = df[df["category"].str.lower() == category.lower()]
    return filtered["intent"].value_counts().to_dict()


@tool(args_schema=ExampleArgs)
def get_examples(
    category: Optional[str] = None,
    intent: Optional[str] = None,
    n: int = 5,
) -> list[dict]:
    """Return N example rows (instruction + response) from the dataset.
    Filter by category and/or intent. Use for 'show me examples of X' queries.
    Max 20 examples at a time to avoid context overflow.
    """
    df = _get_df()
    if category:
        df = df[df["category"].str.lower() == category.lower()]
    if intent:
        df = df[df["intent"].str.lower() == intent.lower()]
    sample = df.sample(min(n, 20, len(df)))
    return sample[["instruction", "response", "intent", "category"]].to_dict(orient="records")


@tool(args_schema=RequiredCategoryFilter)
def summarize_category(category: str) -> str:
    """Summarise the types of customer queries and agent responses in a given category.
    Samples up to 30 rows and returns them as a formatted string for the agent to synthesise.
    Use for open-ended questions like 'Summarise the FEEDBACK category'.
    """
    df = _get_df()
    filtered = df[df["category"].str.lower() == category.lower()]
    if filtered.empty:
        return f"No rows found for category '{category}'."
    sample = filtered.sample(min(30, len(filtered)))
    lines = []
    for _, row in sample.iterrows():
        lines.append(f"Intent: {row['intent']}\nQuery: {row['instruction']}\nResponse: {row['response']}\n---")
    return f"Sample from '{category}' ({len(filtered)} total rows):\n\n" + "\n".join(lines)


@tool
def get_my_profile(config: RunnableConfig) -> str:
    """Return the user's current profile.
    Use this when the user asks "what do you remember about me?" or similar queries.
    """
    session_id = config["configurable"]["thread_id"]
    profile = load_profile(session_id)
    if not profile:
        return "I don't have a profile for you yet."

    # Format the profile as a readable string
    lines = ["── My notes about you ──"]
    for k, v in profile.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


ALL_TOOLS = [
    list_categories,
    list_intents,
    count_rows,
    get_distribution,
    get_examples,
    summarize_category,
    get_my_profile,
]
