"""Query router — classifies user input as structured, unstructured, or out_of_scope."""
from __future__ import annotations

from typing import Literal

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

QueryType = Literal["structured", "unstructured", "out_of_scope"]


class RouteDecision(BaseModel):
    """Structured output schema for the router LLM call."""

    query_type: QueryType
    reasoning: str


_ROUTER_PROMPT = """\
You classify user questions about the Bitext Customer Service dataset.

Categories:
- structured: has a concrete data-driven answer (counts, distributions, examples, category/intent lists)
- unstructured: requires summarization or qualitative analysis of the data
- out_of_scope: unrelated to the dataset (general knowledge, coding help, creative writing, etc.)

User question: {question}

Respond with the category and a one-sentence reasoning.
"""


def classify_query(question: str, model: BaseChatModel) -> QueryType:
    """Classify a user question using the router LLM.

    Args:
        question: Raw user question string.
        model: Instantiated LLM to use for classification.

    Returns:
        One of 'structured', 'unstructured', or 'out_of_scope'.
    """
    structured_model = model.with_structured_output(RouteDecision)  # Ensure router works correctly
    result: RouteDecision = structured_model.invoke(
        _ROUTER_PROMPT.format(question=question)
    )
    return result.query_type
