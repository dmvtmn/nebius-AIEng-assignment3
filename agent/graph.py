"""LangGraph ReAct agent graph with router, persistent memory, and user profile."""
from __future__ import annotations

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent

from agent.tools import ALL_TOOLS
from agent.router import classify_query, QueryType
from agent.profile import update_profile

load_dotenv()

NEBIUS_BASE_URL = "https://api.studio.nebius.com/v1/"
AGENT_MODEL = os.getenv("NEBIUS_AGENT_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct")
ROUTER_MODEL = os.getenv("NEBIUS_ROUTER_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")

SYSTEM_PROMPT = """You are a data analyst assistant for the Bitext Customer Service dataset.
You can only answer questions about this dataset. For out-of-scope questions, politely decline.

Available tools let you explore categories, intents, distributions, and examples.
Always reason step by step, using the minimum number of tool calls needed.
To count refund requests: first call list_intents to find the exact intent name, then call count_rows with that intent.
"""


def _make_llm(model: str) -> ChatOpenAI:
    """Instantiate a Nebius Token Factory LLM."""
    return ChatOpenAI(
        model=model,
        api_key=os.environ["NEBIUS_API_KEY"],
        base_url=NEBIUS_BASE_URL,
    )


class AgentState(MessagesState):
    """Extended graph state that also carries routing result and user profile."""
    query_type: QueryType | None  # set by router_node
    user_profile: str  # persisted distilled facts about the user


def router_node(state: AgentState) -> AgentState:
    """Classify the latest user message as structured / unstructured / out_of_scope."""
    last_human = next(
        (m for m in reversed(state["messages"]) if m.type == "human"), None
    )
    if last_human is None:
        return {"query_type": "out_of_scope"}
    qt = classify_query(last_human.content, model=_make_llm(ROUTER_MODEL))
    print(f"  Router classification: {qt}")
    return {"query_type": qt}


def profile_update_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """Update user profile using the last Human and AI messages."""
    session_id = config["configurable"]["thread_id"]

    last_human = next(
        (m.content for m in reversed(state["messages"]) if m.type == "human"), ""
    )
    last_agent = next(
        (m.content for m in reversed(state["messages"]) if m.type == "ai"), ""
    )

    update_profile(
        session_id=session_id,
        last_human=last_human,
        last_agent=last_agent,
        model=_make_llm(ROUTER_MODEL)
    )
    return state


def decline_node(state: AgentState) -> AgentState:
    """Return a polite refusal for out-of-scope queries."""
    from langchain_core.messages import AIMessage
    return {
        "messages": [
            AIMessage(
                content="I'm sorry, that question is outside my scope. "
                "I can only answer questions about the Bitext Customer Service dataset."
            )
        ]
    }


def route_after_router(
    state: AgentState,
) -> Literal["react_agent", "decline_node"]:
    """Decide which branch to run based on query_type."""
    if state.get("query_type") == "out_of_scope":
        return "decline_node"
    return "react_agent"


def build_graph(system_prompt: str | None = None) -> any:
    """Compile and return the LangGraph application.

    Args:
        system_prompt: Optional custom system prompt string. If None, uses default.

    Returns:
        A compiled LangGraph `CompiledGraph` with SqliteSaver checkpointing.
    """
    llm = _make_llm(AGENT_MODEL)

    prompt_to_use = system_prompt if system_prompt is not None else SYSTEM_PROMPT

    # Inner ReAct agent (handles both structured + unstructured branches)
    react = create_react_agent(
        llm,
        tools=ALL_TOOLS,
        state_modifier=SystemMessage(content=prompt_to_use),
    )

    builder = StateGraph(AgentState)
    builder.add_node("router_node", router_node)
    builder.add_node("react_agent", react)
    builder.add_node("profile_update_node", profile_update_node)
    builder.add_node("decline_node", decline_node)

    builder.add_edge(START, "router_node")
    builder.add_conditional_edges(
        "router_node",
        route_after_router,
        {"react_agent": "react_agent", "decline_node": "decline_node"},
    )
    builder.add_edge("react_agent", "profile_update_node")
    builder.add_edge("profile_update_node", END)
    builder.add_edge("decline_node", END)

    # The thread_id from the config links the CLI session ID to the persisted checkpoint here
    checkpointer = SqliteSaver.from_conn_string("agent_memory.db")
    return builder.compile(checkpointer=checkpointer)
