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
    pending_suggestion: str | None


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


def recommender_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """Generate a specific query suggestion based on conversation history and user profile."""
    from langchain_core.messages import AIMessage
    from agent.profile import load_profile

    session_id = config["configurable"]["thread_id"]
    profile = load_profile(session_id)

    # Get last 6 messages
    recent_messages = state["messages"][-6:]
    history_text = "\n".join([f"{m.type}: {m.content}" for m in recent_messages])

    prompt = f"""Based on this conversation history and user profile, suggest ONE specific follow-up query about the Bitext dataset. Be specific — name the category or intent. Format your response as:
SUGGESTION: <the suggested query text>
REASON: <one sentence why>

Profile: {profile}

History:
{history_text}"""

    model = _make_llm(ROUTER_MODEL)
    response = model.invoke(prompt)

    suggestion = ""
    reason = ""
    for line in response.content.split('\n'):
        if line.startswith('SUGGESTION:'):
            suggestion = line.replace('SUGGESTION:', '').strip()
        elif line.startswith('REASON:'):
            reason = line.replace('REASON:', '').strip()

    return {
        "pending_suggestion": suggestion,
        "messages": [
            AIMessage(
                content=f"Based on your recent interest, I'd suggest:\n**{suggestion}**\nShould I go ahead, or would you like to refine it?"
            )
        ]
    }

def confirmation_check_node(state: AgentState) -> dict:
    """Check if the user confirmed or refined the suggestion."""
    from langchain_core.messages import HumanMessage
    import re

    last_message = state["messages"][-1].content.lower()
    confirm_words = [r"\byes\b", r"\bgo\b", r"\bdo it\b", r"\bsure\b", r"\bproceed\b", r"\bok\b"]

    is_confirmed = any(re.search(word, last_message) for word in confirm_words)

    if is_confirmed:
        return {
            "pending_suggestion": None,
            "messages": [HumanMessage(content=state["pending_suggestion"])]
        }
    else:
        return {"pending_suggestion": state["pending_suggestion"]}

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


def route_start(state: AgentState) -> Literal["confirmation_check_node", "router_node"]:
    """Decide where to start based on pending_suggestion."""
    if state.get("pending_suggestion") is not None:
        return "confirmation_check_node"
    return "router_node"

def route_after_router(
    state: AgentState,
) -> Literal["react_agent", "decline_node", "recommender_node"]:
    """Decide which branch to run based on query_type."""
    if state.get("query_type") == "out_of_scope":
        return "decline_node"
    elif state.get("query_type") == "recommendation":
        return "recommender_node"
    return "react_agent"

def route_after_confirmation(state: AgentState) -> Literal["react_agent", "recommender_node"]:
    """Route based on whether the suggestion was confirmed."""
    if state.get("pending_suggestion") is None:
        return "react_agent"
    return "recommender_node"

def build_graph() -> any:
    """Compile and return the LangGraph application.

    Returns:
        A compiled LangGraph `CompiledGraph` with SqliteSaver checkpointing.
    """
    llm = _make_llm(AGENT_MODEL)

    # Inner ReAct agent (handles both structured + unstructured branches)
    react = create_react_agent(
        llm,
        tools=ALL_TOOLS,
        prompt=SYSTEM_PROMPT,
    )

    builder = StateGraph(AgentState)
    builder.add_node("router_node", router_node)
    builder.add_node("react_agent", react)
    builder.add_node("profile_update_node", profile_update_node)
    builder.add_node("decline_node", decline_node)

    builder.add_node("recommender_node", recommender_node)
    builder.add_node("confirmation_check_node", confirmation_check_node)

    builder.add_conditional_edges(
        START,
        route_start,
        {"confirmation_check_node": "confirmation_check_node", "router_node": "router_node"},
    )
    builder.add_conditional_edges(
        "router_node",
        route_after_router,
        {"react_agent": "react_agent", "decline_node": "decline_node", "recommender_node": "recommender_node"},
    )

    builder.add_conditional_edges(
        "confirmation_check_node",
        route_after_confirmation,
        {"react_agent": "react_agent", "recommender_node": "recommender_node"},
    )

    builder.add_edge("recommender_node", END)
    builder.add_edge("react_agent", "profile_update_node")
    builder.add_edge("profile_update_node", END)
    builder.add_edge("decline_node", END)

    # The thread_id from the config links the CLI session ID to the persisted checkpoint here
    checkpointer = SqliteSaver.from_conn_string("agent_memory.db")
    # Using the context manager returned from from_conn_string which evaluates to a checkpointer.
    # However since we need to return it, we should ensure the DB connection stays open for the graph.
    # The SqliteSaver from_conn_string actually returns a context manager that we shouldn't use directly.
    import sqlite3
    conn = sqlite3.connect("agent_memory.db", check_same_thread=False)
    saver = SqliteSaver(conn)
    return builder.compile(checkpointer=saver)
