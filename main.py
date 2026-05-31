"""CLI entry point for the Customer Service Data Analyst Agent."""
import argparse
from agent.graph import build_graph
from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.markdown import Markdown
from langgraph.errors import GraphRecursionError

console = Console()


def run_cli(session_id: str) -> None:
    """Start an interactive conversation loop with the agent.

    Args:
        session_id: Unique session identifier. Restoring the same ID resumes
            the persisted conversation history from the SQLite checkpointer.
    """
    app = build_graph()
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 25,  # 12 agent steps ≈ 25 graph nodes (each step = ~2 nodes)
    }

    console.print(f"[bold green]Agent ready[/] — session: [cyan]{session_id}[/]")
    console.print("Type [bold]exit[/] or [bold]quit[/] to stop.\n")

    while True:
        try:
            user_input = console.input("[bold blue]You:[/] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/]")
            break

        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        console.rule("[dim]Reasoning[/]")
        try:
            for event in app.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="values",
            ):
                # Print tool calls and observations as they arrive
                last_msg = event["messages"][-1]
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        console.print(f"  [yellow]→ Tool:[/] {tc['name']}  [dim]{tc['args']}[/]")
                elif last_msg.type == "tool":
                    console.print(f"  [green]← Obs:[/] {str(last_msg.content)[:300]}")

            console.rule("[dim]Answer[/]")
            final = event["messages"][-1].content
            console.print(Markdown(str(final)))
            console.print()
        except GraphRecursionError:
            console.print("[red]I wasn't able to complete this in the allowed steps. Please try rephrasing your question.[/]")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer Service Data Analyst Agent")
    parser.add_argument(
        "--session",
        default="default",
        help="Session ID for persistent conversation memory (default: 'default')",
    )
    args = parser.parse_args()
    run_cli(session_id=args.session)
