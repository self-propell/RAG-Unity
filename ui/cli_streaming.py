"""
CLI Streaming Support for LangGraph Agent

Provides Rich Live display for streaming chat responses.
"""
import asyncio
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.console import Console
from rich.spinner import Spinner
from rich.text import Text

console = Console()


async def chat_with_streaming(agent, user_message: str, **kwargs):
    """
    Chat with streaming output using Rich Live display.

    Args:
        agent: UnityLangGraphAgent instance
        user_message: User's question
        **kwargs: Additional chat parameters (top_k, filter_path, etc.)

    Returns:
        Chat result dict
    """
    accumulated_text = ""
    current_tool = None

    # Create initial state
    from llm.agent_langgraph import create_initial_state

    initial_state = create_initial_state(
        project_id=agent.project.project_id if agent.project else "_global",
        conv_id=agent._current_conv_id,
        provider_name=agent.provider.provider_name(),
        model_name=agent.provider.model_name(),
        user_message=user_message,
        stream_tokens=True,
        **kwargs
    )

    config_dict = {
        "configurable": {
            "thread_id": agent._current_conv_id or f"conv_{int(__import__('time').time())}"
        }
    }

    # Start Live display
    with Live(console=console, auto_refresh=True, vertical_overflow="visible") as live:

        async for event in agent.graph.astream_events(initial_state, config_dict, version="v1"):
            event_type = event.get("event")

            if event_type == "on_chat_model_stream":
                # Token from LLM
                chunk = event.get("data", {}).get("chunk", {})
                if hasattr(chunk, 'content') and chunk.content:
                    accumulated_text += chunk.content

                    # Update display
                    content = Markdown(accumulated_text)
                    if current_tool:
                        tool_text = Text(f"\n\n🔧 {current_tool}", style="dim")
                        live.update(Panel(content, title="[cyan]Assistant[/cyan]", subtitle=tool_text))
                    else:
                        live.update(Panel(content, title="[cyan]Assistant[/cyan]"))

            elif event_type == "on_tool_start":
                # Tool execution started
                tool_name = event.get("name", "unknown")
                current_tool = f"Executing: {tool_name}"

                # Show spinner
                spinner_text = Text(f"🔧 {current_tool}", style="yellow")
                live.update(Panel(Markdown(accumulated_text), title="[cyan]Assistant[/cyan]", subtitle=spinner_text))

            elif event_type == "on_tool_end":
                # Tool execution completed
                tool_name = event.get("name", "unknown")
                current_tool = f"✓ Completed: {tool_name}"

                # Update with checkmark
                done_text = Text(f"✓ {tool_name}", style="green")
                live.update(Panel(Markdown(accumulated_text), title="[cyan]Assistant[/cyan]", subtitle=done_text))

                # Clear after a moment
                await asyncio.sleep(0.5)
                current_tool = None

    # Get final state
    final_state = agent.graph.get_state(config_dict)

    # Extract final response
    messages = final_state.values.get("messages", [])
    assistant_messages = []
    for m in messages:
        if hasattr(m, 'type') and m.type == "ai":
            assistant_messages.append(m)

    final_reply = ""
    if assistant_messages:
        last_msg = assistant_messages[-1]
        if hasattr(last_msg, 'content'):
            final_reply = str(last_msg.content)

    return {
        "reply": final_reply or accumulated_text,
        "rag_results": final_state.values.get("rag_results", []),
        "tool_calls": final_state.values.get("tool_results", []),
        "conv_id": final_state.values.get("conv_id"),
    }


def chat_with_streaming_sync(agent, user_message: str, **kwargs):
    """
    Synchronous wrapper for streaming chat.

    Args:
        agent: UnityLangGraphAgent instance
        user_message: User's question
        **kwargs: Additional chat parameters

    Returns:
        Chat result dict
    """
    return asyncio.run(chat_with_streaming(agent, user_message, **kwargs))
