"""
Should Continue Edge

Conditional edge that decides whether to continue tool loop or finish.
"""
from typing import Literal


def should_continue_edge(state: dict) -> Literal["llm_call", "format_output"]:
    """
    Decide whether to continue tool loop or finish.

    Returns:
        "llm_call" to continue the loop
        "format_output" to end the conversation
    """
    # Check max iterations
    if state.get("current_iteration", 0) >= state.get("max_iterations", 10):
        return "format_output"

    # If there are still pending tool calls, something went wrong
    # (they should have been cleared by tool_execution_node)
    if state.get("pending_tool_calls"):
        return "format_output"

    # Continue loop - LLM will decide if more tools are needed
    return "llm_call"
