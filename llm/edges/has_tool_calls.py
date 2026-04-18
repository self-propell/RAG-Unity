"""
Has Tool Calls Edge

Conditional edge that routes based on whether LLM returned tool calls.
"""
from typing import Literal


def has_tool_calls_edge(state: dict) -> Literal["approval_gate", "format_output"]:
    """
    Route based on whether LLM returned tool calls.

    Returns:
        "approval_gate" if there are pending tool calls
        "format_output" if no tool calls (conversation complete)
    """
    if state.get("pending_tool_calls"):
        return "approval_gate"
    return "format_output"
