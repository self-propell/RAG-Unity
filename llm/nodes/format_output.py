"""
Format Output Node

Final node that prepares the response for return.
"""


def format_output_node(state: dict) -> dict:
    """
    Format final output.
    This is a simple passthrough for now.
    """
    # Extract final assistant message
    messages = state.get("messages", [])
    if not messages:
        return {}

    assistant_messages = []
    for m in messages:
        if hasattr(m, 'type') and m.type == "ai":
            assistant_messages.append(m)
        elif isinstance(m, dict) and m.get("role") == "assistant":
            assistant_messages.append(m)

    if not assistant_messages:
        return {}

    # No state changes, just validation
    return {}
