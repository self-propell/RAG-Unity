"""
Approval Gate Node

Checks if pending tool calls need human approval.
Pauses graph execution if approval required.
"""
from llm.approval import check_needs_approval, format_approval_request, prompt_user_approval


def approval_gate_node(state: dict) -> dict:
    """
    Check if pending tool calls need approval.
    Returns interrupt signal if approval required.
    """
    if not state.get("pending_tool_calls"):
        return {"pending_approval": False}

    # YOLO mode: approve everything after first approval
    if state.get("yolo_active"):
        return {"pending_approval": False}

    # Auto mode: no approval needed
    if state.get("approval_mode") == "auto":
        return {"pending_approval": False}

    # Check which tools need approval
    approval_config = state.get("approval_config", {})
    pending_calls = state["pending_tool_calls"]

    needs_approval = check_needs_approval(
        pending_calls,
        approval_config,
        state.get("yolo_active", False)
    )

    if not needs_approval:
        return {"pending_approval": False}

    # Approval required - prompt user
    approved, yolo_activated = prompt_user_approval(pending_calls)

    if not approved:
        # Rejected - clear pending tool calls
        print("\n[REJECTED] Tool calls skipped.\n")
        return {
            "pending_tool_calls": [],
            "pending_approval": False,
        }

    # Approved
    print("\n[APPROVED] Executing tools...\n")
    return {
        "pending_approval": False,
        "yolo_active": yolo_activated or state.get("yolo_active", False),
    }
