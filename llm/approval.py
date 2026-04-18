"""
Approval System for LangGraph Agent

Handles human-in-the-loop approval for dangerous operations.
Supports: auto, batch, yolo modes
"""
from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class ApprovalConfig:
    """Configuration for tool approval"""
    mode: Literal["auto", "batch", "yolo"] = "batch"
    tools: dict[str, bool] = None  # {tool_name: needs_approval}

    def __post_init__(self):
        if self.tools is None:
            self.tools = get_default_approval_config()

    def needs_approval(self, tool_name: str) -> bool:
        """Check if a tool needs approval"""
        return self.tools.get(tool_name, False)


def get_default_approval_config() -> dict[str, bool]:
    """Get default approval configuration"""
    return {
        # Dangerous operations (require approval by default)
        "run_command": True,
        "rag_rebuild": True,

        # File operations (configurable per project)
        "write_file": False,
        "replace_in_file": False,

        # Safe operations (never require approval)
        "read_file": False,
        "list_files": False,
        "search_code": False,
        "rag_search": False,
        "rag_update": False,
        "rag_stats": False,
        "diff_file": False,
        "use_skill": False,
        "list_skills": False,
    }


def check_needs_approval(
    tool_calls: list[dict],
    approval_config: dict[str, bool],
    yolo_active: bool = False
) -> bool:
    """
    Check if any tool calls need approval.

    Args:
        tool_calls: List of pending tool calls
        approval_config: Approval configuration
        yolo_active: If True, skip all approvals

    Returns:
        True if approval needed, False otherwise
    """
    if yolo_active:
        return False

    return any(
        approval_config.get(tc["name"], False)
        for tc in tool_calls
    )


def format_approval_request(tool_calls: list[dict]) -> str:
    """
    Format tool calls for approval prompt.

    Returns formatted string for display.
    """
    lines = ["Tool calls pending approval:"]
    for i, tc in enumerate(tool_calls, 1):
        args_str = str(tc.get("arguments", {}))
        if len(args_str) > 100:
            args_str = args_str[:97] + "..."
        lines.append(f"  {i}. {tc['name']}({args_str})")

    return "\n".join(lines)


def prompt_user_approval(tool_calls: list[dict]) -> tuple[bool, bool]:
    """
    Prompt user for approval (CLI version).

    Returns:
        (approved: bool, yolo_activated: bool)
    """
    print("\n" + "="*60)
    print(format_approval_request(tool_calls))
    print("="*60)
    print("\nOptions:")
    print("  y     - Approve this batch")
    print("  n     - Reject (skip these tools)")
    print("  yolo  - Approve and enable YOLO mode (auto-approve all future)")
    print("  c     - Configure approval settings")

    while True:
        choice = input("\nApprove? [y/n/yolo/c]: ").strip().lower()

        if choice == "y":
            return True, False
        elif choice == "n":
            return False, False
        elif choice == "yolo":
            print("\n[YOLO MODE ACTIVATED] All future tool calls will be auto-approved.")
            return True, True
        elif choice == "c":
            print("\nConfiguration editing not yet implemented.")
            print("Edit approval_config in projects.json manually.")
            continue
        else:
            print("Invalid choice. Please enter y, n, yolo, or c.")


def format_approval_request_html(tool_calls: list[dict]) -> dict:
    """
    Format tool calls for Dashboard approval modal.

    Returns dict with HTML-ready data.
    """
    return {
        "tool_calls": [
            {
                "name": tc["name"],
                "arguments": tc.get("arguments", {}),
                "display": f"{tc['name']}({str(tc.get('arguments', {}))})"
            }
            for tc in tool_calls
        ],
        "count": len(tool_calls)
    }
