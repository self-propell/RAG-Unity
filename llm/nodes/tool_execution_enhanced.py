"""
Enhanced Tool Execution Node with error handling

Uses safe tool execution wrapper with timeout, retry, and loop detection.
"""
import json

from core.tool_execution_safe import execute_tool_safe
from core.projects import get_project_manager


def tool_execution_node(state: dict) -> dict:
    """
    Execute approved tool calls with enhanced error handling.

    Features:
    - Timeout protection
    - Automatic retry
    - Error loop detection
    - Fallback strategies
    """
    if not state.get("pending_tool_calls"):
        return {}

    # Get project
    project_id = state.get("project_id")
    project = None
    project_root = ""

    if project_id and project_id != "_global":
        pm = get_project_manager()
        project = pm.get_project(project_id)
        if project:
            project_root = project.path

    # Execute each tool with safety wrapper
    tool_results = []
    tool_result_messages = []

    for tc in state["pending_tool_calls"]:
        # Execute tool safely
        result = execute_tool_safe(
            project_root=project_root,
            tool_name=tc["name"],
            arguments=tc["arguments"],
            rag_context={
                "db_path": project.db_path if project else "",
                "collection_name": project.collection_name if project else "",
                "project_id": project_id,
            } if project else {}
        )

        tool_results.append({
            "call_id": tc["call_id"],
            "tool": tc["name"],
            "arguments": tc["arguments"],
            "result": result,
        })

        # Format tool result message for provider (as LangGraph Message)
        from langchain_core.messages import ToolMessage

        # Truncate large results
        result_str = json.dumps(result, ensure_ascii=False)
        if len(result_str) > 8000:
            result_str = result_str[:7900] + "\n... (truncated)"

        tool_result_msg = ToolMessage(
            content=result_str,
            tool_call_id=tc["call_id"]
        )
        tool_result_messages.append(tool_result_msg)

    return {
        "messages": tool_result_messages,
        "tool_results": state.get("tool_results", []) + tool_results,
        "pending_tool_calls": [],
        "tool_call_count": state.get("tool_call_count", 0) + len(tool_results),
    }
