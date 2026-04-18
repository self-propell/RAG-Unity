"""
Tool Execution Node

Executes approved tool calls and adds results to message history.
"""
import json

from core.tools import execute_tool
from core.projects import get_project_manager


def tool_execution_node(state: dict) -> dict:
    """
    Execute approved tool calls.
    Streams progress for long-running operations (Phase 3).
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

    # Execute each tool
    tool_results = []
    tool_result_messages = []

    for tc in state["pending_tool_calls"]:
        # Execute tool
        result = execute_tool(
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

        tool_result_msg = ToolMessage(
            content=json.dumps(result, ensure_ascii=False)[:8000],
            tool_call_id=tc["call_id"]
        )
        tool_result_messages.append(tool_result_msg)

    return {
        "messages": tool_result_messages,
        "tool_results": state.get("tool_results", []) + tool_results,
        "pending_tool_calls": [],
        "tool_call_count": state.get("tool_call_count", 0) + len(tool_results),
    }
