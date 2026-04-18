"""
Enhanced LLM Call Node with error handling

Uses safe LLM call wrapper with timeout and retry.
"""
from llm.providers import create_provider
from llm.agent import build_system_prompt
from core.projects import get_project_manager
from core.llm_call_safe import call_llm_safe
from rag.retriever import Retriever


def llm_call_node(state: dict) -> dict:
    """
    Call LLM with current message history.
    Enhanced with timeout, retry, and error handling.
    """
    # Get provider
    provider = create_provider(
        provider_type=state["provider_name"],
        model=state["model_name"]
    )

    # Build system prompt
    project_id = state.get("project_id")
    project = None
    retriever = None

    if project_id and project_id != "_global":
        pm = get_project_manager()
        project = pm.get_project(project_id)
        if project:
            retriever = Retriever(
                db_path=project.db_path,
                collection_name=project.collection_name,
            )

    system_prompt = build_system_prompt(retriever, project)

    # Get messages (convert LangGraph Message objects to dicts)
    messages = []
    for m in state["messages"]:
        if hasattr(m, 'type'):
            # LangGraph Message object
            if m.type == "human":
                messages.append({"role": "user", "content": m.content})
            elif m.type == "ai":
                messages.append({"role": "assistant", "content": m.content})
            elif m.type == "tool":
                messages.append({"role": "tool", "content": m.content, "tool_call_id": getattr(m, 'tool_call_id', '')})
        elif isinstance(m, dict):
            # Already a dict
            messages.append({k: v for k, v in m.items() if not k.startswith("_")})
        else:
            # Unknown format, skip
            continue

    # Call LLM with safety wrapper
    from core.tools import TOOL_DEFINITIONS

    success, result = call_llm_safe(
        provider=provider,
        system_prompt=system_prompt,
        messages=messages,
        tools=TOOL_DEFINITIONS,
        max_tokens=4096,
        timeout=state.get("llm_timeout", 60)
    )

    # Handle failure
    if not success:
        # result is error message string
        from langchain_core.messages import AIMessage
        return {
            "messages": [AIMessage(content=result)],
            "pending_tool_calls": [],
            "current_iteration": state["current_iteration"] + 1,
            "llm_error": True,
        }

    # Format assistant message as LangGraph Message
    from langchain_core.messages import AIMessage

    provider_name = state["provider_name"]
    if provider_name == "claude":
        # Claude format: content blocks
        content = []
        if result.text:
            content.append({"type": "text", "text": result.text})
        for tc in result.tool_calls:
            content.append({
                "type": "tool_use",
                "id": tc.call_id,
                "name": tc.name,
                "input": tc.arguments,
            })
        assistant_msg = AIMessage(content=str(content))
    else:
        # OpenAI format
        assistant_msg = AIMessage(
            content=result.text or "",
            additional_kwargs={
                "tool_calls": [
                    {
                        "id": tc.call_id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                    }
                    for tc in result.tool_calls
                ] if result.tool_calls else []
            }
        )

    # Extract pending tool calls
    pending_tool_calls = [
        {
            "call_id": tc.call_id,
            "name": tc.name,
            "arguments": tc.arguments,
        }
        for tc in result.tool_calls
    ]

    return {
        "messages": [assistant_msg],
        "pending_tool_calls": pending_tool_calls,
        "current_iteration": state["current_iteration"] + 1,
    }
