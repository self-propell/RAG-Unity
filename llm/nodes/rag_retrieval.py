"""
RAG Retrieval Node

Performs semantic search on the first user message and injects context.
"""
from core.projects import get_project_manager
from rag.retriever import Retriever


def rag_retrieval_node(state: dict) -> dict:
    """
    Retrieve relevant code chunks based on user query.
    Only runs on first iteration (when rag_results is empty).
    """
    # Skip if already retrieved
    if state.get("rag_results"):
        return {}

    # Get last user message
    messages = state.get("messages", [])
    if not messages:
        return {}

    # Handle both dict and LangGraph Message objects
    user_messages = []
    for m in messages:
        if hasattr(m, 'type') and m.type == "human":
            # LangGraph HumanMessage object
            user_messages.append({"role": "user", "content": m.content})
        elif isinstance(m, dict) and m.get("role") == "user":
            # Plain dict
            user_messages.append(m)

    if not user_messages:
        return {}

    last_user_msg = user_messages[-1].get("content", "")
    if not last_user_msg or not isinstance(last_user_msg, str):
        return {}

    # Get retriever for project
    project_id = state.get("project_id")
    if not project_id or project_id == "_global":
        return {}

    pm = get_project_manager()
    project = pm.get_project(project_id)
    if not project:
        return {}

    retriever = Retriever(
        db_path=project.db_path,
        collection_name=project.collection_name,
    )

    # Perform RAG search
    results = retriever.search(
        query=last_user_msg,
        top_k=state.get("top_k", 6),
        filter_path=state.get("filter_path"),
        filter_domain=state.get("filter_domain"),
    )

    # Build context block
    if not results:
        context_block = "## 相关代码上下文\n\n（未检索到相关代码片段）"
    else:
        parts = ["## 相关代码上下文（按相关度排列）\n"]
        for i, r in enumerate(results, 1):
            parts.append(f"### [{i}] {r.format_for_prompt()}")
        context_block = "\n\n".join(parts)

    # Inject context into user message
    enhanced_msg = f"{context_block}\n\n## 问题\n\n{last_user_msg}"

    # Update the last user message
    # Need to handle LangGraph Message objects
    from langchain_core.messages import HumanMessage

    messages = list(state["messages"])
    # Replace last message with enhanced version
    messages[-1] = HumanMessage(content=enhanced_msg)

    return {
        "messages": messages,
        "rag_results": [
            {
                "path": r.relative_path,
                "class_name": r.class_name,
                "method_name": r.method_name,
                "chunk_type": r.chunk_type,
                "score": r.score,
                "text": r.text[:500],
                "collection_domain": r.collection_domain,
            }
            for r in results
        ],
        "rag_context_block": context_block,
    }
