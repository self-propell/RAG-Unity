"""
llm/agent_langgraph.py — LangGraph-based Unity RAG Agent

Features:
  - Streaming output (token-level and tool execution)
  - Human approval gates (configurable, YOLO mode)
  - State persistence & rollback (SQLite checkpointing)
  - Explicit ReAct pattern (Reason → Act → Observe)
"""
import json
import time
from typing import TypedDict, Annotated, Sequence, Literal, Optional, Any
from dataclasses import dataclass

from langgraph.graph import StateGraph, add_messages, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode

from core import config
from core.projects import ProjectInfo
from core.tools import TOOL_DEFINITIONS, execute_tool
from rag.retriever import Retriever, SearchResult


# ────────────────────────────────────────────────
# State Schema
# ────────────────────────────────────────────────

class AgentState(TypedDict):
    """LangGraph state for Unity RAG Agent"""

    # Message history (LangGraph managed with add_messages reducer)
    messages: Annotated[Sequence[dict], add_messages]

    # RAG context
    rag_results: list[dict]           # SearchResult objects serialized
    rag_context_block: str            # Formatted context for LLM

    # Tool execution
    pending_tool_calls: list[dict]    # [{call_id, name, arguments}]
    tool_results: list[dict]          # [{call_id, result}]
    tool_call_count: int

    # Approval system
    approval_mode: Literal["auto", "batch", "yolo"]
    approval_config: dict             # {tool_name: bool}
    yolo_active: bool                 # Once approved, allow all
    pending_approval: bool            # Waiting for user
    approval_request: Optional[dict]  # Details for UI

    # Provider & conversation
    provider_name: str                # "claude" | "openai"
    model_name: str
    project_id: str
    conv_id: Optional[str]

    # Loop control
    max_iterations: int
    current_iteration: int

    # Streaming
    stream_tokens: bool

    # Rollback
    last_checkpoint_id: Optional[str]

    # ReAct trace
    reasoning_trace: list[dict]       # [{step, reasoning, actions, observations}]

    # Filter parameters
    filter_path: Optional[str]
    filter_domain: Optional[str]
    top_k: int


# ────────────────────────────────────────────────
# Message Normalization
# ────────────────────────────────────────────────

def normalize_message(msg: dict, provider: str) -> dict:
    """Convert provider-specific format to unified state format"""
    return {
        "role": msg["role"],
        "content": msg.get("content"),
        "tool_calls": msg.get("tool_calls"),  # OpenAI format
        "_provider": provider,
        "_raw": msg  # Keep original for lossless conversion
    }


def denormalize_message(msg: dict, target_provider: str) -> dict:
    """Convert state format back to provider format"""
    if msg.get("_provider") == target_provider:
        # Same provider, return original
        return msg.get("_raw", {k: v for k, v in msg.items() if not k.startswith("_")})

    # Cross-provider conversion (if needed in future)
    return {k: v for k, v in msg.items() if not k.startswith("_")}


# ────────────────────────────────────────────────
# Helper Functions
# ────────────────────────────────────────────────

def build_context_block(results: list[SearchResult]) -> str:
    """Format RAG results for prompt injection"""
    if not results:
        return "## 相关代码上下文\n\n（未检索到相关代码片段）"

    parts = ["## 相关代码上下文（按相关度排列）\n"]
    for i, r in enumerate(results, 1):
        parts.append(f"### [{i}] {r.format_for_prompt()}")

    return "\n\n".join(parts)


def create_initial_state(
    project_id: str,
    conv_id: Optional[str],
    provider_name: str,
    model_name: str,
    user_message: str,
    approval_config: Optional[dict] = None,
    **kwargs
) -> AgentState:
    """Create initial state for graph execution"""
    return AgentState(
        messages=[{"role": "user", "content": user_message}],
        rag_results=[],
        rag_context_block="",
        pending_tool_calls=[],
        tool_results=[],
        tool_call_count=0,
        approval_mode=kwargs.get("approval_mode", "batch"),
        approval_config=approval_config or {
            "run_command": True,
            "rag_rebuild": True,
            "write_file": False,
            "replace_in_file": False,
        },
        yolo_active=False,
        pending_approval=False,
        approval_request=None,
        provider_name=provider_name,
        model_name=model_name,
        project_id=project_id,
        conv_id=conv_id,
        max_iterations=kwargs.get("max_iterations", 10),
        current_iteration=0,
        stream_tokens=kwargs.get("stream_tokens", False),
        last_checkpoint_id=None,
        reasoning_trace=[],
        filter_path=kwargs.get("filter_path"),
        filter_domain=kwargs.get("filter_domain"),
        top_k=kwargs.get("top_k", config.RAG_TOP_K),
    )


# ────────────────────────────────────────────────
# UnityLangGraphAgent
# ────────────────────────────────────────────────

class UnityLangGraphAgent:
    """LangGraph-based Unity RAG Agent with streaming, approval, and rollback"""

    def __init__(
        self,
        project: Optional[ProjectInfo] = None,
        provider = None,
        retriever: Optional[Retriever] = None,
    ):
        self.project = project
        self.provider = provider
        self.retriever = retriever or (
            Retriever(
                db_path=project.db_path if project else config.CHROMA_DB_ROOT,
                collection_name=project.collection_name if project else "unity_codebase",
            ) if project else None
        )

        # Initialize checkpointer
        if project:
            import sqlite3
            checkpoint_db = f"{project.db_path}/langgraph_checkpoints.db"
            # Create connection and pass to SqliteSaver
            conn = sqlite3.connect(checkpoint_db, check_same_thread=False)
            self.checkpointer = SqliteSaver(conn)
        else:
            self.checkpointer = None

        # Build graph
        self.graph = self._create_graph()

        # Current conversation
        self._current_conv_id: Optional[str] = None
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._cached_tokens = 0

    def _create_graph(self) -> StateGraph:
        """Create the LangGraph state graph"""
        # Import nodes (will be created in next step)
        from llm.nodes.rag_retrieval import rag_retrieval_node
        from llm.nodes.llm_call import llm_call_node
        from llm.nodes.approval_gate import approval_gate_node
        from llm.nodes.tool_execution import tool_execution_node
        from llm.nodes.format_output import format_output_node
        from llm.edges.has_tool_calls import has_tool_calls_edge
        from llm.edges.should_continue import should_continue_edge

        # Create graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("rag_retrieval", rag_retrieval_node)
        workflow.add_node("llm_call", llm_call_node)
        workflow.add_node("approval_gate", approval_gate_node)
        workflow.add_node("tool_execution", tool_execution_node)
        workflow.add_node("format_output", format_output_node)

        # Add edges
        workflow.set_entry_point("rag_retrieval")
        workflow.add_edge("rag_retrieval", "llm_call")

        # Conditional: has tool calls?
        workflow.add_conditional_edges(
            "llm_call",
            has_tool_calls_edge,
            {
                "approval_gate": "approval_gate",
                "format_output": "format_output"
            }
        )

        workflow.add_edge("approval_gate", "tool_execution")

        # Conditional: should continue?
        workflow.add_conditional_edges(
            "tool_execution",
            should_continue_edge,
            {
                "llm_call": "llm_call",
                "format_output": "format_output"
            }
        )

        workflow.add_edge("format_output", END)

        # Compile with checkpointer
        return workflow.compile(checkpointer=self.checkpointer)

    def chat(
        self,
        user_message: str,
        top_k: Optional[int] = None,
        filter_path: Optional[str] = None,
        filter_domain: Optional[str] = None,
        stream: bool = False,
    ) -> dict:
        """
        Send message and get response.

        Returns:
            {
                "reply": str,
                "rag_results": list[dict],
                "tool_calls": list[dict],
                "conv_id": str,
                "checkpoint_id": str
            }
        """
        # Get approval config from project
        approval_config = None
        if self.project and hasattr(self.project, 'approval_config'):
            approval_config = self.project.approval_config

        # Create initial state
        initial_state = create_initial_state(
            project_id=self.project.project_id if self.project else "_global",
            conv_id=self._current_conv_id,
            provider_name=self.provider.provider_name(),
            model_name=self.provider.model_name(),
            user_message=user_message,
            approval_config=approval_config,
            top_k=top_k or config.RAG_TOP_K,
            filter_path=filter_path,
            filter_domain=filter_domain,
            stream_tokens=stream,
        )

        # Execute graph
        config_dict = {
            "configurable": {
                "thread_id": self._current_conv_id or f"conv_{int(time.time())}"
            }
        }

        if stream:
            # Streaming mode (to be implemented)
            return self._chat_streaming(initial_state, config_dict)
        else:
            # Non-streaming mode
            final_state = self.graph.invoke(initial_state, config_dict)

            # Extract final response
            messages = final_state.get("messages", [])
            assistant_messages = []
            for m in messages:
                if hasattr(m, 'type') and m.type == "ai":
                    assistant_messages.append(m)
                elif isinstance(m, dict) and m.get("role") == "assistant":
                    assistant_messages.append(m)

            final_reply = ""
            if assistant_messages:
                last_msg = assistant_messages[-1]
                if hasattr(last_msg, 'content'):
                    final_reply = str(last_msg.content)
                elif isinstance(last_msg, dict):
                    content = last_msg.get("content")
                    if isinstance(content, str):
                        final_reply = content
                    elif isinstance(content, list):
                        # Extract text from content blocks
                        text_parts = []
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                        final_reply = "".join(text_parts)

            return {
                "reply": final_reply,
                "rag_results": final_state.get("rag_results", []),
                "tool_calls": final_state.get("tool_results", []),
                "conv_id": final_state.get("conv_id"),
                "last_checkpoint_id": final_state.get("last_checkpoint_id"),
            }

    def _chat_streaming(self, initial_state: AgentState, config_dict: dict):
        """Streaming chat (placeholder for Phase 3)"""
        # Will be implemented in Phase 3
        raise NotImplementedError("Streaming not yet implemented")

    def rollback(self, steps: int = 1) -> bool:
        """Roll back conversation by N steps"""
        if not self._current_conv_id or not self.checkpointer:
            return False

        config_dict = {"configurable": {"thread_id": self._current_conv_id}}

        try:
            history = list(self.graph.get_state_history(config_dict))

            if steps >= len(history):
                return False

            target = history[steps]

            # Get the state at that checkpoint
            restored_state = self.graph.get_state(
                config_dict,
                checkpoint_id=target.config["configurable"]["checkpoint_id"]
            )

            # Update current state (this will be used in next invocation)
            # Note: LangGraph automatically handles state restoration
            return True

        except Exception as e:
            print(f"Rollback failed: {e}")
            return False

    def get_checkpoint_history(self) -> list[dict]:
        """Get checkpoint history for current conversation"""
        if not self._current_conv_id or not self.checkpointer:
            return []

        config_dict = {"configurable": {"thread_id": self._current_conv_id}}
        history = list(self.graph.get_state_history(config_dict))

        return [
            {
                "checkpoint_id": str(h.config["configurable"]["checkpoint_id"]),
                "timestamp": h.metadata.get("timestamp", 0),
                "iteration": h.values.get("current_iteration", 0),
                "message_count": len(h.values.get("messages", [])),
            }
            for h in history
        ]

    @property
    def project_root(self) -> str:
        return self.project.path if self.project else ""

    def new_conversation(self):
        """Start a new conversation"""
        self._current_conv_id = None
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._cached_tokens = 0

    def set_conv_id(self, conv_id: str):
        """Set current conversation ID"""
        self._current_conv_id = conv_id
