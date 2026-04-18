"""
Planning System for Unity RAG Agent

Implements a Plan → Execute → Reflect workflow using LangGraph.

Features:
- Multi-step task decomposition
- Dynamic plan adjustment
- Execution monitoring
- Reflection and replanning
"""
from typing import TypedDict, Annotated, Sequence, Literal, Optional
from dataclasses import dataclass
from langgraph.graph import StateGraph, add_messages, END


# ────────────────────────────────────────────────
# Planning State Schema
# ────────────────────────────────────────────────

@dataclass
class PlanStep:
    """A single step in the plan"""
    step_id: int
    description: str
    tool: str
    arguments: dict
    status: Literal["pending", "in_progress", "completed", "failed"] = "pending"
    result: Optional[dict] = None
    reasoning: str = ""


class PlanningState(TypedDict):
    """State for planning workflow"""

    # User request
    user_request: str

    # Planning
    plan: list[dict]  # List of PlanStep serialized
    current_step: int

    # Execution
    execution_results: list[dict]

    # Reflection
    reflection: str
    needs_replan: bool

    # Messages (for LLM interaction)
    messages: Annotated[Sequence[dict], add_messages]

    # Context
    project_id: str
    provider_name: str
    model_name: str

    # Control
    max_replans: int
    replan_count: int


# ────────────────────────────────────────────────
# Planning Nodes
# ────────────────────────────────────────────────

def planning_node(state: dict) -> dict:
    """
    Create initial plan based on user request.

    Uses LLM to decompose the task into steps.
    """
    from llm.providers import create_provider

    provider = create_provider(state["provider_name"], state["model_name"])

    planning_prompt = f"""
You are a planning assistant for Unity development tasks.

User Request: {state['user_request']}

Create a detailed step-by-step plan to accomplish this task. For each step:
1. Describe what needs to be done
2. Specify which tool to use
3. Provide the tool arguments

Available tools:
- read_file: Read a file from the project
- search_code: Search for code patterns
- list_files: List files in a directory
- write_file: Write content to a file
- replace_in_file: Replace text in a file
- run_command: Execute a shell command
- rag_search: Search the codebase semantically

Output format (JSON):
{{
  "reasoning": "Why this plan will work...",
  "steps": [
    {{
      "step_id": 1,
      "description": "Read the Player.cs file",
      "tool": "read_file",
      "arguments": {{"path": "Assets/Scripts/Player.cs"}}
    }},
    ...
  ]
}}

Think step by step. Be specific and concrete.
"""

    messages = [{"role": "user", "content": planning_prompt}]
    result = provider.chat(
        system_prompt="You are a helpful planning assistant. Always respond with valid JSON.",
        messages=messages,
        max_tokens=2048
    )

    # Parse plan from LLM response
    import json
    try:
        plan_data = json.loads(result.text)
        steps = plan_data.get("steps", [])
        reasoning = plan_data.get("reasoning", "")

        return {
            "plan": steps,
            "current_step": 0,
            "messages": [{"role": "assistant", "content": result.text}],
            "reflection": f"Initial plan created: {reasoning}"
        }
    except json.JSONDecodeError:
        # Fallback: create a simple plan
        return {
            "plan": [{
                "step_id": 1,
                "description": "Execute user request directly",
                "tool": "rag_search",
                "arguments": {"query": state["user_request"]}
            }],
            "current_step": 0,
            "reflection": "Created fallback plan"
        }


def execution_node(state: dict) -> dict:
    """
    Execute the current step in the plan.
    """
    from core.tools import execute_tool
    from core.projects import get_project_manager

    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)

    if current_step >= len(plan):
        return {"current_step": current_step}

    step = plan[current_step]

    # Get project
    project_id = state.get("project_id")
    project = None
    project_root = ""

    if project_id and project_id != "_global":
        pm = get_project_manager()
        project = pm.get_project(project_id)
        if project:
            project_root = project.path

    # Execute tool
    try:
        result = execute_tool(
            project_root=project_root,
            tool_name=step["tool"],
            arguments=step["arguments"],
            rag_context={
                "db_path": project.db_path if project else "",
                "collection_name": project.collection_name if project else "",
                "project_id": project_id,
            } if project else {}
        )

        # Update step status
        step["status"] = "completed"
        step["result"] = result

        # Record execution result
        execution_results = state.get("execution_results", [])
        execution_results.append({
            "step_id": step["step_id"],
            "tool": step["tool"],
            "result": result,
            "success": not result.get("error")
        })

        return {
            "plan": plan,
            "current_step": current_step + 1,
            "execution_results": execution_results
        }

    except Exception as e:
        # Mark step as failed
        step["status"] = "failed"
        step["result"] = {"error": str(e)}

        execution_results = state.get("execution_results", [])
        execution_results.append({
            "step_id": step["step_id"],
            "tool": step["tool"],
            "result": {"error": str(e)},
            "success": False
        })

        return {
            "plan": plan,
            "current_step": current_step + 1,
            "execution_results": execution_results,
            "needs_replan": True
        }


def reflection_node(state: dict) -> dict:
    """
    Reflect on execution results and decide if replanning is needed.
    """
    from llm.providers import create_provider

    provider = create_provider(state["provider_name"], state["model_name"])

    # Build reflection prompt
    plan_summary = "\n".join([
        f"Step {s['step_id']}: {s['description']} [{s.get('status', 'pending')}]"
        for s in state.get("plan", [])
    ])

    execution_summary = "\n".join([
        f"Step {r['step_id']}: {r['tool']} - {'✓' if r['success'] else '✗'}"
        for r in state.get("execution_results", [])
    ])

    reflection_prompt = f"""
Reflect on the execution of this plan:

Original Request: {state['user_request']}

Plan:
{plan_summary}

Execution Results:
{execution_summary}

Questions:
1. Did the plan achieve the user's goal?
2. Were there any failures that require replanning?
3. Is the task complete, or do we need to continue?

Respond with JSON:
{{
  "reflection": "Your analysis...",
  "task_complete": true/false,
  "needs_replan": true/false,
  "next_steps": "What to do next..."
}}
"""

    messages = [{"role": "user", "content": reflection_prompt}]
    result = provider.chat(
        system_prompt="You are a reflective assistant. Analyze execution results and provide insights.",
        messages=messages,
        max_tokens=1024
    )

    # Parse reflection
    import json
    try:
        reflection_data = json.loads(result.text)

        return {
            "reflection": reflection_data.get("reflection", ""),
            "needs_replan": reflection_data.get("needs_replan", False),
            "messages": state.get("messages", []) + [{"role": "assistant", "content": result.text}]
        }
    except json.JSONDecodeError:
        return {
            "reflection": result.text,
            "needs_replan": False
        }


def replanning_node(state: dict) -> dict:
    """
    Create a new plan based on reflection.
    """
    from llm.providers import create_provider

    provider = create_provider(state["provider_name"], state["model_name"])

    # Build replanning prompt
    previous_plan = "\n".join([
        f"Step {s['step_id']}: {s['description']} [{s.get('status', 'pending')}]"
        for s in state.get("plan", [])
    ])

    replanning_prompt = f"""
The previous plan did not fully achieve the goal. Create a new plan.

Original Request: {state['user_request']}

Previous Plan:
{previous_plan}

Reflection: {state.get('reflection', '')}

Create a NEW plan that addresses the issues. Use the same JSON format as before.
"""

    messages = [{"role": "user", "content": replanning_prompt}]
    result = provider.chat(
        system_prompt="You are a planning assistant. Create improved plans based on feedback.",
        messages=messages,
        max_tokens=2048
    )

    # Parse new plan
    import json
    try:
        plan_data = json.loads(result.text)
        steps = plan_data.get("steps", [])

        return {
            "plan": steps,
            "current_step": 0,
            "execution_results": [],
            "replan_count": state.get("replan_count", 0) + 1,
            "needs_replan": False
        }
    except json.JSONDecodeError:
        # Keep old plan if parsing fails
        return {
            "needs_replan": False
        }


# ────────────────────────────────────────────────
# Conditional Edges
# ────────────────────────────────────────────────

def should_continue_execution(state: dict) -> Literal["execution", "reflection"]:
    """Check if there are more steps to execute"""
    current_step = state.get("current_step", 0)
    plan = state.get("plan", [])

    if current_step < len(plan):
        return "execution"
    return "reflection"


def should_replan(state: dict) -> Literal["replanning", "end"]:
    """Check if replanning is needed"""
    needs_replan = state.get("needs_replan", False)
    replan_count = state.get("replan_count", 0)
    max_replans = state.get("max_replans", 2)

    if needs_replan and replan_count < max_replans:
        return "replanning"
    return "end"


# ────────────────────────────────────────────────
# Planning Graph
# ────────────────────────────────────────────────

def create_planning_graph() -> StateGraph:
    """Create the planning workflow graph"""

    workflow = StateGraph(PlanningState)

    # Add nodes
    workflow.add_node("planning", planning_node)
    workflow.add_node("execution", execution_node)
    workflow.add_node("reflection", reflection_node)
    workflow.add_node("replanning", replanning_node)

    # Add edges
    workflow.set_entry_point("planning")
    workflow.add_edge("planning", "execution")

    # Conditional: continue execution or reflect?
    workflow.add_conditional_edges(
        "execution",
        should_continue_execution,
        {
            "execution": "execution",
            "reflection": "reflection"
        }
    )

    # Conditional: replan or end?
    workflow.add_conditional_edges(
        "reflection",
        should_replan,
        {
            "replanning": "replanning",
            "end": END
        }
    )

    # After replanning, go back to execution
    workflow.add_edge("replanning", "execution")

    return workflow.compile()


# ────────────────────────────────────────────────
# Planning Agent
# ────────────────────────────────────────────────

class PlanningAgent:
    """Agent with planning capabilities"""

    def __init__(self, project=None, provider=None):
        self.project = project
        self.provider = provider
        self.graph = create_planning_graph()

    def plan_and_execute(self, user_request: str, max_replans: int = 2) -> dict:
        """
        Create a plan and execute it.

        Args:
            user_request: User's task description
            max_replans: Maximum number of replanning attempts

        Returns:
            Final state with plan, execution results, and reflection
        """
        initial_state = {
            "user_request": user_request,
            "plan": [],
            "current_step": 0,
            "execution_results": [],
            "reflection": "",
            "needs_replan": False,
            "messages": [],
            "project_id": self.project.project_id if self.project else "_global",
            "provider_name": self.provider.provider_name(),
            "model_name": self.provider.model_name(),
            "max_replans": max_replans,
            "replan_count": 0
        }

        final_state = self.graph.invoke(initial_state)

        return {
            "plan": final_state.get("plan", []),
            "execution_results": final_state.get("execution_results", []),
            "reflection": final_state.get("reflection", ""),
            "replan_count": final_state.get("replan_count", 0),
            "success": not final_state.get("needs_replan", False)
        }
