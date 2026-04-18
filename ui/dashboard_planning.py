"""
Dashboard API endpoints for Planning Agent
"""
from flask import jsonify, request


def add_planning_routes(app):
    """Add planning-related routes to Flask app"""

    @app.route("/api/planning/create", methods=["POST"])
    def api_planning_create():
        """Create a plan for a task"""
        data = request.json
        project_id = data.get("project_id")
        task = data.get("task", "").strip()
        model = data.get("model")

        if not task:
            return jsonify({"error": "Task description required"}), 400

        try:
            from core.projects import get_project_manager
            from llm.providers import create_provider
            from llm.planning import PlanningAgent

            pm = get_project_manager()
            project = pm.get_project(project_id)

            if not project:
                return jsonify({"error": "Project not found"}), 404

            provider = create_provider(model=model)
            agent = PlanningAgent(project=project, provider=provider)

            # Create plan (just planning, no execution)
            from llm.planning import planning_node

            initial_state = {
                "user_request": task,
                "plan": [],
                "messages": [],
                "project_id": project_id,
                "provider_name": provider.provider_name(),
                "model_name": provider.model_name()
            }

            result_state = planning_node(initial_state)

            return jsonify({
                "plan": {
                    "steps": result_state.get("plan", []),
                    "reasoning": result_state.get("reflection", "")
                }
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    @app.route("/api/planning/execute", methods=["POST"])
    def api_planning_execute():
        """Execute a plan"""
        data = request.json
        project_id = data.get("project_id")
        plan = data.get("plan")

        if not plan:
            return jsonify({"error": "Plan required"}), 400

        try:
            from core.projects import get_project_manager
            from llm.providers import create_provider
            from llm.planning import PlanningAgent

            pm = get_project_manager()
            project = pm.get_project(project_id)

            if not project:
                return jsonify({"error": "Project not found"}), 404

            provider = create_provider()
            agent = PlanningAgent(project=project, provider=provider)

            # Execute the plan
            result = agent.plan_and_execute(
                user_request=data.get("task", "Execute plan"),
                max_replans=0  # Don't replan during execution
            )

            return jsonify({
                "execution_results": result.get("execution_results", []),
                "reflection": result.get("reflection", ""),
                "needs_replan": not result.get("success", False),
                "replan_count": result.get("replan_count", 0),
                "success": result.get("success", False)
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    @app.route("/api/planning/replan", methods=["POST"])
    def api_planning_replan():
        """Create a new plan based on reflection"""
        data = request.json
        project_id = data.get("project_id")
        previous_plan = data.get("previous_plan")
        reflection = data.get("reflection", "")

        try:
            from core.projects import get_project_manager
            from llm.providers import create_provider
            from llm.planning import replanning_node

            pm = get_project_manager()
            project = pm.get_project(project_id)

            if not project:
                return jsonify({"error": "Project not found"}), 404

            provider = create_provider()

            # Create new plan
            state = {
                "user_request": data.get("task", ""),
                "plan": previous_plan.get("steps", []),
                "reflection": reflection,
                "execution_results": data.get("execution_results", []),
                "project_id": project_id,
                "provider_name": provider.provider_name(),
                "model_name": provider.model_name(),
                "replan_count": data.get("replan_count", 0)
            }

            result_state = replanning_node(state)

            return jsonify({
                "plan": {
                    "steps": result_state.get("plan", []),
                    "reasoning": "Replanned based on previous execution"
                }
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500
