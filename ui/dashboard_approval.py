"""
Dashboard API endpoints for approval system
"""
from flask import jsonify, request

# Global state for pending approvals
_pending_approvals = {}  # {session_id: {tool_calls, callback}}


def add_approval_routes(app):
    """Add approval-related routes to Flask app"""

    @app.route("/api/approval/pending", methods=["GET"])
    def api_approval_pending():
        """Check if there are pending approvals"""
        session_id = request.args.get("session_id", "default")
        pending = _pending_approvals.get(session_id)

        if pending:
            return jsonify({
                "has_pending": True,
                "tool_calls": pending["tool_calls"],
                "count": len(pending["tool_calls"])
            })
        else:
            return jsonify({"has_pending": False})

    @app.route("/api/approval/respond", methods=["POST"])
    def api_approval_respond():
        """Respond to approval request"""
        data = request.json
        session_id = data.get("session_id", "default")
        action = data.get("action")  # "approve", "reject", "yolo"

        pending = _pending_approvals.get(session_id)
        if not pending:
            return jsonify({"error": "No pending approval"}), 404

        # Call the callback with user's decision
        callback = pending.get("callback")
        if callback:
            if action == "approve":
                callback(approved=True, yolo=False)
            elif action == "yolo":
                callback(approved=True, yolo=True)
            else:
                callback(approved=False, yolo=False)

        # Clear pending approval
        del _pending_approvals[session_id]

        return jsonify({"success": True, "action": action})

    @app.route("/api/approval/config", methods=["GET", "POST"])
    def api_approval_config():
        """Get or update approval configuration"""
        project_id = request.args.get("project_id") or request.json.get("project_id")

        if request.method == "GET":
            # Get current config
            from core.projects import get_project_manager
            pm = get_project_manager()
            project = pm.get_project(project_id)

            if not project:
                return jsonify({"error": "Project not found"}), 404

            config = getattr(project, 'approval_config', {
                "mode": "batch",
                "tools": {
                    "run_command": True,
                    "rag_rebuild": True,
                }
            })

            return jsonify({"approval_config": config})

        else:  # POST
            # Update config
            new_config = request.json.get("approval_config")

            from core.projects import get_project_manager
            pm = get_project_manager()
            project = pm.get_project(project_id)

            if not project:
                return jsonify({"error": "Project not found"}), 404

            # Update project config
            project.approval_config = new_config

            # Save to projects.json
            pm.save_projects()

            return jsonify({"success": True, "approval_config": new_config})


def register_pending_approval(session_id: str, tool_calls: list, callback):
    """Register a pending approval request"""
    _pending_approvals[session_id] = {
        "tool_calls": tool_calls,
        "callback": callback
    }
