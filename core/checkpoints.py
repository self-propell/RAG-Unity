"""
Checkpoint management for LangGraph conversations

Provides utilities for managing conversation checkpoints and rollback.
"""
import json
import time
from typing import Optional
from pathlib import Path


class CheckpointManager:
    """Manages conversation checkpoints"""

    def __init__(self, project_id: str, checkpoints_dir: str = "./conversations"):
        self.project_id = project_id
        self.checkpoints_dir = Path(checkpoints_dir) / project_id
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint_metadata(self, conv_id: str, checkpoint_id: str, metadata: dict):
        """Save checkpoint metadata to JSON"""
        checkpoint_file = self.checkpoints_dir / f"{conv_id}_checkpoints.json"

        # Load existing checkpoints
        if checkpoint_file.exists():
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {"conv_id": conv_id, "checkpoints": []}

        # Add new checkpoint
        data["checkpoints"].append({
            "checkpoint_id": checkpoint_id,
            "timestamp": metadata.get("timestamp", time.time()),
            "iteration": metadata.get("iteration", 0),
            "message_count": metadata.get("message_count", 0),
            "tool_calls": metadata.get("tool_calls", 0),
            "user_message_preview": metadata.get("user_message_preview", ""),
        })

        # Save
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_checkpoint_history(self, conv_id: str) -> list[dict]:
        """Get checkpoint history for a conversation"""
        checkpoint_file = self.checkpoints_dir / f"{conv_id}_checkpoints.json"

        if not checkpoint_file.exists():
            return []

        with open(checkpoint_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data.get("checkpoints", [])

    def cleanup_old_checkpoints(self, conv_id: str, keep_last_n: int = 50):
        """Keep only the last N checkpoints"""
        checkpoint_file = self.checkpoints_dir / f"{conv_id}_checkpoints.json"

        if not checkpoint_file.exists():
            return

        with open(checkpoint_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        checkpoints = data.get("checkpoints", [])
        if len(checkpoints) > keep_last_n:
            data["checkpoints"] = checkpoints[-keep_last_n:]

            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
