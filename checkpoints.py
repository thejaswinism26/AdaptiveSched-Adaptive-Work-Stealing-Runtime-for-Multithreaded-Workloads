"""Agent checkpointing mechanism.

Saves and restores agent working memory and task state to a dedicated directory.
Stored independently from the SQLite effect ledger so checkpoint rollbacks revert
agent memory without wiping the durable effect history.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from recoveryguard.models import CheckpointMetadata

DEFAULT_CHECKPOINT_DIR = Path("checkpoints_storage")


class CheckpointManager:
    """Manages file-based agent state snapshots."""

    def __init__(self, storage_dir: Optional[Union[str, Path]] = None) -> None:
        self.storage_dir = Path(storage_dir) if storage_dir else DEFAULT_CHECKPOINT_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def create_checkpoint(
        self,
        task_id: str,
        step_index: int,
        agent_state: Dict[str, Any],
        checkpoint_id: Optional[str] = None,
    ) -> CheckpointMetadata:
        """Serializes and saves agent working memory to disk."""
        now = time.time()
        task_tag = task_id[:16] if len(task_id) > 16 else task_id
        cid = checkpoint_id or f"cp_{task_tag}_{step_index}_{int(now * 1000)}"
        state_repr = json.dumps(agent_state, sort_keys=True)
        state_hash = hashlib.sha256(state_repr.encode("utf-8")).hexdigest()[:16]

        meta = CheckpointMetadata(
            checkpoint_id=cid,
            task_id=task_id,
            step_index=step_index,
            created_at=now,
            state_hash=state_hash,
            agent_state=agent_state,
        )

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        target_file = self.storage_dir / f"{cid}.json"
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(meta.to_dict(), f, indent=2)

        return meta

    def load_checkpoint(self, checkpoint_id: str) -> Optional[CheckpointMetadata]:
        """Loads a checkpoint by ID."""
        target_file = self.storage_dir / f"{checkpoint_id}.json"
        if not target_file.exists():
            return None
        with open(target_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return CheckpointMetadata.from_dict(data)

    def restore_checkpoint(self, checkpoint_id: str) -> Optional[CheckpointMetadata]:
        """Restores agent workflow state from JSON.

        Crucially, this only loads the serialized agent memory and does NOT
        restore or erase external service state or the SQLite effect ledger.
        """
        return self.load_checkpoint(checkpoint_id)

    def list_checkpoints(self, task_id: str) -> List[CheckpointMetadata]:
        """Lists all saved checkpoints for a task sorted by step_index."""
        checkpoints: List[CheckpointMetadata] = []
        for file in self.storage_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("task_id") == task_id:
                    checkpoints.append(CheckpointMetadata.from_dict(data))
            except (json.JSONDecodeError, OSError):
                continue
        checkpoints.sort(key=lambda cp: (cp.step_index, cp.created_at))
        return checkpoints

    def get_latest_checkpoint(self, task_id: str) -> Optional[CheckpointMetadata]:
        """Returns the most recent checkpoint for a task, or None."""
        cps = self.list_checkpoints(task_id)
        return cps[-1] if cps else None

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Deletes a specific checkpoint file."""
        target_file = self.storage_dir / f"{checkpoint_id}.json"
        if target_file.exists():
            target_file.unlink()
            return True
        return False
