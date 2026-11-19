"""Durable effect ledger backed by SQLite.

Provides transaction-safe tracking of agent tool intentions and execution outcomes.
Survives agent memory rollbacks and correlates retries using stable operation IDs.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from recoveryguard.database import DatabaseManager
from recoveryguard.models import (
    ActionIntent,
    EffectEvidence,
    EffectRecord,
    EffectStatus,
    EvidenceSource,
    EvidenceState,
    InvariantResult,
    RecoveryDecision,
    RecoveryOutcome,
)


class DurableEffectLedger:
    """Manages persistent action records and recovery decisions."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def record_intent(self, intent: ActionIntent) -> EffectRecord:
        """Records an agent's intention to perform an action."""
        now = time.time()
        timestamp = intent.timestamp if intent.timestamp > 0 else now
        params_json = json.dumps(intent.parameters)

        sql = """
        INSERT INTO effect_records (
            action_id, task_id, operation_id, tool_name, status,
            params_json, result_json, error_message, checkpoint_id,
            evidence_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?, '[]', ?, ?)
        """
        with self.db.transaction() as cur:
            cur.execute(
                sql,
                (
                    intent.action_id,
                    intent.task_id,
                    intent.operation_id,
                    intent.tool_name,
                    EffectStatus.PENDING.value,
                    params_json,
                    intent.checkpoint_id,
                    timestamp,
                    timestamp,
                ),
            )
            record_id = cur.lastrowid

        return EffectRecord(
            record_id=record_id,
            action_id=intent.action_id,
            task_id=intent.task_id,
            operation_id=intent.operation_id,
            tool_name=intent.tool_name,
            status=EffectStatus.PENDING,
            params=intent.parameters,
            result=None,
            error_message=None,
            checkpoint_id=intent.checkpoint_id,
            created_at=timestamp,
            updated_at=timestamp,
            evidence_list=[],
        )

    def record_attempt(
        self,
        intent: ActionIntent,
        status: EffectStatus = EffectStatus.ATTEMPTED,
    ) -> EffectRecord:
        """Records a new execution attempt for a logical operation."""
        now = time.time()
        timestamp = intent.timestamp if intent.timestamp > 0 else now
        params_json = json.dumps(intent.parameters)

        sql = """
        INSERT INTO effect_records (
            action_id, task_id, operation_id, tool_name, status,
            params_json, result_json, error_message, checkpoint_id,
            evidence_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?, '[]', ?, ?)
        """
        with self.db.transaction() as cur:
            cur.execute(
                sql,
                (
                    intent.action_id,
                    intent.task_id,
                    intent.operation_id,
                    intent.tool_name,
                    status.value,
                    params_json,
                    intent.checkpoint_id,
                    timestamp,
                    timestamp,
                ),
            )
            record_id = cur.lastrowid

        return EffectRecord(
            record_id=record_id,
            action_id=intent.action_id,
            task_id=intent.task_id,
            operation_id=intent.operation_id,
            tool_name=intent.tool_name,
            status=status,
            params=intent.parameters,
            result=None,
            error_message=None,
            checkpoint_id=intent.checkpoint_id,
            created_at=timestamp,
            updated_at=timestamp,
            evidence_list=[],
        )

    def mark_started(self, action_id: str) -> bool:
        """Marks that execution of the action has begun."""
        now = time.time()
        sql = """
        UPDATE effect_records
        SET status = ?, updated_at = ?
        WHERE action_id = ?
        """
        with self.db.transaction() as cur:
            cur.execute(sql, (EffectStatus.STARTED.value, now, action_id))
            return cur.rowcount > 0

    def mark_attempted(self, action_id: str) -> bool:
        """Marks that an execution attempt is actively in progress."""
        now = time.time()
        sql = """
        UPDATE effect_records
        SET status = ?, updated_at = ?
        WHERE action_id = ?
        """
        with self.db.transaction() as cur:
            cur.execute(sql, (EffectStatus.ATTEMPTED.value, now, action_id))
            return cur.rowcount > 0

    def record_evidence(self, action_id: str, evidence: EffectEvidence) -> bool:
        """Appends structured evidence to an action record."""
        now = time.time()
        record = self.get_effect(action_id)
        if not record:
            return False

        existing_evidence = [e.to_dict() for e in record.evidence_list]
        existing_evidence.append(evidence.to_dict())
        evidence_json = json.dumps(existing_evidence)

        sql = """
        UPDATE effect_records
        SET evidence_json = ?, updated_at = ?
        WHERE action_id = ?
        """
        with self.db.transaction() as cur:
            cur.execute(sql, (evidence_json, now, action_id))
            return cur.rowcount > 0

    def mark_committed(
        self,
        action_id: str,
        result: Dict[str, Any],
        evidence: Optional[EffectEvidence] = None,
    ) -> bool:
        """Marks that an action completed and its side effects committed successfully."""
        now = time.time()
        result_json = json.dumps(result)
        evidence_json = None
        if evidence:
            record = self.get_effect(action_id)
            existing = [e.to_dict() for e in record.evidence_list] if record else []
            existing.append(evidence.to_dict())
            evidence_json = json.dumps(existing)

        if evidence_json:
            sql = """
            UPDATE effect_records
            SET status = ?, result_json = ?, evidence_json = ?, updated_at = ?
            WHERE action_id = ?
            """
            params = (EffectStatus.COMMITTED.value, result_json, evidence_json, now, action_id)
        else:
            sql = """
            UPDATE effect_records
            SET status = ?, result_json = ?, updated_at = ?
            WHERE action_id = ?
            """
            params = (EffectStatus.COMMITTED.value, result_json, now, action_id)

        with self.db.transaction() as cur:
            cur.execute(sql, params)
            return cur.rowcount > 0

    def mark_confirmed_success(
        self,
        action_id: str,
        result: Dict[str, Any],
        evidence: Optional[EffectEvidence] = None,
    ) -> bool:
        """Marks that an action is confirmed to have successfully executed externally."""
        now = time.time()
        result_json = json.dumps(result)
        evidence_json = None
        if evidence:
            record = self.get_effect(action_id)
            existing = [e.to_dict() for e in record.evidence_list] if record else []
            existing.append(evidence.to_dict())
            evidence_json = json.dumps(existing)

        if evidence_json:
            sql = """
            UPDATE effect_records
            SET status = ?, result_json = ?, evidence_json = ?, updated_at = ?
            WHERE action_id = ?
            """
            params = (EffectStatus.CONFIRMED_SUCCESS.value, result_json, evidence_json, now, action_id)
        else:
            sql = """
            UPDATE effect_records
            SET status = ?, result_json = ?, updated_at = ?
            WHERE action_id = ?
            """
            params = (EffectStatus.CONFIRMED_SUCCESS.value, result_json, now, action_id)

        with self.db.transaction() as cur:
            cur.execute(sql, params)
            return cur.rowcount > 0

    def mark_confirmed_absent(
        self,
        action_id: str,
        details: str = "",
        evidence: Optional[EffectEvidence] = None,
    ) -> bool:
        """Marks that external evidence confirms the side effect definitely did NOT occur."""
        now = time.time()
        evidence_json = None
        if evidence:
            record = self.get_effect(action_id)
            existing = [e.to_dict() for e in record.evidence_list] if record else []
            existing.append(evidence.to_dict())
            evidence_json = json.dumps(existing)

        if evidence_json:
            sql = """
            UPDATE effect_records
            SET status = ?, error_message = ?, evidence_json = ?, updated_at = ?
            WHERE action_id = ?
            """
            params = (EffectStatus.CONFIRMED_ABSENT.value, details, evidence_json, now, action_id)
        else:
            sql = """
            UPDATE effect_records
            SET status = ?, error_message = ?, updated_at = ?
            WHERE action_id = ?
            """
            params = (EffectStatus.CONFIRMED_ABSENT.value, details, now, action_id)

        with self.db.transaction() as cur:
            cur.execute(sql, params)
            return cur.rowcount > 0

    def mark_unknown(
        self,
        action_id: str,
        details: str = "",
        evidence: Optional[EffectEvidence] = None,
    ) -> bool:
        """Marks that the execution status is unknown."""
        now = time.time()
        evidence_json = None
        if evidence:
            record = self.get_effect(action_id)
            existing = [e.to_dict() for e in record.evidence_list] if record else []
            existing.append(evidence.to_dict())
            evidence_json = json.dumps(existing)

        if evidence_json:
            sql = """
            UPDATE effect_records
            SET status = ?, error_message = ?, evidence_json = ?, updated_at = ?
            WHERE action_id = ?
            """
            params = (EffectStatus.UNKNOWN.value, details, evidence_json, now, action_id)
        else:
            sql = """
            UPDATE effect_records
            SET status = ?, error_message = ?, updated_at = ?
            WHERE action_id = ?
            """
            params = (EffectStatus.UNKNOWN.value, details, now, action_id)

        with self.db.transaction() as cur:
            cur.execute(sql, params)
            return cur.rowcount > 0

    def mark_conflicting(
        self,
        action_id: str,
        details: str = "",
        evidence: Optional[EffectEvidence] = None,
    ) -> bool:
        """Marks that conflicting evidence was detected for this action."""
        now = time.time()
        evidence_json = None
        if evidence:
            record = self.get_effect(action_id)
            existing = [e.to_dict() for e in record.evidence_list] if record else []
            existing.append(evidence.to_dict())
            evidence_json = json.dumps(existing)

        if evidence_json:
            sql = """
            UPDATE effect_records
            SET status = ?, error_message = ?, evidence_json = ?, updated_at = ?
            WHERE action_id = ?
            """
            params = (EffectStatus.CONFLICTING.value, details, evidence_json, now, action_id)
        else:
            sql = """
            UPDATE effect_records
            SET status = ?, error_message = ?, updated_at = ?
            WHERE action_id = ?
            """
            params = (EffectStatus.CONFLICTING.value, details, now, action_id)

        with self.db.transaction() as cur:
            cur.execute(sql, params)
            return cur.rowcount > 0

    def mark_failed(self, action_id: str, error_message: str) -> bool:
        """Marks an action as failed with an error message."""
        now = time.time()
        sql = """
        UPDATE effect_records
        SET status = ?, error_message = ?, updated_at = ?
        WHERE action_id = ?
        """
        with self.db.transaction() as cur:
            cur.execute(sql, (EffectStatus.FAILED.value, error_message, now, action_id))
            return cur.rowcount > 0

    def mark_timeout(self, action_id: str, error_message: str = "Request timed out") -> bool:
        """Marks an action as TIMEOUT.

        Crucially, a timed-out action is marked TIMEOUT, NOT FAILED,
        acknowledging external execution status is unknown.
        """
        now = time.time()
        sql = """
        UPDATE effect_records
        SET status = ?, error_message = ?, updated_at = ?
        WHERE action_id = ?
        """
        with self.db.transaction() as cur:
            cur.execute(sql, (EffectStatus.TIMEOUT.value, error_message, now, action_id))
            return cur.rowcount > 0

    def get_effect(self, action_id: str) -> Optional[EffectRecord]:
        """Retrieves an effect record by its specific action ID."""
        sql = """
        SELECT record_id, action_id, task_id, operation_id, tool_name,
               status, params_json, result_json, error_message, checkpoint_id,
               created_at, updated_at
        FROM effect_records
        WHERE action_id = ?
        """
        conn = self.db.get_connection()
        try:
            cur = conn.cursor()
            cur.execute(sql, (action_id,))
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_effect(row)
        finally:
            if not self.db._is_memory:
                conn.close()

    def get_effects_for_operation(self, operation_id: str) -> List[EffectRecord]:
        """Retrieves all attempt records associated with a logical operation ID."""
        sql = """
        SELECT record_id, action_id, task_id, operation_id, tool_name,
               status, params_json, result_json, error_message, checkpoint_id,
               created_at, updated_at
        FROM effect_records
        WHERE operation_id = ?
        ORDER BY record_id ASC
        """
        conn = self.db.get_connection()
        try:
            cur = conn.cursor()
            cur.execute(sql, (operation_id,))
            return [self._row_to_effect(row) for row in cur.fetchall()]
        finally:
            if not self.db._is_memory:
                conn.close()

    def get_effects_for_task(self, task_id: str) -> List[EffectRecord]:
        """Retrieves all effect records for a task."""
        sql = """
        SELECT record_id, action_id, task_id, operation_id, tool_name,
               status, params_json, result_json, error_message, checkpoint_id,
               created_at, updated_at
        FROM effect_records
        WHERE task_id = ?
        ORDER BY record_id ASC
        """
        conn = self.db.get_connection()
        try:
            cur = conn.cursor()
            cur.execute(sql, (task_id,))
            return [self._row_to_effect(row) for row in cur.fetchall()]
        finally:
            if not self.db._is_memory:
                conn.close()

    def has_committed_operation(self, operation_id: str) -> bool:
        """Checks if any prior attempt for this logical operation succeeded."""
        sql = """
        SELECT 1 FROM effect_records
        WHERE operation_id = ? AND (status = ? OR status = ?)
        LIMIT 1
        """
        conn = self.db.get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                sql,
                (
                    operation_id,
                    EffectStatus.COMMITTED.value,
                    EffectStatus.CONFIRMED_SUCCESS.value,
                ),
            )
            return cur.fetchone() is not None
        finally:
            if not self.db._is_memory:
                conn.close()

    def get_attempts_for_operation(self, operation_id: str) -> List[EffectRecord]:
        """Retrieves all historical execution attempts for an operation in chronological order."""
        return self.get_effects_for_operation(operation_id)

    def _row_to_effect(self, row: Any) -> EffectRecord:
        params = json.loads(row["params_json"]) if row["params_json"] else {}
        result = json.loads(row["result_json"]) if row["result_json"] else None
        evidence_list = []
        try:
            # Check if evidence_json column is present in the row
            row_keys = row.keys() if hasattr(row, "keys") else []
            if "evidence_json" in row_keys and row["evidence_json"]:
                raw_ev = json.loads(row["evidence_json"])
                evidence_list = [EffectEvidence.from_dict(e) for e in raw_ev]
        except Exception:
            evidence_list = []

        return EffectRecord(
            record_id=row["record_id"],
            action_id=row["action_id"],
            task_id=row["task_id"],
            operation_id=row["operation_id"],
            tool_name=row["tool_name"],
            status=EffectStatus(row["status"]),
            params=params,
            result=result,
            error_message=row["error_message"],
            checkpoint_id=row["checkpoint_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            evidence_list=evidence_list,
        )

    def record_decision(self, decision: RecoveryDecision) -> None:
        """Stores a recovery gate decision record."""
        evidence_json = json.dumps([e.to_dict() for e in decision.evidence_list])
        invariants_json = json.dumps([i.to_dict() for i in decision.invariants_checked])

        sql = """
        INSERT OR REPLACE INTO recovery_decisions (
            decision_id, task_id, action_id, operation_id, checkpoint_id,
            outcome, is_safe_to_execute, rationale, timestamp,
            evidence_json, invariants_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self.db.transaction() as cur:
            cur.execute(
                sql,
                (
                    decision.decision_id,
                    decision.task_id,
                    decision.action_id,
                    decision.operation_id,
                    decision.checkpoint_id,
                    decision.outcome.value,
                    1 if decision.is_safe_to_execute else 0,
                    decision.rationale,
                    decision.timestamp,
                    evidence_json,
                    invariants_json,
                ),
            )

    def get_decisions_for_operation(self, operation_id: str) -> List[RecoveryDecision]:
        """Retrieves recovery gate decisions for an operation."""
        sql = """
        SELECT decision_id, task_id, action_id, operation_id, checkpoint_id,
               outcome, is_safe_to_execute, rationale, timestamp,
               evidence_json, invariants_json
        FROM recovery_decisions
        WHERE operation_id = ?
        ORDER BY timestamp ASC
        """
        conn = self.db.get_connection()
        try:
            cur = conn.cursor()
            cur.execute(sql, (operation_id,))
            return [self._row_to_decision(row) for row in cur.fetchall()]
        finally:
            if not self.db._is_memory:
                conn.close()

    def _row_to_decision(self, row: Any) -> RecoveryDecision:
        evidence_data = json.loads(row["evidence_json"]) if row["evidence_json"] else []
        invariants_data = json.loads(row["invariants_json"]) if row["invariants_json"] else []
        return RecoveryDecision(
            decision_id=row["decision_id"],
            task_id=row["task_id"],
            action_id=row["action_id"],
            operation_id=row["operation_id"],
            checkpoint_id=row["checkpoint_id"],
            outcome=RecoveryOutcome(row["outcome"]),
            is_safe_to_execute=bool(row["is_safe_to_execute"]),
            rationale=row["rationale"],
            timestamp=row["timestamp"],
            evidence_list=[EffectEvidence.from_dict(e) for e in evidence_data],
            invariants_checked=[InvariantResult.from_dict(i) for i in invariants_data],
        )
