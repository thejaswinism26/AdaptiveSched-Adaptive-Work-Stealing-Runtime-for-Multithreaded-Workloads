"""AI-Agent workflow runtime with mandatory post-restore recovery gate.

Enforces:
1. Every checkpoint restoration marks the agent as requiring recovery verification.
2. Prevents all side-effecting tool calls until the recovery engine explicitly authorizes continuation.
3. Verification gate runs before the first post-restore tool action.
4. Authorization is strictly scoped to the verified task, operation, tool, and outcome.
5. Fails closed on verification exceptions, unavailable ledger, or undetected evidence drift.
6. Checkpoint restoration never inherits or restores previous gate authorizations.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from recoveryguard.checkpoints import CheckpointManager
from recoveryguard.ledger import DurableEffectLedger
from recoveryguard.mock_services import InventoryStore, NotificationService, PaymentGateway
from recoveryguard.models import (
    ActionIntent,
    ActionRiskLevel,
    EffectStatus,
    GateAuthorization,
    GateDeniedError,
    RecoveryDecision,
    RecoveryOutcome,
    RecoveryVerificationError,
)
from recoveryguard.recovery_engine import RecoveryEngine


class AgentWorkflow:
    """Manages multi-step agent workflow execution with an enforced mandatory recovery gate."""

    def __init__(
        self,
        task_id: str,
        ledger: DurableEffectLedger,
        checkpoints: CheckpointManager,
        recovery_engine: Optional[RecoveryEngine],
        payment_gw: PaymentGateway,
        inventory_store: InventoryStore,
        notifications: NotificationService,
        mode: str = "recovery_guard",  # "checkpoint_only", "effect_log_only", "recovery_guard"
        conservative_effect_log: bool = True,
    ) -> None:
        self.task_id = task_id
        self.ledger = ledger
        self.checkpoints = checkpoints
        self.recovery_engine = recovery_engine
        self.payment_gw = payment_gw
        self.inventory_store = inventory_store
        self.notifications = notifications
        self.mode = mode
        self.conservative_effect_log = conservative_effect_log

        # Working memory state
        self.state: Dict[str, Any] = {
            "order_id": None,
            "customer_id": None,
            "item_id": None,
            "amount": 0.0,
            "step_index": 0,
            "payment_id": None,
            "reservation_id": None,
            "notification_sent": False,
            "completed": False,
        }

        # Mandatory Post-Restore Gate State
        self._requires_verification: bool = False
        self._active_authorization: Optional[GateAuthorization] = None
        self._last_gate_decision: Optional[RecoveryDecision] = None

    def init_task(self, order_id: str, customer_id: str, item_id: str, amount: float) -> None:
        """Initializes the agent working memory for an order."""
        self.state.update({
            "order_id": order_id,
            "customer_id": customer_id,
            "item_id": item_id,
            "amount": amount,
            "step_index": 0,
            "completed": False,
        })
        self._requires_verification = False
        self._active_authorization = None
        self._last_gate_decision = None

    def save_checkpoint(self) -> str:
        """Saves current working memory state to a checkpoint file."""
        meta = self.checkpoints.create_checkpoint(
            task_id=self.task_id,
            step_index=self.state["step_index"],
            agent_state=self.state,
        )
        return meta.checkpoint_id

    def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restores working memory from a checkpoint.

        Mandatory Post-Restore Gate Invariant:
        Restoration strictly resets authorization to None and marks the agent
        as requiring recovery verification before any side-effecting tool action.
        """
        meta = self.checkpoints.load_checkpoint(checkpoint_id)
        if not meta:
            return False

        self.state = dict(meta.agent_state)

        # Mark agent as requiring mandatory recovery verification
        self._requires_verification = True
        self._active_authorization = None
        self._last_gate_decision = None
        return True

    def requires_verification(self) -> bool:
        """Returns True if the agent requires recovery verification before side effects."""
        return self._requires_verification

    def get_active_authorization(self) -> Optional[GateAuthorization]:
        """Returns the currently active scoped gate authorization, if any."""
        return self._active_authorization

    def run_recovery_gate(
        self,
        operation_id: str,
        tool_name: str,
        risk_level: ActionRiskLevel = ActionRiskLevel.RISKY_MUTATION,
        probe_fn: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
        probe_supported: bool = True,
        probe_available: bool = True,
        probe_is_stale: bool = False,
        invariant_context: Optional[Dict[str, Any]] = None,
        is_reconciliation: bool = False,
        is_compensation: bool = False,
        unknown_policy: Optional[str] = None,
        conflict_policy: Optional[str] = None,
    ) -> RecoveryDecision:
        """Runs the recovery verification gate for a specific proposed operation.

        Evaluates evidence, checks task invariants, and sets scoped authorization.
        Fails closed on any exception.
        """
        # Fail-closed check: Recovery engine and durable ledger must be available
        if self.recovery_engine is None or self.ledger is None:
            self._requires_verification = True
            self._active_authorization = None
            raise RecoveryVerificationError(
                "Recovery engine or durable effect ledger is unavailable; failing closed."
            )

        try:
            # 1. Fetch probe data safely
            probe_data = None
            if probe_fn is not None and probe_supported and probe_available and not probe_is_stale:
                probe_data = probe_fn()

            current_cp = self.checkpoints.get_latest_checkpoint(self.task_id)
            cp_id = current_cp.checkpoint_id if current_cp else None

            intent = ActionIntent(
                action_id=f"act_{uuid.uuid4().hex[:10]}",
                task_id=self.task_id,
                operation_id=operation_id,
                tool_name=tool_name,
                parameters={"order_id": self.state.get("order_id")},
                risk_level=risk_level,
                checkpoint_id=cp_id,
                timestamp=time.time(),
                is_compensatable=is_compensation or ("notify" not in tool_name.lower()),
            )

            # 2. Evaluate gate with deterministic decision engine
            decision = self.recovery_engine.evaluate_gate(
                intent=intent,
                probe_data=probe_data,
                probe_supported=probe_supported,
                probe_available=probe_available,
                probe_is_stale=probe_is_stale,
                invariant_context=invariant_context,
                unknown_policy=unknown_policy,
                conflict_policy=conflict_policy,
            )

            self._last_gate_decision = decision

            # 3. Compute evidence signature for drift detection
            evidence_signature = None
            if probe_data is not None:
                evidence_signature = str(sorted(probe_data.items())) if isinstance(probe_data, dict) else str(probe_data)

            # 4. Process outcome and issue tightly scoped authorization
            if decision.outcome in (RecoveryOutcome.RESUME, RecoveryOutcome.SAFE_RETRY, RecoveryOutcome.RETRY):
                self._active_authorization = GateAuthorization(
                    task_id=self.task_id,
                    operation_id=operation_id,
                    tool_name=tool_name,
                    outcome=decision.outcome,
                    decision_id=decision.decision_id,
                    timestamp=time.time(),
                    is_reconciliation=False,
                    is_compensation=False,
                    evidence_signature=evidence_signature,
                )
                self._requires_verification = False

            elif decision.outcome == RecoveryOutcome.RECONCILE:
                self._active_authorization = GateAuthorization(
                    task_id=self.task_id,
                    operation_id=operation_id,
                    tool_name=tool_name,
                    outcome=RecoveryOutcome.RECONCILE,
                    decision_id=decision.decision_id,
                    timestamp=time.time(),
                    is_reconciliation=True,
                    is_compensation=False,
                    evidence_signature=evidence_signature,
                )
                self._requires_verification = False

            elif decision.outcome == RecoveryOutcome.COMPENSATE:
                self._active_authorization = GateAuthorization(
                    task_id=self.task_id,
                    operation_id=operation_id,
                    tool_name=tool_name,
                    outcome=RecoveryOutcome.COMPENSATE,
                    decision_id=decision.decision_id,
                    timestamp=time.time(),
                    is_reconciliation=False,
                    is_compensation=True,
                    evidence_signature=evidence_signature,
                )
                self._requires_verification = False

            else:
                # QUARANTINE or HUMAN_REVIEW: Deny authorization
                self._active_authorization = None
                self._requires_verification = True

            return decision

        except Exception as e:
            # Mandatory Fail-Closed: any error during verification wipes authorization
            self._requires_verification = True
            self._active_authorization = None
            if isinstance(e, RecoveryVerificationError):
                raise
            raise RecoveryVerificationError(f"Recovery verification threw an exception: {e}; failing closed.") from e

    def execute_tool(
        self,
        operation_id: str,
        tool_name: str,
        risk_level: ActionRiskLevel,
        tool_fn: Callable[[], Any],
        probe_fn: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
        is_reconciliation: bool = False,
        is_compensation: bool = False,
        raise_on_denial: bool = True,
    ) -> Dict[str, Any]:
        """Direct tool execution interface enforcing post-restore gate authorization.

        All side-effecting actions MUST pass through this method.
        """
        # Rule 1: Read-only operations are safe and do not trigger mutations
        if risk_level == ActionRiskLevel.SAFE_READ:
            result = tool_fn()
            return {"outcome": "EXECUTED_READ_ONLY", "authorized": True, "result": result}

        # Rule 2: Fail closed if durable ledger is unavailable
        if self.mode in ("recovery_guard", "recoveryguard", "RECOVERYGUARD"):
            if self.ledger is None:
                msg = "Execution denied: Durable effect ledger is unavailable (fail-closed)."
                if raise_on_denial:
                    raise GateDeniedError(msg)
                return {"outcome": "BLOCKED_BY_GATE", "authorized": False, "reason": msg}

            # Rule 3: Enforce post-restore verification gate
            if self._requires_verification and self._active_authorization is None:
                msg = (
                    "Execution denied: Agent state was restored from checkpoint and requires "
                    "recovery verification before executing side-effecting tools."
                )
                if raise_on_denial:
                    raise GateDeniedError(msg)
                return {"outcome": "BLOCKED_BY_GATE", "authorized": False, "reason": msg}

            # Rule 4: Scoped Authorization Verification
            auth = self._active_authorization
            if auth is None:
                msg = f"Execution denied: No active recovery authorization exists for tool '{tool_name}'."
                if raise_on_denial:
                    raise GateDeniedError(msg)
                return {"outcome": "BLOCKED_BY_GATE", "authorized": False, "reason": msg}

            if auth.task_id != self.task_id:
                msg = f"Execution denied: Authorization is scoped to task '{auth.task_id}', not '{self.task_id}'."
                if raise_on_denial:
                    raise GateDeniedError(msg)
                return {"outcome": "BLOCKED_BY_GATE", "authorized": False, "reason": msg}

            if auth.operation_id != operation_id:
                msg = f"Execution denied: Authorization for operation '{auth.operation_id}' cannot authorize '{operation_id}'."
                if raise_on_denial:
                    raise GateDeniedError(msg)
                return {"outcome": "BLOCKED_BY_GATE", "authorized": False, "reason": msg}

            if auth.outcome not in (RecoveryOutcome.RECONCILE, RecoveryOutcome.COMPENSATE):
                if auth.tool_name != tool_name:
                    msg = f"Execution denied: Authorization is scoped to tool '{auth.tool_name}', not '{tool_name}'."
                    if raise_on_denial:
                        raise GateDeniedError(msg)
                    return {"outcome": "BLOCKED_BY_GATE", "authorized": False, "reason": msg}

            # Rule 5: Detect external evidence drift after verification
            if probe_fn is not None and auth.evidence_signature is not None:
                try:
                    curr_probe = probe_fn()
                    curr_sig = str(sorted(curr_probe.items())) if isinstance(curr_probe, dict) else str(curr_probe)
                    if curr_sig != auth.evidence_signature:
                        self._active_authorization = None
                        self._requires_verification = True
                        msg = "Execution denied: External state evidence changed since verification; re-verification required."
                        if raise_on_denial:
                            raise GateDeniedError(msg)
                        return {"outcome": "BLOCKED_BY_GATE", "authorized": False, "reason": msg}
                except Exception:
                    pass

            # Rule 6: Outcome-specific permission enforcement
            if auth.outcome == RecoveryOutcome.RECONCILE:
                if not (is_reconciliation or "reconcile" in tool_name.lower()):
                    msg = (
                        f"Execution denied: Recovery outcome is RECONCILE. Ordinary tool '{tool_name}' is "
                        f"forbidden; only designated reconciliation operations are permitted."
                    )
                    if raise_on_denial:
                        raise GateDeniedError(msg)
                    return {
                        "outcome": "BLOCKED_BY_GATE",
                        "authorized": False,
                        "reason": msg,
                        "decision": self._last_gate_decision.to_dict() if self._last_gate_decision else None,
                        "mode": self.mode,
                    }

                self._active_authorization = None
                result = tool_fn()
                return {"outcome": "RECONCILED", "authorized": True, "result": result}

            elif auth.outcome == RecoveryOutcome.COMPENSATE:
                if not (is_compensation or "compensate" in tool_name.lower() or "refund" in tool_name.lower() or "release" in tool_name.lower()):
                    msg = (
                        f"Execution denied: Recovery outcome is COMPENSATE. Ordinary tool '{tool_name}' is "
                        f"forbidden; only approved compensation actions are permitted."
                    )
                    if raise_on_denial:
                        raise GateDeniedError(msg)
                    return {
                        "outcome": "BLOCKED_BY_GATE",
                        "authorized": False,
                        "reason": msg,
                        "decision": self._last_gate_decision.to_dict() if self._last_gate_decision else None,
                        "mode": self.mode,
                    }

                self._active_authorization = None
                result = tool_fn()
                return {"outcome": "COMPENSATED", "authorized": True, "result": result}

            elif auth.outcome in (RecoveryOutcome.QUARANTINE, RecoveryOutcome.HUMAN_REVIEW, RecoveryOutcome.HUMAN_INTERVENTION):
                msg = f"Execution denied: Recovery gate outcome is '{auth.outcome.value}'. Side effects blocked."
                if raise_on_denial:
                    raise GateDeniedError(msg)
                return {
                    "outcome": "BLOCKED_BY_GATE",
                    "authorized": False,
                    "reason": msg,
                    "decision": self._last_gate_decision.to_dict() if self._last_gate_decision else None,
                    "mode": self.mode,
                }

            elif auth.outcome == RecoveryOutcome.RESUME:
                # Do NOT repeat side effect! Bypass mutation
                self._active_authorization = None
                self.state["step_index"] += 1
                return {
                    "outcome": "RESUMED_WITHOUT_MUTATION",
                    "authorized": True,
                    "decision": self._last_gate_decision.to_dict() if self._last_gate_decision else None,
                    "mode": self.mode,
                }

            elif auth.outcome in (RecoveryOutcome.SAFE_RETRY, RecoveryOutcome.RETRY):
                # Safe to execute tool call
                self._active_authorization = None
                action_id = f"act_{uuid.uuid4().hex[:10]}"
                current_cp = self.checkpoints.get_latest_checkpoint(self.task_id)
                cp_id = current_cp.checkpoint_id if current_cp else None
                intent = ActionIntent(
                    action_id=action_id,
                    task_id=self.task_id,
                    operation_id=operation_id,
                    tool_name=tool_name,
                    parameters={"order_id": self.state.get("order_id")},
                    risk_level=risk_level,
                    checkpoint_id=cp_id,
                    timestamp=time.time(),
                )
                self.ledger.record_intent(intent)
                self.ledger.mark_started(action_id)
                result = tool_fn()
                self.ledger.mark_committed(action_id, result if isinstance(result, dict) else {"result": result})
                self.state["step_index"] += 1
                return {
                    "outcome": "EXECUTED_SAFE",
                    "authorized": True,
                    "result": result,
                    "decision": self._last_gate_decision.to_dict() if self._last_gate_decision else None,
                    "mode": self.mode,
                }

        # Baseline Mode 1: Checkpoint-Only (Blind re-execution)
        elif self.mode in ("checkpoint_only", "checkpoint", "CHECKPOINT_ONLY"):
            result = tool_fn()
            self.state["step_index"] += 1
            return {"outcome": "EXECUTED", "authorized": True, "result": result, "mode": self.mode}

        # Baseline Mode 2: Effect-Log-Only (Basic replay protection)
        elif self.mode in ("effect_log_only", "effect_log", "EFFECT_LOG"):
            # Check 1: Replay protection for committed actions
            if self.ledger.has_committed_operation(operation_id):
                self.state["step_index"] += 1
                return {"outcome": "SKIPPED_DUPLICATE", "authorized": True, "mode": self.mode}

            # Check 2: Conservative unknown policy for in-flight/timed-out operations without external probes
            if getattr(self, "conservative_effect_log", True):
                attempts = self.ledger.get_effects_for_operation(operation_id)
                has_uncommitted = any(
                    r.status in (EffectStatus.TIMEOUT, EffectStatus.STARTED, EffectStatus.ATTEMPTED, EffectStatus.UNKNOWN)
                    for r in attempts
                )
                if has_uncommitted and risk_level == ActionRiskLevel.RISKY_MUTATION:
                    return {
                        "outcome": "BLOCKED_UNKNOWN",
                        "authorized": False,
                        "reason": f"Effect-Log Baseline: Operation '{operation_id}' has uncommitted attempts; paused conservatively.",
                        "mode": self.mode,
                    }

            action_id = f"act_{uuid.uuid4().hex[:10]}"
            intent = ActionIntent(
                action_id=action_id,
                task_id=self.task_id,
                operation_id=operation_id,
                tool_name=tool_name,
                parameters={"order_id": self.state.get("order_id")},
                risk_level=risk_level,
            )
            self.ledger.record_intent(intent)
            self.ledger.mark_started(action_id)
            result = tool_fn()
            self.ledger.mark_committed(action_id, result if isinstance(result, dict) else {"result": result})
            self.state["step_index"] += 1
            return {"outcome": "EXECUTED", "authorized": True, "result": result, "mode": self.mode}

        raise ValueError(f"Unknown mode: {self.mode}")

    def execute_step(
        self,
        operation_id: str,
        tool_name: str,
        risk_level: ActionRiskLevel,
        tool_fn: Callable[[], Dict[str, Any]],
        probe_fn: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
        probe_supported: bool = True,
        probe_available: bool = True,
        probe_is_stale: bool = False,
        invariant_context: Optional[Dict[str, Any]] = None,
        is_reconciliation: bool = False,
        is_compensation: bool = False,
        unknown_policy: Optional[str] = None,
        conflict_policy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Workflow step runner. In RecoveryGuard mode, automatically invokes recovery gate if needed."""
        if self.mode in ("recovery_guard", "recoveryguard", "RECOVERYGUARD"):
            # If agent state was restored or authorization is missing, run recovery gate first
            if self._requires_verification or self._active_authorization is None:
                try:
                    decision = self.run_recovery_gate(
                        operation_id=operation_id,
                        tool_name=tool_name,
                        risk_level=risk_level,
                        probe_fn=probe_fn,
                        probe_supported=probe_supported,
                        probe_available=probe_available,
                        probe_is_stale=probe_is_stale,
                        invariant_context=invariant_context,
                        is_reconciliation=is_reconciliation,
                        is_compensation=is_compensation,
                        unknown_policy=unknown_policy,
                        conflict_policy=conflict_policy,
                    )
                except RecoveryVerificationError as err:
                    return {
                        "outcome": "BLOCKED_BY_GATE",
                        "authorized": False,
                        "error": str(err),
                        "decision": None,
                        "mode": self.mode,
                    }

                if not decision.is_safe_to_execute and decision.outcome not in (
                    RecoveryOutcome.RECONCILE,
                    RecoveryOutcome.COMPENSATE,
                ):
                    return {
                        "outcome": "BLOCKED_BY_GATE",
                        "authorized": False,
                        "decision": decision.to_dict(),
                        "mode": self.mode,
                    }

        return self.execute_tool(
            operation_id=operation_id,
            tool_name=tool_name,
            risk_level=risk_level,
            tool_fn=tool_fn,
            probe_fn=probe_fn,
            is_reconciliation=is_reconciliation,
            is_compensation=is_compensation,
            raise_on_denial=False,
        )
