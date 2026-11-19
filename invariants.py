"""Task-specific consistency invariants for multi-system workflows.

Validates deterministic business constraints across independent external services:
1. Notification delivery: at most one delivery per logical operation.
2. Deployment records: do not create duplicate records for the same logical operation.
3. Simulated payments: never commit the same logical payment twice.

Distinguishes safety invariants (whose violation indicates unsafe cross-system corruption)
from workflow completion conditions (an incomplete workflow is not unsafe).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from recoveryguard.models import InvariantResult, InvariantStatus


class BaseInvariant:
    """Base class for cross-system invariants and workflow conditions."""

    def __init__(
        self,
        invariant_id: str,
        name: str,
        is_safety_invariant: bool = True,
    ) -> None:
        self.invariant_id = invariant_id
        self.name = name
        self.is_safety_invariant = is_safety_invariant

    def evaluate(self, context: Dict[str, Any]) -> InvariantResult:
        raise NotImplementedError


class NotificationSingleDeliveryInvariant(BaseInvariant):
    """Ensures at most one notification is delivered per logical operation."""

    def __init__(self) -> None:
        super().__init__(
            invariant_id="INV_NOTIF_SINGLE_DELIVERY",
            name="At Most One Notification Delivery Per Operation",
            is_safety_invariant=True,
        )

    def evaluate(self, context: Dict[str, Any]) -> InvariantResult:
        now = time.time()
        operation_id = context.get("operation_id", "unknown_op")
        notification_probe = context.get("notification_probe")
        probe_available = context.get("probe_available", True)

        if not probe_available or context.get("probe_is_stale", False):
            return InvariantResult(
                invariant_id=self.invariant_id,
                name=self.name,
                satisfied=True,  # Does not trigger compensation on probe absence alone
                status=InvariantStatus.UNKNOWN,
                details="Notification probe is unavailable or stale; delivery status unknown.",
                evaluated_at=now,
                affected_entities=[f"operation:{operation_id}"],
                operation_id=operation_id,
                is_safety_invariant=True,
                explanation="Notification probe unavailable or stale; cannot reliably count deliveries.",
            )

        if isinstance(notification_probe, list):
            count = len(notification_probe)
        elif isinstance(notification_probe, dict):
            count = 1
        else:
            count = 0

        passed = (count <= 1)
        details = (
            f"Operation {operation_id} has {count} delivered notification(s). "
            + ("Invariant satisfied." if passed else "VIOLATION: Duplicate delivery detected.")
        )
        return InvariantResult(
            invariant_id=self.invariant_id,
            name=self.name,
            satisfied=passed,
            status=InvariantStatus.PASS if passed else InvariantStatus.FAIL,
            details=details,
            evaluated_at=now,
            affected_entities=[f"operation:{operation_id}", f"deliveries:{count}"],
            operation_id=operation_id,
            is_safety_invariant=True,
            explanation=details,
        )


class DeploymentNoDuplicateInvariant(BaseInvariant):
    """Ensures no duplicate deployment records are created for the same logical operation."""

    def __init__(self) -> None:
        super().__init__(
            invariant_id="INV_DEPLOY_NO_DUPLICATE",
            name="No Duplicate Deployment Records Per Operation",
            is_safety_invariant=True,
        )

    def evaluate(self, context: Dict[str, Any]) -> InvariantResult:
        now = time.time()
        operation_id = context.get("operation_id", "unknown_op")
        deployment_probe = context.get("deployment_probe")
        probe_available = context.get("probe_available", True)

        if not probe_available or context.get("probe_is_stale", False):
            return InvariantResult(
                invariant_id=self.invariant_id,
                name=self.name,
                satisfied=True,
                status=InvariantStatus.UNKNOWN,
                details="Deployment probe is unavailable or stale; deployment state unknown.",
                evaluated_at=now,
                affected_entities=[f"operation:{operation_id}"],
                operation_id=operation_id,
                is_safety_invariant=True,
                explanation="Deployment probe unavailable or stale; duplicate status cannot be confirmed.",
            )

        if isinstance(deployment_probe, list):
            count = len(deployment_probe)
        elif isinstance(deployment_probe, dict):
            count = 1
        else:
            count = 0

        passed = (count <= 1)
        details = (
            f"Operation {operation_id} has {count} active deployment record(s). "
            + ("Invariant satisfied." if passed else "VIOLATION: Duplicate deployment record detected.")
        )
        return InvariantResult(
            invariant_id=self.invariant_id,
            name=self.name,
            satisfied=passed,
            status=InvariantStatus.PASS if passed else InvariantStatus.FAIL,
            details=details,
            evaluated_at=now,
            affected_entities=[f"operation:{operation_id}", f"deployments:{count}"],
            operation_id=operation_id,
            is_safety_invariant=True,
            explanation=details,
        )


class PaymentNoDuplicateInvariant(BaseInvariant):
    """Ensures never committing the same logical payment twice."""

    def __init__(self) -> None:
        super().__init__(
            invariant_id="INV_PAYMENT_NO_DUPLICATE",
            name="Never Commit Duplicate Payment",
            is_safety_invariant=True,
        )

    def evaluate(self, context: Dict[str, Any]) -> InvariantResult:
        now = time.time()
        operation_id = context.get("operation_id", context.get("order_id", "unknown_op"))
        payment_probe = context.get("payment_probe")

        if isinstance(payment_probe, list):
            captured_count = sum(1 for c in payment_probe if c.get("status") == "CAPTURED")
        elif isinstance(payment_probe, dict):
            captured_count = 1 if payment_probe.get("status") == "CAPTURED" else 0
        else:
            captured_count = 0

        passed = (captured_count <= 1)
        details = (
            f"Operation {operation_id} has {captured_count} active captured charge(s). "
            + ("Invariant satisfied." if passed else "VIOLATION: Duplicate captured payment detected.")
        )
        return InvariantResult(
            invariant_id=self.invariant_id,
            name=self.name,
            satisfied=passed,
            status=InvariantStatus.PASS if passed else InvariantStatus.FAIL,
            details=details,
            evaluated_at=now,
            affected_entities=[f"operation:{operation_id}", f"charges:{captured_count}"],
            operation_id=operation_id,
            is_safety_invariant=True,
            explanation=details,
        )


class NoDuplicateChargeInvariant(PaymentNoDuplicateInvariant):
    """Backward-compatible alias for PaymentNoDuplicateInvariant."""
    pass


class PaymentBeforeReservationInvariant(BaseInvariant):
    """Ensures inventory is not held RESERVED if payment was REFUNDED or ABSENT."""

    def __init__(self) -> None:
        super().__init__(
            invariant_id="INV_02_PAYMENT_RESERVATION_ALIGNMENT",
            name="Payment & Reservation Alignment",
            is_safety_invariant=True,
        )

    def evaluate(self, context: Dict[str, Any]) -> InvariantResult:
        now = time.time()
        order_id = context.get("order_id", "unknown")
        operation_id = context.get("operation_id", order_id)
        payment_probe = context.get("payment_probe")
        reservation_probe = context.get("reservation_probe")

        if isinstance(payment_probe, list):
            has_payment = any(c.get("status") == "CAPTURED" for c in payment_probe)
        elif isinstance(payment_probe, dict):
            has_payment = bool(payment_probe.get("status") == "CAPTURED")
        else:
            has_payment = False

        if isinstance(reservation_probe, list):
            has_reservation = any(r.get("status") == "RESERVED" for r in reservation_probe)
        elif isinstance(reservation_probe, dict):
            has_reservation = bool(reservation_probe.get("status") == "RESERVED")
        else:
            has_reservation = False

        # If reservation exists, payment must be captured
        if has_reservation and not has_payment:
            details = f"Order {order_id} holds active reservation but lacks captured payment."
            return InvariantResult(
                invariant_id=self.invariant_id,
                name=self.name,
                satisfied=False,
                status=InvariantStatus.FAIL,
                details=details,
                evaluated_at=now,
                affected_entities=[f"order:{order_id}"],
                operation_id=operation_id,
                is_safety_invariant=True,
                explanation=details,
            )

        details = f"Order {order_id} payment and inventory states are aligned."
        return InvariantResult(
            invariant_id=self.invariant_id,
            name=self.name,
            satisfied=True,
            status=InvariantStatus.PASS,
            details=details,
            evaluated_at=now,
            affected_entities=[f"order:{order_id}"],
            operation_id=operation_id,
            is_safety_invariant=True,
            explanation=details,
        )


class WorkflowCompletionCondition(BaseInvariant):
    """Distinguishes workflow completion conditions from safety invariants.

    An incomplete workflow condition does NOT represent a safety violation or corruption.
    """

    def __init__(self, condition_id: str = "COND_WORKFLOW_COMPLETE", name: str = "Workflow Completion") -> None:
        super().__init__(
            invariant_id=condition_id,
            name=name,
            is_safety_invariant=False,  # Not a safety invariant!
        )

    def evaluate(self, context: Dict[str, Any]) -> InvariantResult:
        now = time.time()
        operation_id = context.get("operation_id", "unknown_op")
        completed_steps = context.get("completed_steps", 0)
        total_steps = context.get("total_steps", 1)

        is_complete = (completed_steps >= total_steps)
        details = f"Workflow progress: {completed_steps}/{total_steps} steps completed."
        return InvariantResult(
            invariant_id=self.invariant_id,
            name=self.name,
            satisfied=is_complete,
            status=InvariantStatus.PASS if is_complete else InvariantStatus.UNKNOWN,
            details=details,
            evaluated_at=now,
            affected_entities=[f"steps:{completed_steps}/{total_steps}"],
            operation_id=operation_id,
            is_safety_invariant=False,
            explanation="Workflow is incomplete but safe." if not is_complete else "Workflow fully completed.",
        )


class InvariantSuite:
    """Manages and evaluates a set of task invariants and workflow conditions."""

    def __init__(self, invariants: Optional[List[BaseInvariant]] = None) -> None:
        self.invariants = invariants if invariants is not None else [
            PaymentNoDuplicateInvariant(),
            PaymentBeforeReservationInvariant(),
            NotificationSingleDeliveryInvariant(),
            DeploymentNoDuplicateInvariant(),
        ]

    def evaluate_all(self, context: Dict[str, Any]) -> List[InvariantResult]:
        """Evaluates all registered invariants against the supplied context."""
        return [inv.evaluate(context) for inv in self.invariants]

    def all_satisfied(self, results: List[InvariantResult]) -> bool:
        """Returns True if every evaluated invariant passed."""
        return all(r.satisfied for r in results)

    def safety_invariants_pass(self, results: List[InvariantResult]) -> bool:
        """Returns True if every safety-critical invariant passed."""
        return all(r.satisfied for r in results if r.is_safety_invariant)

    def get_failed_safety_invariants(self, results: List[InvariantResult]) -> List[InvariantResult]:
        """Returns the list of failed safety-critical invariants."""
        return [r for r in results if r.is_safety_invariant and not r.satisfied]
