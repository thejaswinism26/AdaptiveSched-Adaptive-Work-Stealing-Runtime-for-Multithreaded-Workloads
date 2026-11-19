"""Focused tests for RecoveryGuard core recovery decision engine.

Verifies the 8 core epistemic and safety requirements:
1. Confirmed success prevents duplicate execution.
2. Confirmed absence permits a retry only when the invariant and retry-safety checks pass.
3. Unknown evidence blocks risky execution.
4. Conflicting evidence cannot authorize a blind retry.
5. Stale or unavailable probes cannot establish confirmed absence.
6. Missing ledger records do not establish that an action never occurred.
7. An irreversible notification is not incorrectly treated as compensatable.
8. Every recovery decision records a clear reason and its evidence.
"""

import time
import pytest

from recoveryguard import (
    ActionIntent,
    ActionRiskLevel,
    DatabaseManager,
    DeploymentRecordService,
    DurableEffectLedger,
    EffectEvidence,
    EffectStatus,
    EvidenceSource,
    EvidenceState,
    NotificationService,
    PaymentGateway,
    RecoveryEngine,
    RecoveryOutcome,
    StateVerifier,
)
from recoveryguard.invariants import (
    DeploymentNoDuplicateInvariant,
    InvariantSuite,
    NotificationSingleDeliveryInvariant,
    PaymentBeforeReservationInvariant,
    PaymentNoDuplicateInvariant,
    WorkflowCompletionCondition,
)


@pytest.fixture
def engine():
    db = DatabaseManager(":memory:")
    ledger = DurableEffectLedger(db)
    verifier = StateVerifier(ledger)
    suite = InvariantSuite([
        PaymentNoDuplicateInvariant(),
        PaymentBeforeReservationInvariant(),
        NotificationSingleDeliveryInvariant(),
        DeploymentNoDuplicateInvariant(),
    ])
    return RecoveryEngine(verifier, suite)


# ---------------------------------------------------------------------------
# Requirement 1: Confirmed success prevents duplicate execution
# ---------------------------------------------------------------------------
def test_1_confirmed_success_prevents_duplicate_execution(engine):
    """Proves: When an action's side effect is confirmed committed, the recovery gate

    yields RESUME (is_safe_to_execute=True to bypass re-execution), preventing duplicates.
    """
    intent = ActionIntent(
        action_id="act_pay_001",
        task_id="task_order_1",
        operation_id="op_charge_order_1",
        tool_name="charge_payment",
        parameters={"order_id": "ord_1", "amount": 150.0},
        risk_level=ActionRiskLevel.RISKY_MUTATION,
    )

    # External probe confirms the charge was already captured
    probe_data = {"charge_id": "ch_ord_1", "status": "CAPTURED", "order_id": "ord_1"}

    decision = engine.evaluate_gate(
        intent=intent,
        probe_data=probe_data,
        probe_supported=True,
    )

    assert decision.outcome == RecoveryOutcome.RESUME
    assert decision.is_safe_to_execute is True
    assert "already confirmed committed" in decision.rationale.lower()
    assert any(e.state == EvidenceState.CONFIRMED_SUCCESS for e in decision.evidence_list)


# ---------------------------------------------------------------------------
# Requirement 2: Confirmed absence permits retry ONLY when invariants pass
# ---------------------------------------------------------------------------
def test_2_confirmed_absence_permits_retry_only_when_invariants_pass(engine):
    """Proves: Confirmed absence permits SAFE_RETRY when invariants pass,

    but BLOCKS execution if a task safety invariant is violated.
    """
    # Case 2A: Action confirmed absent AND safety invariants pass -> SAFE_RETRY
    intent_safe = ActionIntent(
        action_id="act_deploy_001",
        task_id="task_deploy_1",
        operation_id="op_deploy_api_v1",
        tool_name="create_deployment",
        parameters={"deployment_id": "dep_1"},
        risk_level=ActionRiskLevel.RISKY_MUTATION,
    )
    # Probe confirms absence (None) and probe is fresh/available
    decision_safe = engine.evaluate_gate(
        intent=intent_safe,
        probe_data=None,
        probe_supported=True,
    )
    assert decision_safe.outcome == RecoveryOutcome.SAFE_RETRY
    assert decision_safe.is_safe_to_execute is True

    # Case 2B: Action confirmed absent BUT a safety invariant fails -> BLOCKED
    # e.g. Trying to reserve inventory when payment was refunded/absent
    intent_unsafe = ActionIntent(
        action_id="act_reserve_001",
        task_id="task_order_2",
        operation_id="op_reserve_order_2",
        tool_name="reserve_inventory",
        parameters={"order_id": "ord_2"},
        risk_level=ActionRiskLevel.RISKY_MUTATION,
    )
    # Reservation probe is absent, but invariant context shows reservation is held with NO payment
    broken_inv_context = {
        "order_id": "ord_2",
        "payment_probe": None,  # No payment
        "reservation_probe": {"status": "RESERVED"},  # But reservation exists!
    }
    decision_blocked = engine.evaluate_gate(
        intent=intent_unsafe,
        probe_data=None,
        probe_supported=True,
        invariant_context=broken_inv_context,
    )
    assert decision_blocked.is_safe_to_execute is False
    assert decision_blocked.outcome in (RecoveryOutcome.COMPENSATE, RecoveryOutcome.QUARANTINE)
    assert "invariant violation" in decision_blocked.rationale.lower()


# ---------------------------------------------------------------------------
# Requirement 3: Unknown evidence blocks risky execution
# ---------------------------------------------------------------------------
def test_3_unknown_evidence_blocks_risky_execution(engine):
    """Proves: Unknown evidence blocks RISKY_MUTATION actions, routing to HUMAN_REVIEW,

    while permitting SAFE_READ queries.
    """
    # 3A: RISKY_MUTATION with unknown status -> BLOCKED
    intent_risky = ActionIntent(
        action_id="act_risky_01",
        task_id="task_3",
        operation_id="op_risky_01",
        tool_name="charge_payment",
        parameters={"amount": 500.0},
        risk_level=ActionRiskLevel.RISKY_MUTATION,
    )
    decision_risky = engine.evaluate_gate(
        intent=intent_risky,
        probe_data=None,
        probe_supported=False,  # Unprobed service
    )
    assert decision_risky.is_safe_to_execute is False
    assert decision_risky.outcome == RecoveryOutcome.HUMAN_REVIEW
    assert "unknown for risky_mutation" in decision_risky.rationale.lower()

    # 3B: SAFE_READ with unknown status -> ALLOWED (read-only queries are safe)
    intent_read = ActionIntent(
        action_id="act_read_01",
        task_id="task_3",
        operation_id="op_read_01",
        tool_name="get_order_status",
        parameters={"order_id": "ord_3"},
        risk_level=ActionRiskLevel.SAFE_READ,
    )
    decision_read = engine.evaluate_gate(
        intent=intent_read,
        probe_data=None,
        probe_supported=False,
    )
    assert decision_read.is_safe_to_execute is True
    assert decision_read.outcome == RecoveryOutcome.SAFE_RETRY


# ---------------------------------------------------------------------------
# Requirement 4: Conflicting evidence cannot authorize a blind retry
# ---------------------------------------------------------------------------
def test_4_conflicting_evidence_cannot_authorize_blind_retry(engine):
    """Proves: When ledger and external probe contradict, the gate forbids blind retry

    and selects RECONCILE or HUMAN_REVIEW with is_safe_to_execute=False.
    """
    intent = ActionIntent(
        action_id="act_conflict_01",
        task_id="task_4",
        operation_id="op_conflict_01",
        tool_name="create_deployment",
        parameters={"deployment_id": "dep_conflict"},
        risk_level=ActionRiskLevel.RISKY_MUTATION,
    )

    # Durable ledger claims COMMITTED
    engine.verifier.ledger.record_intent(intent)
    engine.verifier.ledger.mark_committed(intent.action_id, {"status": "DEPLOYED"})

    # But fresh external probe confirms entity is ABSENT (e.g. wiped or wrong target)
    decision = engine.evaluate_gate(
        intent=intent,
        probe_data=None,
        probe_supported=True,
        probe_available=True,
    )

    assert decision.is_safe_to_execute is False
    assert decision.outcome in (RecoveryOutcome.RECONCILE, RecoveryOutcome.HUMAN_REVIEW)
    assert "contradiction" in decision.rationale.lower()
    assert any(e.state == EvidenceState.CONFLICTING for e in [
        EffectEvidence(
            evidence_id="syn", action_id=intent.action_id, task_id="t", operation_id="o",
            source=EvidenceSource.INVARIANT_CHECK, state=engine.verifier.synthesize_state(decision.evidence_list),
            details="", observed_at=0.0
        )
    ])


# ---------------------------------------------------------------------------
# Requirement 5: Stale or unavailable probes cannot establish confirmed absence
# ---------------------------------------------------------------------------
def test_5_stale_or_unavailable_probes_cannot_establish_confirmed_absence(engine):
    """Proves: A stale or unavailable probe yields UNKNOWN, never CONFIRMED_ABSENT."""
    intent = ActionIntent(
        action_id="act_stale_01",
        task_id="task_5",
        operation_id="op_stale_01",
        tool_name="charge_payment",
        parameters={"order_id": "ord_5"},
        risk_level=ActionRiskLevel.RISKY_MUTATION,
    )

    # 5A: Probe is unavailable -> status is UNKNOWN
    decision_unavail = engine.evaluate_gate(
        intent=intent,
        probe_data=None,
        probe_supported=True,
        probe_available=False,
    )
    assert decision_unavail.is_safe_to_execute is False
    assert decision_unavail.outcome == RecoveryOutcome.HUMAN_REVIEW
    syn_state_unavail = engine.verifier.synthesize_state(decision_unavail.evidence_list)
    assert syn_state_unavail == EvidenceState.UNKNOWN
    assert syn_state_unavail != EvidenceState.CONFIRMED_ABSENT

    # 5B: Probe is flagged stale -> status is UNKNOWN
    decision_stale = engine.evaluate_gate(
        intent=intent,
        probe_data={"order_id": "ord_5", "status": "UNKNOWN_OR_OLD", "is_stale": True},
        probe_supported=True,
        probe_available=True,
        probe_is_stale=True,
    )
    assert decision_stale.is_safe_to_execute is False
    syn_state_stale = engine.verifier.synthesize_state(decision_stale.evidence_list)
    assert syn_state_stale == EvidenceState.UNKNOWN
    assert syn_state_stale != EvidenceState.CONFIRMED_ABSENT


# ---------------------------------------------------------------------------
# Requirement 6: Missing ledger records do not establish that an action never occurred
# ---------------------------------------------------------------------------
def test_6_missing_ledger_records_do_not_establish_absence(engine):
    """Proves: If the agent crashed before logging (missing ledger record) and the probe

    is unavailable, the gate treats status as UNKNOWN, never CONFIRMED_ABSENT.
    """
    # Intent has never been seen by ledger
    intent = ActionIntent(
        action_id="act_unlogged_01",
        task_id="task_6",
        operation_id="op_unlogged_call",
        tool_name="create_deployment",
        parameters={"service": "auth"},
        risk_level=ActionRiskLevel.RISKY_MUTATION,
    )

    # Ledger has zero records for this operation
    assert len(engine.verifier.ledger.get_effects_for_operation("op_unlogged_call")) == 0

    # Service cannot be probed (probe_supported=False)
    decision = engine.evaluate_gate(
        intent=intent,
        probe_data=None,
        probe_supported=False,
    )

    # Missing ledger record + unsupported probe MUST be UNKNOWN, not CONFIRMED_ABSENT
    syn_state = engine.verifier.synthesize_state(decision.evidence_list)
    assert syn_state == EvidenceState.UNKNOWN
    assert syn_state != EvidenceState.CONFIRMED_ABSENT
    assert decision.is_safe_to_execute is False
    assert decision.outcome == RecoveryOutcome.HUMAN_REVIEW


# ---------------------------------------------------------------------------
# Requirement 7: Irreversible notifications not treated as compensatable
# ---------------------------------------------------------------------------
def test_7_irreversible_notification_not_treated_as_compensatable(engine):
    """Proves: When an invariant fails on an irreversible action (notification),

    the gate yields QUARANTINE instead of COMPENSATE (cannot unsend messages).
    """
    intent = ActionIntent(
        action_id="act_notif_inv_fail",
        task_id="task_7",
        operation_id="op_notify_user_7",
        tool_name="send_notification",
        parameters={"recipient": "customer@example.com"},
        risk_level=ActionRiskLevel.RISKY_MUTATION,
        is_compensatable=False,  # Notifications cannot be undone!
    )

    # Invariant context shows duplicate notifications already delivered (> 1)
    violating_ctx = {
        "operation_id": "op_notify_user_7",
        "notification_probe": [
            {"notification_id": "msg_1", "status": "DELIVERED"},
            {"notification_id": "msg_2", "status": "DELIVERED"},
        ],
        "probe_available": True,
    }

    decision = engine.evaluate_gate(
        intent=intent,
        probe_data=violating_ctx["notification_probe"],
        probe_supported=True,
        invariant_context=violating_ctx,
    )

    # Must be QUARANTINE (or HUMAN_REVIEW), NEVER COMPENSATE!
    assert decision.is_safe_to_execute is False
    assert decision.outcome == RecoveryOutcome.QUARANTINE
    assert decision.outcome != RecoveryOutcome.COMPENSATE
    assert "irreversible" in decision.rationale.lower()


# ---------------------------------------------------------------------------
# Requirement 8: Every decision records clear reason and its evidence
# ---------------------------------------------------------------------------
def test_8_every_recovery_decision_records_clear_reason_and_evidence(engine):
    """Proves: Every recovery gate decision stores a detailed rationale, the list of

    evidence considered, invariant evaluation results, and persists to the durable ledger.
    """
    intent = ActionIntent(
        action_id="act_audit_01",
        task_id="task_8",
        operation_id="op_audit_01",
        tool_name="charge_payment",
        parameters={"order_id": "ord_8", "amount": 99.0},
        risk_level=ActionRiskLevel.RISKY_MUTATION,
    )

    decision = engine.evaluate_gate(
        intent=intent,
        probe_data={"status": "CAPTURED", "order_id": "ord_8"},
        probe_supported=True,
    )

    # 1. Non-empty clear rationale explaining why allowed or blocked
    assert decision.rationale is not None
    assert len(decision.rationale) > 20
    assert "bypassing duplicate" in decision.rationale.lower()

    # 2. Complete evidence list attached
    assert len(decision.evidence_list) >= 2
    for ev in decision.evidence_list:
        assert ev.source in (EvidenceSource.LOCAL_LEDGER, EvidenceSource.EXTERNAL_PROBE)
        assert ev.state in (EvidenceState.CONFIRMED_SUCCESS, EvidenceState.CONFIRMED_ABSENT, EvidenceState.UNKNOWN)
        assert len(ev.details) > 0

    # 3. Invariants checked list attached
    assert len(decision.invariants_checked) > 0
    for inv in decision.invariants_checked:
        assert inv.invariant_id is not None
        assert inv.name is not None
        assert isinstance(inv.satisfied, bool)
        assert len(inv.details) > 0

    # 4. Durable persistence in effect ledger
    persisted_decisions = engine.verifier.ledger.get_decisions_for_operation("op_audit_01")
    assert len(persisted_decisions) == 1
    assert persisted_decisions[0].decision_id == decision.decision_id
    assert persisted_decisions[0].outcome == RecoveryOutcome.RESUME
    assert persisted_decisions[0].is_safe_to_execute is True


# ---------------------------------------------------------------------------
# Distinction test: Safety invariants vs Workflow completion conditions
# ---------------------------------------------------------------------------
def test_safety_invariant_vs_workflow_completion_condition(engine):
    """Proves: Incomplete workflow completion conditions do not trigger quarantine or

    compensation when safety invariants are satisfied.
    """
    suite = InvariantSuite([
        PaymentNoDuplicateInvariant(),
        WorkflowCompletionCondition(condition_id="COND_STEP_PROGRESS", name="Step Progress"),
    ])
    engine_with_progress = RecoveryEngine(engine.verifier, suite)

    intent = ActionIntent(
        action_id="act_step_2",
        task_id="task_progress",
        operation_id="op_step_2",
        tool_name="charge_payment",
        parameters={"order_id": "ord_p"},
        risk_level=ActionRiskLevel.RISKY_MUTATION,
    )

    # Context shows workflow is incomplete: 1 of 3 steps completed
    ctx = {
        "order_id": "ord_p",
        "operation_id": "op_step_2",
        "completed_steps": 1,
        "total_steps": 3,
        "payment_probe": None,
    }

    decision = engine_with_progress.evaluate_gate(
        intent=intent,
        probe_data=None,
        probe_supported=True,
        invariant_context=ctx,
    )

    # Incomplete workflow condition did NOT fail safety:
    # All safety invariants passed, entity confirmed absent -> safe to proceed with step 2!
    assert decision.is_safe_to_execute is True
    assert decision.outcome == RecoveryOutcome.SAFE_RETRY
