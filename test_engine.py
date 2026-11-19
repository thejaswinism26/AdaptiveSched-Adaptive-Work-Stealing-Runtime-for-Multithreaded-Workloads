"""Tests for state verification and recovery decision engine."""

import pytest
from recoveryguard.database import DatabaseManager
from recoveryguard.invariants import InvariantSuite, NoDuplicateChargeInvariant
from recoveryguard.ledger import DurableEffectLedger
from recoveryguard.models import (
    ActionIntent,
    ActionRiskLevel,
    EvidenceState,
    RecoveryOutcome,
)
from recoveryguard.recovery_engine import RecoveryEngine
from recoveryguard.verifier import StateVerifier


@pytest.fixture
def engine():
    db = DatabaseManager(":memory:")
    ledger = DurableEffectLedger(db)
    verifier = StateVerifier(ledger)
    return RecoveryEngine(verifier, InvariantSuite())


def test_confirmed_success_yields_resume(engine):
    intent = ActionIntent(
        action_id="act_resume_test",
        task_id="task_1",
        operation_id="op_charge_1",
        tool_name="charge_payment",
        parameters={"order_id": "ord_100"},
        risk_level=ActionRiskLevel.RISKY_MUTATION,
    )
    # External probe finds existing captured charge
    probe_data = {"charge_id": "ch_100", "status": "CAPTURED", "order_id": "ord_100"}

    decision = engine.evaluate_gate(intent, probe_data=probe_data, probe_supported=True)
    assert decision.outcome == RecoveryOutcome.RESUME
    assert decision.is_safe_to_execute is True


def test_confirmed_absent_yields_safe_retry(engine):
    intent = ActionIntent(
        action_id="act_retry_test",
        task_id="task_1",
        operation_id="op_charge_2",
        tool_name="charge_payment",
        parameters={"order_id": "ord_200"},
        risk_level=ActionRiskLevel.RISKY_MUTATION,
    )
    # External probe confirms entity is absent
    decision = engine.evaluate_gate(intent, probe_data=None, probe_supported=True)
    assert decision.outcome == RecoveryOutcome.SAFE_RETRY
    assert decision.is_safe_to_execute is True


def test_unknown_risky_mutation_yields_human_review(engine):
    intent = ActionIntent(
        action_id="act_unknown_test",
        task_id="task_1",
        operation_id="op_charge_3",
        tool_name="charge_payment",
        parameters={"order_id": "ord_300"},
        risk_level=ActionRiskLevel.RISKY_MUTATION,
    )
    # Service does not support probes, and ledger has no record -> status UNKNOWN
    decision = engine.evaluate_gate(intent, probe_data=None, probe_supported=False)
    assert decision.outcome == RecoveryOutcome.HUMAN_REVIEW
    assert decision.is_safe_to_execute is False


def test_conflicting_evidence_yields_quarantine(engine):
    intent = ActionIntent(
        action_id="act_conflict_test",
        task_id="task_1",
        operation_id="op_charge_4",
        tool_name="charge_payment",
        parameters={"order_id": "ord_400"},
        risk_level=ActionRiskLevel.RISKY_MUTATION,
    )
    # Ledger says COMMITTED
    engine.verifier.ledger.record_intent(intent)
    engine.verifier.ledger.mark_committed(intent.action_id, {"status": "SUCCESS"})

    # But external probe confirms ABSENT (e.g. data loss on external system or wrong database target)
    decision = engine.evaluate_gate(intent, probe_data=None, probe_supported=True)
    assert decision.outcome in (RecoveryOutcome.RECONCILE, RecoveryOutcome.QUARANTINE)
    assert decision.is_safe_to_execute is False


def test_invariant_violation_yields_compensate(engine):
    intent = ActionIntent(
        action_id="act_inv_test",
        task_id="task_1",
        operation_id="op_reserve_5",
        tool_name="reserve_inventory",
        parameters={"order_id": "ord_500"},
        risk_level=ActionRiskLevel.RISKY_MUTATION,
    )
    # External state: Reservation is held, but payment is None (breaks alignment invariant)
    inv_ctx = {
        "order_id": "ord_500",
        "payment_probe": None,
        "reservation_probe": {"status": "RESERVED"},
    }
    decision = engine.evaluate_gate(
        intent,
        probe_data={"status": "RESERVED"},
        probe_supported=True,
        invariant_context=inv_ctx,
    )
    assert decision.outcome == RecoveryOutcome.COMPENSATE
    assert decision.is_safe_to_execute is False
