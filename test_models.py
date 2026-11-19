"""Unit tests for RecoveryGuard core models and domain types."""

import pytest
from recoveryguard.models import (
    ActionIntent,
    ActionRiskLevel,
    CheckpointMetadata,
    EffectEvidence,
    EffectRecord,
    EffectStatus,
    EvidenceSource,
    EvidenceState,
    InvariantResult,
    RecoveryDecision,
    RecoveryOutcome,
)


def test_evidence_state_values():
    assert EvidenceState.CONFIRMED_SUCCESS == "CONFIRMED_SUCCESS"
    assert EvidenceState.CONFIRMED_ABSENT == "CONFIRMED_ABSENT"
    assert EvidenceState.UNKNOWN == "UNKNOWN"
    assert EvidenceState.CONFLICTING == "CONFLICTING"


def test_recovery_outcome_values():
    expected = {"RESUME", "RECONCILE", "SAFE_RETRY", "COMPENSATE", "QUARANTINE", "HUMAN_REVIEW"}
    actual = {o.value for o in RecoveryOutcome}
    assert actual == expected


def test_action_intent_serialization():
    intent = ActionIntent(
        action_id="act_123",
        task_id="task_456",
        operation_id="op_789",
        tool_name="charge_payment",
        parameters={"order_id": "ord_999", "amount": 100.0},
        risk_level=ActionRiskLevel.RISKY_MUTATION,
        checkpoint_id="cp_01",
        timestamp=1700000000.0,
    )
    d = intent.to_dict()
    assert d["action_id"] == "act_123"
    assert d["risk_level"] == "RISKY_MUTATION"

    restored = ActionIntent.from_dict(d)
    assert restored == intent
    assert restored.risk_level == ActionRiskLevel.RISKY_MUTATION


def test_effect_evidence_creation():
    evidence = EffectEvidence(
        evidence_id="ev_001",
        action_id="act_123",
        task_id="task_456",
        operation_id="op_789",
        source=EvidenceSource.EXTERNAL_PROBE,
        state=EvidenceState.CONFIRMED_SUCCESS,
        details="Probe detected active charge.",
        observed_at=1700000005.0,
        raw_payload={"charge_id": "ch_99"},
        checkpoint_id="cp_01",
    )
    d = evidence.to_dict()
    restored = EffectEvidence.from_dict(d)
    assert restored.state == EvidenceState.CONFIRMED_SUCCESS
    assert restored.source == EvidenceSource.EXTERNAL_PROBE
    assert restored.raw_payload == {"charge_id": "ch_99"}


def test_recovery_decision_immutability():
    decision = RecoveryDecision(
        decision_id="dec_001",
        task_id="task_456",
        action_id="act_123",
        operation_id="op_789",
        outcome=RecoveryOutcome.HUMAN_REVIEW,
        is_safe_to_execute=False,
        rationale="Timeout on risky action requires human review.",
        timestamp=1700000010.0,
    )
    assert not decision.is_safe_to_execute
    assert decision.outcome == RecoveryOutcome.HUMAN_REVIEW

    with pytest.raises(Exception):
        # Frozen dataclass mutation must raise an exception
        decision.is_safe_to_execute = True  # type: ignore


def test_checkpoint_metadata_serialization():
    meta = CheckpointMetadata(
        checkpoint_id="cp_100",
        task_id="task_456",
        step_index=2,
        created_at=1700000000.0,
        state_hash="a1b2c3d4",
        agent_state={"step": 2, "balance": 500},
    )
    d = meta.to_dict()
    restored = CheckpointMetadata.from_dict(d)
    assert restored.checkpoint_id == "cp_100"
    assert restored.agent_state["balance"] == 500
