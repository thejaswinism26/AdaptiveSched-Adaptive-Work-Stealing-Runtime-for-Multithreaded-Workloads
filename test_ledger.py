"""Unit tests for SQLite durable effect ledger and transaction safety."""

import sqlite3
import pytest
from recoveryguard.checkpoints import CheckpointManager
from recoveryguard.database import DatabaseManager
from recoveryguard.ledger import DurableEffectLedger
from recoveryguard.models import (
    ActionIntent,
    ActionRiskLevel,
    EffectStatus,
    EvidenceSource,
    EvidenceState,
    RecoveryDecision,
    RecoveryOutcome,
)


@pytest.fixture
def ledger(tmp_path):
    db_file = tmp_path / "test_ledger.db"
    db = DatabaseManager(db_file)
    return DurableEffectLedger(db)


def test_record_intent_and_commit(ledger):
    intent = ActionIntent(
        action_id="act_001",
        task_id="task_A",
        operation_id="op_charge_order1",
        tool_name="charge_payment",
        parameters={"order_id": "order1", "amount": 250.0},
        risk_level=ActionRiskLevel.RISKY_MUTATION,
        checkpoint_id="cp_01",
    )

    record = ledger.record_intent(intent)
    assert record.status == EffectStatus.PENDING
    assert record.action_id == "act_001"
    assert record.operation_id == "op_charge_order1"

    # Verify status in database
    retrieved = ledger.get_effect("act_001")
    assert retrieved is not None
    assert retrieved.status == EffectStatus.PENDING

    # Transition to STARTED
    assert ledger.mark_started("act_001")
    retrieved = ledger.get_effect("act_001")
    assert retrieved.status == EffectStatus.STARTED

    # Transition to COMMITTED
    payload = {"charge_id": "ch_123", "status": "CAPTURED"}
    assert ledger.mark_committed("act_001", payload)
    retrieved = ledger.get_effect("act_001")
    assert retrieved.status == EffectStatus.COMMITTED
    assert retrieved.result == payload
    assert ledger.has_committed_operation("op_charge_order1")


def test_timeout_is_not_assumed_failed(ledger):
    intent = ActionIntent(
        action_id="act_002",
        task_id="task_B",
        operation_id="op_charge_order2",
        tool_name="charge_payment",
        parameters={"order_id": "order2"},
        risk_level=ActionRiskLevel.RISKY_MUTATION,
    )
    ledger.record_intent(intent)
    ledger.mark_started("act_002")
    ledger.mark_timeout("act_002", "Gateway timed out after 30s")

    record = ledger.get_effect("act_002")
    assert record is not None
    assert record.status == EffectStatus.TIMEOUT
    assert record.status != EffectStatus.FAILED
    assert "timed out" in record.error_message


def test_stable_operation_id_correlates_retries(ledger):
    op_id = "op_reserve_order3"

    # Attempt 1 (timed out)
    intent1 = ActionIntent(
        action_id="act_attempt_1",
        task_id="task_C",
        operation_id=op_id,
        tool_name="reserve_inventory",
        parameters={"order_id": "order3"},
    )
    ledger.record_intent(intent1)
    ledger.mark_timeout("act_attempt_1")

    # Attempt 2 (succeeded)
    intent2 = ActionIntent(
        action_id="act_attempt_2",
        task_id="task_C",
        operation_id=op_id,
        tool_name="reserve_inventory",
        parameters={"order_id": "order3"},
    )
    ledger.record_intent(intent2)
    ledger.mark_committed("act_attempt_2", {"reservation_id": "res_99"})

    # Query all attempts for operation_id
    effects = ledger.get_effects_for_operation(op_id)
    assert len(effects) == 2
    assert effects[0].action_id == "act_attempt_1"
    assert effects[0].status == EffectStatus.TIMEOUT
    assert effects[1].action_id == "act_attempt_2"
    assert effects[1].status == EffectStatus.COMMITTED
    assert ledger.has_committed_operation(op_id)


def test_ledger_survives_checkpoint_restoration(tmp_path):
    db_file = tmp_path / "ledger_persistent.db"
    cp_dir = tmp_path / "checkpoints"

    db = DatabaseManager(db_file)
    ledger = DurableEffectLedger(db)
    cp_mgr = CheckpointManager(cp_dir)

    # 1. Agent takes checkpoint 1
    cp1 = cp_mgr.create_checkpoint("task_D", step_index=1, agent_state={"step": 1, "balance": 100})

    # 2. Agent executes mutation and records in ledger
    intent = ActionIntent(
        action_id="act_003",
        task_id="task_D",
        operation_id="op_charge_order4",
        tool_name="charge_payment",
        parameters={"order_id": "order4"},
        checkpoint_id=cp1.checkpoint_id,
    )
    ledger.record_intent(intent)
    ledger.mark_committed("act_003", {"charge_id": "ch_456"})

    # 3. Simulate agent crash and rollback to Checkpoint 1
    restored_cp = cp_mgr.load_checkpoint(cp1.checkpoint_id)
    assert restored_cp.agent_state["step"] == 1

    # 4. Durable SQLite ledger still holds the committed effect record
    effect = ledger.get_effect("act_003")
    assert effect is not None
    assert effect.status == EffectStatus.COMMITTED
    assert ledger.has_committed_operation("op_charge_order4")


def test_transaction_rollback_on_error(ledger):
    intent = ActionIntent(
        action_id="act_fail_tx",
        task_id="task_E",
        operation_id="op_tx",
        tool_name="test_tool",
    )
    ledger.record_intent(intent)

    # Attempt an invalid transaction that will fail
    with pytest.raises(sqlite3.IntegrityError):
        with ledger.db.transaction() as cur:
            # Inserting duplicate action_id should violate UNIQUE constraint and trigger rollback
            cur.execute(
                "INSERT INTO effect_records (action_id, task_id, operation_id, tool_name, status, created_at, updated_at) "
                "VALUES ('act_fail_tx', 'task_E', 'op_tx', 'test_tool', 'PENDING', 0, 0)"
            )

    # Database remains consistent
    records = ledger.get_effects_for_operation("op_tx")
    assert len(records) == 1


def test_record_and_retrieve_recovery_decision(ledger):
    decision = RecoveryDecision(
        decision_id="dec_test_01",
        task_id="task_F",
        action_id="act_004",
        operation_id="op_charge_order5",
        outcome=RecoveryOutcome.RESUME,
        is_safe_to_execute=True,
        rationale="External charge verified by probe; skipping duplicate.",
        timestamp=1700000050.0,
        checkpoint_id="cp_02",
    )
    ledger.record_decision(decision)

    decisions = ledger.get_decisions_for_operation("op_charge_order5")
    assert len(decisions) == 1
    assert decisions[0].decision_id == "dec_test_01"
    assert decisions[0].outcome == RecoveryOutcome.RESUME
    assert decisions[0].is_safe_to_execute is True
