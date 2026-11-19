"""Tests for RecoveryGuard persistence, independent mock services, and failure simulation.

Verifies the five core research guarantees:
A. A notification can exist even when the restored agent believes it has not been sent.
B. The ledger survives checkpoint restoration.
C. External state survives checkpoint restoration.
D. A timeout is recorded as UNKNOWN unless independent evidence establishes the actual outcome.
E. Stable operation IDs can correlate multiple attempts.
"""

import time
import pytest

from recoveryguard import (
    ActionIntent,
    ActionRiskLevel,
    AgentCrashError,
    CheckpointManager,
    DatabaseManager,
    DeploymentRecordService,
    DurableEffectLedger,
    EffectStatus,
    EvidenceState,
    FaultInjector,
    FaultType,
    NotificationService,
    PaymentGateway,
    PreExecutionFailureError,
    RecoveryEngine,
    RecoveryOutcome,
    ServiceTimeoutError,
    StateVerifier,
)
from recoveryguard.invariants import InvariantSuite


# ---------------------------------------------------------------------------
# Test A: Notification exists even when restored agent believes unsent
# ---------------------------------------------------------------------------
def test_a_notification_exists_when_restored_agent_believes_unsent(tmp_path):
    """Proves: A notification can exist even when the restored agent believes

    it has not been sent (internal memory rollback vs external persistence).
    """
    cp_dir = tmp_path / "checkpoints"
    cp_mgr = CheckpointManager(cp_dir)
    notification_service = NotificationService()

    # Initial agent memory state at step 0
    agent_memory = {
        "step": 0,
        "order_id": "ord_1001",
        "notification_sent": False,
    }

    # Step 0: Agent saves checkpoint CP0
    cp0 = cp_mgr.create_checkpoint("task_notify", step_index=0, agent_state=agent_memory)

    # Step 1: Agent calls notification service externally
    res = notification_service.send_notification(
        recipient="user@example.com",
        message="Your package has shipped!",
        order_id="ord_1001",
        notification_id="ntf_1001",
    )
    assert res["status"] == "DELIVERED"
    assert notification_service.get_delivered_count() == 1

    # Simulate agent crash and reboot: agent restores from CP0
    restored_cp = cp_mgr.restore_checkpoint(cp0.checkpoint_id)
    assert restored_cp is not None
    restored_agent_memory = dict(restored_cp.agent_state)

    # The agent's restored internal memory believes notification was NEVER sent
    assert restored_agent_memory["notification_sent"] is False
    assert restored_agent_memory["step"] == 0

    # BUT the external notification service DID persist the delivered notification!
    probed = notification_service.probe_notification("ntf_1001")
    assert probed is not None
    assert probed["status"] == "DELIVERED"
    assert probed["recipient"] == "user@example.com"
    assert probed["order_id"] == "ord_1001"


# ---------------------------------------------------------------------------
# Test B: Ledger survives checkpoint restoration
# ---------------------------------------------------------------------------
def test_b_ledger_survives_checkpoint_restoration(tmp_path):
    """Proves: The durable SQLite effect ledger survives checkpoint restoration."""
    db_file = tmp_path / "durable_ledger.db"
    cp_dir = tmp_path / "checkpoints"

    db = DatabaseManager(db_file)
    ledger = DurableEffectLedger(db)
    cp_mgr = CheckpointManager(cp_dir)

    # Checkpoint 0 created before execution
    cp0 = cp_mgr.create_checkpoint("task_ledger", step_index=0, agent_state={"step": 0})

    # Record action intent and committed outcome in SQLite ledger
    intent = ActionIntent(
        action_id="act_tx_001",
        task_id="task_ledger",
        operation_id="op_deploy_api",
        tool_name="create_deployment",
        parameters={"service": "payment-api", "version": "2.1.0"},
        risk_level=ActionRiskLevel.RISKY_MUTATION,
        checkpoint_id=cp0.checkpoint_id,
        timestamp=time.time(),
    )
    record = ledger.record_intent(intent)
    assert record.status == EffectStatus.PENDING

    ledger.mark_started("act_tx_001")
    ledger.mark_committed("act_tx_001", {"deployment_id": "dep_001", "status": "DEPLOYED"})

    # Checkpoint 1 created
    cp1 = cp_mgr.create_checkpoint("task_ledger", step_index=1, agent_state={"step": 1})

    # Agent crashes and rolls back to Checkpoint 0
    restored = cp_mgr.restore_checkpoint(cp0.checkpoint_id)
    assert restored.step_index == 0
    assert restored.agent_state["step"] == 0

    # Verify that SQLite ledger was NOT rolled back or erased
    effect = ledger.get_effect("act_tx_001")
    assert effect is not None
    assert effect.status == EffectStatus.COMMITTED
    assert effect.result["deployment_id"] == "dep_001"
    assert ledger.has_committed_operation("op_deploy_api") is True


# ---------------------------------------------------------------------------
# Test C: External state survives checkpoint restoration
# ---------------------------------------------------------------------------
def test_c_external_state_survives_checkpoint_restoration(tmp_path):
    """Proves: External service state lives independently and survives checkpoint rollback."""
    cp_dir = tmp_path / "checkpoints"
    cp_mgr = CheckpointManager(cp_dir)

    # Independent external services
    deployment_service = DeploymentRecordService()
    notification_service = NotificationService()
    payment_gw = PaymentGateway()

    # Step 0 checkpoint
    cp0 = cp_mgr.create_checkpoint("task_services", step_index=0, agent_state={"step": 0})

    # Mutate all external services
    deployment_service.create_deployment(
        deployment_id="dep_release_9",
        service_name="order-service",
        version="v1.4.0",
        environment="production",
    )
    notification_service.send_notification(
        recipient="admin@platform.com",
        message="Production release deployed",
        order_id="rel_9",
        notification_id="ntf_rel_9",
    )
    payment_gw.charge(
        idempotency_key="pay_rel_9",
        order_id="rel_9",
        amount=0.0,  # Fake, non-monetary
        customer_id="cust_test",
    )

    # Roll back agent to checkpoint 0
    restored = cp_mgr.restore_checkpoint(cp0.checkpoint_id)
    assert restored.step_index == 0

    # Assert external state on all services remains intact
    dep_probe = deployment_service.probe_deployment("dep_release_9")
    assert dep_probe is not None
    assert dep_probe["status"] == "DEPLOYED"
    assert dep_probe["version"] == "v1.4.0"

    ntf_probe = notification_service.probe_notification("ntf_rel_9")
    assert ntf_probe is not None
    assert ntf_probe["status"] == "DELIVERED"

    pay_probe = payment_gw.probe("rel_9")
    assert pay_probe is not None
    assert pay_probe["status"] == "CAPTURED"


# ---------------------------------------------------------------------------
# Test D: Timeout recorded as UNKNOWN unless independent evidence proves outcome
# ---------------------------------------------------------------------------
def test_d_timeout_recorded_as_unknown_until_independent_evidence():
    """Proves: A timeout is recorded as UNKNOWN unless independent evidence

    establishes the actual outcome.
    """
    db = DatabaseManager(":memory:")
    ledger = DurableEffectLedger(db)
    verifier = StateVerifier(ledger)
    engine = RecoveryEngine(verifier, InvariantSuite())
    deployment_service = DeploymentRecordService()
    fault_injector = FaultInjector()

    operation_id = "op_deploy_search"
    action_id = "act_timeout_1"

    intent = ActionIntent(
        action_id=action_id,
        task_id="task_timeout",
        operation_id=operation_id,
        tool_name="create_deployment",
        parameters={"deployment_id": "dep_search_1", "service": "search"},
        risk_level=ActionRiskLevel.RISKY_MUTATION,
    )

    # Record intent in ledger before execution
    ledger.record_intent(intent)
    ledger.mark_started(action_id)

    # Inject timeout after service has successfully executed action
    fault_injector.trigger_timeout_after_execution(operation_id)

    # Execute with fault injection: service creates deployment, then network times out
    with pytest.raises(ServiceTimeoutError):
        fault_injector.execute_with_faults(
            target=operation_id,
            action_fn=lambda: deployment_service.create_deployment(
                deployment_id="dep_search_1",
                service_name="search",
                version="v3.0.0",
            ),
        )

    # Agent marks timeout in ledger (crucially: TIMEOUT, NOT FAILED)
    ledger.mark_timeout(action_id, "Service call timed out after 30s")
    record = ledger.get_effect(action_id)
    assert record.status == EffectStatus.TIMEOUT

    # Part 1: Probe UNAVAILABLE scenario
    # When no independent probe is available, synthesized state MUST be UNKNOWN
    deployment_service.set_probe_available(False)
    decision_unknown = engine.evaluate_gate(
        intent=intent,
        probe_data=deployment_service.probe("dep_search_1"),
        probe_supported=False,
    )
    # Never assume failed! Because risk_level is RISKY_MUTATION, must pause for human review
    assert decision_unknown.outcome == RecoveryOutcome.HUMAN_REVIEW
    assert decision_unknown.is_safe_to_execute is False
    assert any(e.state == EvidenceState.UNKNOWN for e in decision_unknown.evidence_list)

    # Part 2: Independent probe BECOMES AVAILABLE scenario
    # Now independent evidence probe is executed against the deployment service
    deployment_service.set_probe_available(True)
    probe_result = deployment_service.probe("dep_search_1")
    assert probe_result is not None
    assert probe_result["status"] == "DEPLOYED"

    # Evaluator now synthesizes the probe report with the timeout record
    decision_resolved = engine.evaluate_gate(
        intent=intent,
        probe_data=probe_result,
        probe_supported=True,
    )
    # The timeout is now resolved to CONFIRMED_SUCCESS, leading to safe RESUME
    assert decision_resolved.outcome == RecoveryOutcome.RESUME
    assert decision_resolved.is_safe_to_execute is True


# ---------------------------------------------------------------------------
# Test E: Stable operation IDs correlate multiple attempts
# ---------------------------------------------------------------------------
def test_e_stable_operation_ids_correlate_multiple_attempts():
    """Proves: Stable operation IDs correlate multiple attempts and preserve history."""
    db = DatabaseManager(":memory:")
    ledger = DurableEffectLedger(db)

    operation_id = "op_fulfill_order_500"

    # Attempt 1: In-flight timeout
    intent1 = ActionIntent(
        action_id="act_attempt_1",
        task_id="task_retries",
        operation_id=operation_id,
        tool_name="charge_payment",
        parameters={"order_id": "ord_500", "amount": 75.0},
    )
    ledger.record_attempt(intent1, status=EffectStatus.ATTEMPTED)
    ledger.mark_timeout("act_attempt_1", "Gateway connection dropped")

    # Attempt 2: Failure before execution (e.g. DNS failure)
    intent2 = ActionIntent(
        action_id="act_attempt_2",
        task_id="task_retries",
        operation_id=operation_id,
        tool_name="charge_payment",
        parameters={"order_id": "ord_500", "amount": 75.0},
    )
    ledger.record_attempt(intent2, status=EffectStatus.ATTEMPTED)
    ledger.mark_failed("act_attempt_2", "DNS resolution failed")

    # Attempt 3: Successful committed execution
    intent3 = ActionIntent(
        action_id="act_attempt_3",
        task_id="task_retries",
        operation_id=operation_id,
        tool_name="charge_payment",
        parameters={"order_id": "ord_500", "amount": 75.0},
    )
    ledger.record_attempt(intent3, status=EffectStatus.ATTEMPTED)
    ledger.mark_committed("act_attempt_3", {"charge_id": "ch_500", "status": "CAPTURED"})

    # Query all attempts for this stable operation ID
    attempts = ledger.get_attempts_for_operation(operation_id)
    assert len(attempts) == 3

    # Preserves historical sequence without overwriting
    assert attempts[0].action_id == "act_attempt_1"
    assert attempts[0].status == EffectStatus.TIMEOUT
    assert attempts[0].error_message == "Gateway connection dropped"

    assert attempts[1].action_id == "act_attempt_2"
    assert attempts[1].status == EffectStatus.FAILED
    assert attempts[1].error_message == "DNS resolution failed"

    assert attempts[2].action_id == "act_attempt_3"
    assert attempts[2].status == EffectStatus.COMMITTED
    assert attempts[2].result["charge_id"] == "ch_500"

    # Correlates to show the operation ultimately succeeded
    assert ledger.has_committed_operation(operation_id) is True


# ---------------------------------------------------------------------------
# Additional Fault Simulation Tests: Crash Before Record & Stale Evidence
# ---------------------------------------------------------------------------
def test_crash_before_agent_records_response():
    """Simulates agent crash immediately after service execution but before ledger commit."""
    db = DatabaseManager(":memory:")
    ledger = DurableEffectLedger(db)
    notification_service = NotificationService()
    fault_injector = FaultInjector()

    op_id = "op_send_welcome"
    act_id = "act_welcome_01"

    intent = ActionIntent(
        action_id=act_id,
        task_id="task_crash",
        operation_id=op_id,
        tool_name="send_notification",
        parameters={"recipient": "bob@example.com"},
    )
    ledger.record_intent(intent)
    ledger.mark_started(act_id)

    # Injects crash before agent records response
    fault_injector.trigger_crash_before_record(op_id)

    with pytest.raises(AgentCrashError):
        fault_injector.execute_with_faults(
            target=op_id,
            action_fn=lambda: notification_service.send_notification(
                recipient="bob@example.com",
                message="Welcome aboard!",
                notification_id="ntf_bob_01",
            ),
        )

    # External service DID execute and save the message
    assert notification_service.probe_notification("ntf_bob_01") is not None

    # But local ledger still only has STARTED (crash happened before recording committed)
    effect = ledger.get_effect(act_id)
    assert effect.status == EffectStatus.STARTED
    assert effect.result is None


def test_failure_before_external_action():
    """Simulates failure before external service action executes."""
    deployment_service = DeploymentRecordService()
    fault_injector = FaultInjector()
    op_id = "op_fail_early"

    fault_injector.trigger_failure_before_execution(op_id)

    with pytest.raises(PreExecutionFailureError):
        fault_injector.execute_with_faults(
            target=op_id,
            action_fn=lambda: deployment_service.create_deployment(
                deployment_id="dep_never_runs",
                service_name="billing",
                version="v1.0",
            ),
        )

    # External service was never touched
    assert deployment_service.probe_deployment("dep_never_runs") is None


def test_configurable_stale_evidence():
    """Verifies that configurable stale evidence can simulate delayed probe sync."""
    deployment_service = DeploymentRecordService()

    # Create new deployment v2
    deployment_service.create_deployment(
        deployment_id="dep_stale_test",
        service_name="auth",
        version="v2.0.0",
    )

    # Configure stale evidence probe returning previous version v1.0.0
    stale_record = {
        "deployment_id": "dep_stale_test",
        "service_name": "auth",
        "version": "v1.0.0",
        "status": "DEPLOYED",
    }
    deployment_service.set_stale_evidence(stale_record)

    probed = deployment_service.probe_deployment("dep_stale_test")
    assert probed["version"] == "v1.0.0"

    # Clear stale evidence -> probe returns true current state
    deployment_service.set_stale_evidence(None)
    probed_current = deployment_service.probe_deployment("dep_stale_test")
    assert probed_current["version"] == "v2.0.0"
