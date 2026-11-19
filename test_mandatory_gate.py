"""Tests for the mandatory post-restore recovery gate in RecoveryGuard.

Verifies:
1. A restored agent cannot execute a side effect before verification.
2. A verified, authorized action can execute.
3. Authorization for one operation cannot authorize another operation.
4. HUMAN_REVIEW and QUARANTINE block ordinary tools.
5. RECONCILE permits only designated reconciliation tools.
6. A verification exception fails closed.
7. Restoring an older checkpoint does not reuse stale authorization.
8. The gate is checked even when a caller invokes the tool interface directly.
"""

from unittest.mock import MagicMock
import pytest

from recoveryguard import (
    ActionRiskLevel,
    AgentWorkflow,
    CheckpointManager,
    DatabaseManager,
    DeploymentRecordService,
    DurableEffectLedger,
    GateDeniedError,
    InventoryStore,
    NotificationService,
    PaymentGateway,
    RecoveryEngine,
    RecoveryOutcome,
    RecoveryVerificationError,
    StateVerifier,
)
from recoveryguard.invariants import (
    DeploymentNoDuplicateInvariant,
    InvariantSuite,
    NotificationSingleDeliveryInvariant,
    PaymentBeforeReservationInvariant,
    PaymentNoDuplicateInvariant,
)


@pytest.fixture
def agent_fixture(tmp_path):
    db = DatabaseManager(tmp_path / "gate_test.db")
    ledger = DurableEffectLedger(db)
    checkpoints = CheckpointManager(tmp_path / "checkpoints")
    verifier = StateVerifier(ledger)
    suite = InvariantSuite([
        PaymentNoDuplicateInvariant(),
        PaymentBeforeReservationInvariant(),
        NotificationSingleDeliveryInvariant(),
        DeploymentNoDuplicateInvariant(),
    ])
    engine = RecoveryEngine(verifier, suite)
    payment_gw = PaymentGateway()
    inventory_store = InventoryStore()
    notifications = NotificationService()

    agent = AgentWorkflow(
        task_id="task_gate_audit",
        ledger=ledger,
        checkpoints=checkpoints,
        recovery_engine=engine,
        payment_gw=payment_gw,
        inventory_store=inventory_store,
        notifications=notifications,
        mode="recovery_guard",
    )
    agent.init_task(order_id="ord_gate_1", customer_id="cust_1", item_id="laptop", amount=100.0)
    cp_id = agent.save_checkpoint()
    return agent, cp_id, payment_gw


# ---------------------------------------------------------------------------
# Test 1: Restored agent cannot execute side effect before verification
# ---------------------------------------------------------------------------
def test_1_restored_agent_cannot_execute_side_effect_before_verification(agent_fixture):
    """Proves: A restored agent cannot execute side-effecting tools before verification."""
    agent, cp_id, _ = agent_fixture

    # Restore checkpoint
    assert agent.restore_checkpoint(cp_id) is True
    assert agent.requires_verification() is True
    assert agent.get_active_authorization() is None

    # Attempt side-effecting tool execution before running the recovery gate
    executed = False

    def do_side_effect():
        nonlocal executed
        executed = True
        return {"status": "SUCCESS"}

    # Attempting to execute directly raises GateDeniedError
    with pytest.raises(GateDeniedError) as exc_info:
        agent.execute_tool(
            operation_id="op_charge_101",
            tool_name="charge_payment",
            risk_level=ActionRiskLevel.RISKY_MUTATION,
            tool_fn=do_side_effect,
            raise_on_denial=True,
        )

    assert "requires recovery verification" in str(exc_info.value).lower()
    assert executed is False  # Side effect was strictly prevented!


# ---------------------------------------------------------------------------
# Test 2: Verified, authorized action can execute
# ---------------------------------------------------------------------------
def test_2_verified_authorized_action_can_execute(agent_fixture):
    """Proves: Once verified and authorized by the gate, the action can execute."""
    agent, cp_id, payment_gw = agent_fixture

    agent.restore_checkpoint(cp_id)

    # 1. Run mandatory recovery verification gate
    decision = agent.run_recovery_gate(
        operation_id="op_charge_102",
        tool_name="charge_payment",
        risk_level=ActionRiskLevel.RISKY_MUTATION,
        probe_fn=lambda: payment_gw.probe("ord_gate_1"),
        probe_supported=True,
    )
    assert decision.outcome == RecoveryOutcome.SAFE_RETRY
    assert decision.is_safe_to_execute is True
    assert agent.get_active_authorization() is not None

    # 2. Execute authorized tool
    executed = False

    def do_charge():
        nonlocal executed
        executed = True
        return payment_gw.charge("idemp_102", "ord_gate_1", 100.0, "cust_1")

    res = agent.execute_tool(
        operation_id="op_charge_102",
        tool_name="charge_payment",
        risk_level=ActionRiskLevel.RISKY_MUTATION,
        tool_fn=do_charge,
        probe_fn=lambda: payment_gw.probe("ord_gate_1"),
        raise_on_denial=True,
    )

    assert executed is True
    assert res["authorized"] is True
    assert res["outcome"] == "EXECUTED_SAFE"
    assert payment_gw.probe("ord_gate_1")["status"] == "CAPTURED"


# ---------------------------------------------------------------------------
# Test 3: Authorization for one operation cannot authorize another operation
# ---------------------------------------------------------------------------
def test_3_authorization_scoped_to_specific_operation(agent_fixture):
    """Proves: Scoped authorization for operation A cannot be used to execute operation B."""
    agent, cp_id, payment_gw = agent_fixture
    agent.restore_checkpoint(cp_id)

    # Authorize operation A
    agent.run_recovery_gate(
        operation_id="op_charge_authorized",
        tool_name="charge_payment",
        risk_level=ActionRiskLevel.RISKY_MUTATION,
        probe_fn=lambda: payment_gw.probe("ord_gate_1"),
        probe_supported=True,
    )

    # Attempt to execute operation B with the token for operation A
    with pytest.raises(GateDeniedError) as exc_info:
        agent.execute_tool(
            operation_id="op_ship_unauthorized",
            tool_name="ship_item",
            risk_level=ActionRiskLevel.RISKY_MUTATION,
            tool_fn=lambda: {"shipped": True},
            raise_on_denial=True,
        )

    assert "cannot authorize 'op_ship_unauthorized'" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 4: HUMAN_REVIEW and QUARANTINE block ordinary tools
# ---------------------------------------------------------------------------
def test_4_human_review_and_quarantine_block_ordinary_tools(agent_fixture):
    """Proves: Gate outcomes HUMAN_REVIEW and QUARANTINE deny ordinary tools."""
    agent, cp_id, _ = agent_fixture
    agent.restore_checkpoint(cp_id)

    # Case 4A: Unknown evidence on risky mutation yields HUMAN_REVIEW
    decision_review = agent.run_recovery_gate(
        operation_id="op_unknown_timeout",
        tool_name="charge_payment",
        risk_level=ActionRiskLevel.RISKY_MUTATION,
        probe_fn=None,
        probe_supported=False,
    )
    assert decision_review.outcome == RecoveryOutcome.HUMAN_REVIEW

    with pytest.raises(GateDeniedError):
        agent.execute_tool(
            operation_id="op_unknown_timeout",
            tool_name="charge_payment",
            risk_level=ActionRiskLevel.RISKY_MUTATION,
            tool_fn=lambda: {"status": "ATTEMPTED"},
            raise_on_denial=True,
        )

    # Case 4B: Invariant failure on irreversible tool yields QUARANTINE
    violating_ctx = {
        "operation_id": "op_duplicate_notif",
        "notification_probe": [{"id": 1}, {"id": 2}],
    }
    decision_quarantine = agent.run_recovery_gate(
        operation_id="op_duplicate_notif",
        tool_name="send_notification",
        risk_level=ActionRiskLevel.RISKY_MUTATION,
        probe_supported=True,
        invariant_context=violating_ctx,
    )
    assert decision_quarantine.outcome == RecoveryOutcome.QUARANTINE

    with pytest.raises(GateDeniedError):
        agent.execute_tool(
            operation_id="op_duplicate_notif",
            tool_name="send_notification",
            risk_level=ActionRiskLevel.RISKY_MUTATION,
            tool_fn=lambda: {"sent": True},
            raise_on_denial=True,
        )


# ---------------------------------------------------------------------------
# Test 5: RECONCILE permits only designated reconciliation tools
# ---------------------------------------------------------------------------
def test_5_reconcile_permits_only_designated_reconciliation_tools(agent_fixture):
    """Proves: Under RECONCILE, ordinary tools are denied; only reconciliation tools run."""
    agent, cp_id, payment_gw = agent_fixture
    agent.restore_checkpoint(cp_id)

    # Induce contradiction: Ledger claims committed, but external probe is None
    action_intent = agent.ledger.record_intent(
        agent_fixture[0].__class__.__name__ and None or
        __import__("recoveryguard").ActionIntent(
            action_id="act_reconcile_1",
            task_id=agent.task_id,
            operation_id="op_divergent_state",
            tool_name="charge_payment",
        )
    )
    agent.ledger.mark_committed(action_intent.action_id, {"status": "SUCCESS"})

    decision = agent.run_recovery_gate(
        operation_id="op_divergent_state",
        tool_name="charge_payment",
        risk_level=ActionRiskLevel.RISKY_MUTATION,
        probe_fn=lambda: None,  # External entity missing
        probe_supported=True,
        conflict_policy="reconcile",
    )
    assert decision.outcome == RecoveryOutcome.RECONCILE

    # 1. Attempt ordinary side-effecting tool -> DENIED
    with pytest.raises(GateDeniedError) as exc_info:
        agent.execute_tool(
            operation_id="op_divergent_state",
            tool_name="charge_payment",
            risk_level=ActionRiskLevel.RISKY_MUTATION,
            tool_fn=lambda: {"mutated": True},
            is_reconciliation=False,
            raise_on_denial=True,
        )
    assert "only designated reconciliation operations are permitted" in str(exc_info.value)

    # 2. Attempt designated reconciliation operation -> PERMITTED
    reconciled = False

    def do_reconcile():
        nonlocal reconciled
        reconciled = True
        return {"reconciled": True}

    res = agent.execute_tool(
        operation_id="op_divergent_state",
        tool_name="reconcile_state",
        risk_level=ActionRiskLevel.RISKY_MUTATION,
        tool_fn=do_reconcile,
        is_reconciliation=True,
        raise_on_denial=True,
    )
    assert reconciled is True
    assert res["outcome"] == "RECONCILED"


# ---------------------------------------------------------------------------
# Test 6: Verification exception fails closed
# ---------------------------------------------------------------------------
def test_6_verification_exception_fails_closed(agent_fixture):
    """Proves: If an unexpected error occurs during verification, execution fails closed."""
    agent, cp_id, _ = agent_fixture
    agent.restore_checkpoint(cp_id)

    # Mock recovery engine to raise unexpected hardware/DB exception
    agent.recovery_engine.evaluate_gate = MagicMock(
        side_effect=RuntimeError("Simulated hardware I/O disk failure")
    )

    with pytest.raises(RecoveryVerificationError) as exc_info:
        agent.run_recovery_gate(
            operation_id="op_fail_closed",
            tool_name="charge_payment",
        )
    assert "failing closed" in str(exc_info.value).lower()
    assert agent.get_active_authorization() is None

    # Subsequent tool call is strictly blocked
    with pytest.raises(GateDeniedError):
        agent.execute_tool(
            operation_id="op_fail_closed",
            tool_name="charge_payment",
            risk_level=ActionRiskLevel.RISKY_MUTATION,
            tool_fn=lambda: {"paid": True},
        )


# ---------------------------------------------------------------------------
# Test 7: Restoring an older checkpoint does not reuse stale authorization
# ---------------------------------------------------------------------------
def test_7_restoring_checkpoint_wipes_stale_authorization(agent_fixture):
    """Proves: Checkpoint restoration wipes all active authorizations and mandates re-check."""
    agent, cp_id, payment_gw = agent_fixture
    agent.restore_checkpoint(cp_id)

    # Authorize operation
    agent.run_recovery_gate(
        operation_id="op_charge_stale_test",
        tool_name="charge_payment",
        probe_fn=lambda: None,
        probe_supported=True,
    )
    assert agent.get_active_authorization() is not None

    # Now restore an older checkpoint
    agent.restore_checkpoint(cp_id)

    # Active authorization MUST be wiped
    assert agent.get_active_authorization() is None
    assert agent.requires_verification() is True

    # Attempting execution without fresh verification is denied
    with pytest.raises(GateDeniedError):
        agent.execute_tool(
            operation_id="op_charge_stale_test",
            tool_name="charge_payment",
            risk_level=ActionRiskLevel.RISKY_MUTATION,
            tool_fn=lambda: {"paid": True},
        )


# ---------------------------------------------------------------------------
# Test 8: Gate is checked even when caller invokes tool interface directly
# ---------------------------------------------------------------------------
def test_8_gate_checked_when_caller_invokes_tool_interface_directly(agent_fixture):
    """Proves: Direct invocation of agent.execute_tool cannot bypass the gate."""
    agent, cp_id, _ = agent_fixture
    agent.restore_checkpoint(cp_id)

    # Directly invoking execute_tool without running gate or workflow
    with pytest.raises(GateDeniedError) as exc_info:
        agent.execute_tool(
            operation_id="op_direct_bypass_attempt",
            tool_name="charge_payment",
            risk_level=ActionRiskLevel.RISKY_MUTATION,
            tool_fn=lambda: {"bypassed": True},
            raise_on_denial=True,
        )

    assert "requires recovery verification" in str(exc_info.value).lower()
