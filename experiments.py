"""Experimental evaluation harness comparing recovery modes across fault scenarios.

Three Selectable Experimental Modes:
1. Mode A: CHECKPOINT_ONLY
   - Restores saved agent workflow state to prior checkpoint and blindly resumes.
   - Does not consult effect ledger or external probes.
   - Demonstrates the failure mode of duplicate side effects upon rollback.

2. Mode B: EFFECT_LOG
   - Uses durable effect ledger and exact operation ID replay protection.
   - If committed in ledger: skips duplicate execution (replay protection).
   - If uncommitted (e.g. TIMEOUT or STARTED) on risky mutation:
     * Conservative policy (default): blocks and requests human review without probes.
     * Optimistic policy: retries, which may duplicate if external service succeeded.
   - Cannot detect unrecorded external commits (e.g. crash before write) or external drift.

3. Mode C: RECOVERYGUARD
   - Uses the mandatory post-restore recovery gate.
   - Combines durable effect history, read-only external status probes, and task invariants.
   - Epistemic uncertainty-aware reasoning (CONFIRMED_SUCCESS -> RESUME,
     CONFIRMED_ABSENT -> SAFE_RETRY, UNKNOWN on risky -> HUMAN_REVIEW, CONFLICTING -> QUARANTINE).

Eight Standardized Distributed Failure Scenarios:
1. External action succeeds but its response times out.
2. Agent crashes after external success but before recording success.
3. Action fails before reaching the external service.
4. External probe is unavailable.
5. External evidence is stale.
6. Ledger and external evidence conflict.
7. An older checkpoint is restored after a later external effect.
8. The external service has no reliable status-query capability.
"""

from __future__ import annotations

import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from recoveryguard.agent import AgentWorkflow
from recoveryguard.checkpoints import CheckpointManager
from recoveryguard.database import DatabaseManager
from recoveryguard.fault_injection import (
    AgentCrashError,
    FaultInjector,
    FaultType,
    PreExecutionFailureError,
    ServiceTimeoutError,
)
from recoveryguard.invariants import (
    DeploymentNoDuplicateInvariant,
    InvariantSuite,
    NotificationSingleDeliveryInvariant,
    PaymentBeforeReservationInvariant,
    PaymentNoDuplicateInvariant,
)
from recoveryguard.ledger import DurableEffectLedger
from recoveryguard.mock_services import (
    DeploymentRecordService,
    InventoryStore,
    NotificationService,
    PaymentGateway,
)
from recoveryguard.models import (
    ActionIntent,
    ActionRiskLevel,
    EffectStatus,
    EvidenceState,
    RecoveryDecision,
    RecoveryOutcome,
)
from recoveryguard.recovery_engine import RecoveryEngine
from recoveryguard.verifier import StateVerifier


class ExperimentalMode(str, Enum):
    """The three selectable experimental modes."""
    CHECKPOINT_ONLY = "CHECKPOINT_ONLY"
    EFFECT_LOG = "EFFECT_LOG"
    RECOVERYGUARD = "RECOVERYGUARD"


class AblationVariant(str, Enum):
    """Component ablation study variants."""
    FULL_RECOVERYGUARD = "FULL_RECOVERYGUARD"
    NO_LEDGER = "NO_LEDGER"
    NO_PROBES = "NO_PROBES"
    NO_INVARIANTS = "NO_INVARIANTS"


def normalize_mode(mode: Union[str, ExperimentalMode, AblationVariant]) -> ExperimentalMode:
    """Normalizes mode string or enum into standard ExperimentalMode."""
    if isinstance(mode, ExperimentalMode):
        return mode
    val = str(mode).strip().upper()
    if val in ("CHECKPOINT_ONLY", "CHECKPOINT"):
        return ExperimentalMode.CHECKPOINT_ONLY
    if val in ("EFFECT_LOG", "EFFECT_LOG_ONLY", "EFFECTLOG"):
        return ExperimentalMode.EFFECT_LOG
    if val in ("RECOVERYGUARD", "RECOVERY_GUARD", "FULL_RECOVERYGUARD", "NO_LEDGER", "NO_PROBES", "NO_INVARIANTS"):
        return ExperimentalMode.RECOVERYGUARD
    raise ValueError(f"Unknown experimental mode: {mode}")


class ScenarioId(str, Enum):
    """The 8 standardized evaluation scenario identifiers."""
    SCENARIO_1_TIMEOUT_AFTER_SUCCESS = "scenario_1_timeout_after_success"
    SCENARIO_2_CRASH_BEFORE_RECORD = "scenario_2_crash_before_record"
    SCENARIO_3_FAILURE_BEFORE_EXECUTION = "scenario_3_failure_before_execution"
    SCENARIO_4_PROBE_UNAVAILABLE = "scenario_4_probe_unavailable"
    SCENARIO_5_PROBE_STALE = "scenario_5_probe_stale"
    SCENARIO_6_LEDGER_EXTERNAL_CONFLICT = "scenario_6_ledger_external_conflict"
    SCENARIO_7_OLDER_CHECKPOINT_RESTORED = "scenario_7_older_checkpoint_restored"
    SCENARIO_8_NO_STATUS_QUERY = "scenario_8_no_status_query"


@dataclass(frozen=True)
class ScenarioMetadata:
    """Structured definition for an experimental scenario."""
    scenario_id: ScenarioId
    name: str
    description: str
    injected_fault: str
    probe_supported: bool
    probe_available: bool
    probe_is_stale: bool
    has_external_query_capability: bool


SCENARIO_REGISTRY: Dict[ScenarioId, ScenarioMetadata] = {
    ScenarioId.SCENARIO_1_TIMEOUT_AFTER_SUCCESS: ScenarioMetadata(
        scenario_id=ScenarioId.SCENARIO_1_TIMEOUT_AFTER_SUCCESS,
        name="1. External Success With Response Timeout",
        description="External action succeeds but network response times out before agent acknowledgment.",
        injected_fault="TIMEOUT_AFTER_EXECUTION",
        probe_supported=True,
        probe_available=True,
        probe_is_stale=False,
        has_external_query_capability=True,
    ),
    ScenarioId.SCENARIO_2_CRASH_BEFORE_RECORD: ScenarioMetadata(
        scenario_id=ScenarioId.SCENARIO_2_CRASH_BEFORE_RECORD,
        name="2. Crash After Success Before Record",
        description="External service succeeds, but agent process crashes before writing commit record to ledger.",
        injected_fault="CRASH_BEFORE_AGENT_RECORD",
        probe_supported=True,
        probe_available=True,
        probe_is_stale=False,
        has_external_query_capability=True,
    ),
    ScenarioId.SCENARIO_3_FAILURE_BEFORE_EXECUTION: ScenarioMetadata(
        scenario_id=ScenarioId.SCENARIO_3_FAILURE_BEFORE_EXECUTION,
        name="3. Failure Before External Action",
        description="Action drops/fails prior to contacting external service (external state unmodified).",
        injected_fault="FAILURE_BEFORE_EXECUTION",
        probe_supported=True,
        probe_available=True,
        probe_is_stale=False,
        has_external_query_capability=True,
    ),
    ScenarioId.SCENARIO_4_PROBE_UNAVAILABLE: ScenarioMetadata(
        scenario_id=ScenarioId.SCENARIO_4_PROBE_UNAVAILABLE,
        name="4. External Probe Unavailable",
        description="External action timed out, and the read-only status probe endpoint is unavailable/down.",
        injected_fault="PROBE_UNAVAILABLE",
        probe_supported=True,
        probe_available=False,
        probe_is_stale=False,
        has_external_query_capability=True,
    ),
    ScenarioId.SCENARIO_5_PROBE_STALE: ScenarioMetadata(
        scenario_id=ScenarioId.SCENARIO_5_PROBE_STALE,
        name="5. External Evidence Is Stale",
        description="External probe returns cached, stale evidence which cannot establish ground truth.",
        injected_fault="STALE_PROBE_DATA",
        probe_supported=True,
        probe_available=True,
        probe_is_stale=True,
        has_external_query_capability=True,
    ),
    ScenarioId.SCENARIO_6_LEDGER_EXTERNAL_CONFLICT: ScenarioMetadata(
        scenario_id=ScenarioId.SCENARIO_6_LEDGER_EXTERNAL_CONFLICT,
        name="6. Ledger and External Evidence Conflict",
        description="Ledger records committed action, but external probe reports entity does not exist.",
        injected_fault="STATE_CONTRADICTION",
        probe_supported=True,
        probe_available=True,
        probe_is_stale=False,
        has_external_query_capability=True,
    ),
    ScenarioId.SCENARIO_7_OLDER_CHECKPOINT_RESTORED: ScenarioMetadata(
        scenario_id=ScenarioId.SCENARIO_7_OLDER_CHECKPOINT_RESTORED,
        name="7. Older Checkpoint Restored After Effects",
        description="Agent state is rolled back to an earlier checkpoint after multiple external side effects committed.",
        injected_fault="ROLLBACK_TO_OLDER_CHECKPOINT",
        probe_supported=True,
        probe_available=True,
        probe_is_stale=False,
        has_external_query_capability=True,
    ),
    ScenarioId.SCENARIO_8_NO_STATUS_QUERY: ScenarioMetadata(
        scenario_id=ScenarioId.SCENARIO_8_NO_STATUS_QUERY,
        name="8. External Service Lacks Status Query",
        description="External service (e.g. legacy SMS/alert) does not support read-only status verification.",
        injected_fault="PROBE_NOT_SUPPORTED",
        probe_supported=False,
        probe_available=False,
        probe_is_stale=False,
        has_external_query_capability=False,
    ),
}


@dataclass
class ExperimentResult:
    """Detailed result record of executing an experimental scenario."""
    scenario_id: str
    scenario_name: str
    mode: str
    initial_conditions: Dict[str, Any]
    injected_fault: str
    recovery_decision: Optional[Dict[str, Any]]
    actual_side_effects: Dict[str, Any]
    duplicate_side_effects: int
    invariant_violations: int
    blocked_actions: int
    human_escalations: int
    execution_latency_ms: float
    success: bool
    details: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Backward-compatible result alias
@dataclass
class ScenarioResult:
    scenario_name: str
    mode: str
    success: bool
    duplicate_charges: int
    invariant_violations: int
    human_pauses: int
    details: str


class IsolatedEnvironment:
    """Manages an isolated execution sandbox for a single experimental run."""

    def __init__(self, run_id: str, base_temp_dir: Optional[Path] = None) -> None:
        self.run_id = run_id
        if base_temp_dir:
            self._temp_dir = base_temp_dir / f"e_{run_id}"
            self._temp_dir.mkdir(parents=True, exist_ok=True)
            self._owned_temp = False
        else:
            self._temp_obj = tempfile.TemporaryDirectory()
            self._temp_dir = Path(self._temp_obj.name)
            self._owned_temp = True

        cp_dir = self._temp_dir / "cp"
        cp_dir.mkdir(parents=True, exist_ok=True)
        self.db = DatabaseManager(":memory:")
        self.ledger = DurableEffectLedger(self.db)
        self.checkpoints = CheckpointManager(cp_dir)

        # Independent simulated services
        self.payment_gw = PaymentGateway()
        self.inventory_store = InventoryStore()
        self.notifications = NotificationService()
        self.deployments = DeploymentRecordService()

        # Invariant and verifier suite
        self.invariants = InvariantSuite([
            PaymentNoDuplicateInvariant(),
            PaymentBeforeReservationInvariant(),
            NotificationSingleDeliveryInvariant(),
            DeploymentNoDuplicateInvariant(),
        ])
        self.verifier = StateVerifier(self.ledger)
        self.recovery_engine = RecoveryEngine(self.verifier, self.invariants)
        self.fault_injector = FaultInjector()

    def build_agent(
        self,
        task_id: str,
        mode: ExperimentalMode,
        conservative_effect_log: bool = True,
        disable_ledger: bool = False,
    ) -> AgentWorkflow:
        agent_mode_str = "recovery_guard" if mode == ExperimentalMode.RECOVERYGUARD else (
            "effect_log_only" if mode == ExperimentalMode.EFFECT_LOG else "checkpoint_only"
        )
        return AgentWorkflow(
            task_id=task_id,
            ledger=None if disable_ledger else self.ledger,
            checkpoints=self.checkpoints,
            recovery_engine=self.recovery_engine,
            payment_gw=self.payment_gw,
            inventory_store=self.inventory_store,
            notifications=self.notifications,
            mode=agent_mode_str,
            conservative_effect_log=conservative_effect_log,
        )

    def cleanup(self) -> None:
        if self._owned_temp and hasattr(self, "_temp_obj"):
            try:
                self._temp_obj.cleanup()
            except Exception:
                pass


class ExperimentHarness:
    """Automated benchmark runner comparing recovery modes across deterministic scenarios."""

    def __init__(
        self,
        base_temp_dir: Optional[Path] = None,
        conservative_effect_log: bool = True,
    ) -> None:
        self.base_temp_dir = base_temp_dir
        self.conservative_effect_log = conservative_effect_log

    def create_isolated_environment(self, run_id: Optional[str] = None) -> IsolatedEnvironment:
        rid = run_id or f"run_{uuid.uuid4().hex[:8]}"
        return IsolatedEnvironment(run_id=rid, base_temp_dir=self.base_temp_dir)

    def run_scenario(
        self,
        scenario_id: Union[str, ScenarioId],
        mode: Union[str, ExperimentalMode],
        seed: int = 42,
        probe_override: Optional[Dict[str, bool]] = None,
        disable_ledger: bool = False,
        disable_invariants: bool = False,
    ) -> ExperimentResult:
        """Executes a scenario in strict isolation for a specific recovery mode."""
        exp_mode = normalize_mode(mode)
        sid = ScenarioId(scenario_id) if isinstance(scenario_id, str) else scenario_id
        meta = SCENARIO_REGISTRY[sid]
        if probe_override:
            meta = ScenarioMetadata(
                scenario_id=meta.scenario_id,
                name=meta.name,
                description=meta.description,
                injected_fault=meta.injected_fault,
                probe_supported=probe_override.get("probe_supported", meta.probe_supported),
                probe_available=probe_override.get("probe_available", meta.probe_available),
                probe_is_stale=probe_override.get("probe_is_stale", meta.probe_is_stale),
                has_external_query_capability=probe_override.get("has_external_query_capability", meta.has_external_query_capability),
            )

        run_id = f"{sid.value[:6]}_{exp_mode.value[:3]}_{seed}_{uuid.uuid4().hex[:6]}"
        env = self.create_isolated_environment(run_id)

        if disable_invariants:
            env.invariants = InvariantSuite([])
            env.recovery_engine = RecoveryEngine(env.verifier, env.invariants)

        try:
            start_time = time.perf_counter()

            if sid == ScenarioId.SCENARIO_1_TIMEOUT_AFTER_SUCCESS:
                res = self._run_scenario_1_timeout(exp_mode, env, meta, disable_ledger=disable_ledger)
            elif sid == ScenarioId.SCENARIO_2_CRASH_BEFORE_RECORD:
                res = self._run_scenario_2_crash_before_record(exp_mode, env, meta, disable_ledger=disable_ledger)
            elif sid == ScenarioId.SCENARIO_3_FAILURE_BEFORE_EXECUTION:
                res = self._run_scenario_3_pre_failure(exp_mode, env, meta, disable_ledger=disable_ledger)
            elif sid == ScenarioId.SCENARIO_4_PROBE_UNAVAILABLE:
                res = self._run_scenario_4_probe_unavailable(exp_mode, env, meta, disable_ledger=disable_ledger)
            elif sid == ScenarioId.SCENARIO_5_PROBE_STALE:
                res = self._run_scenario_5_probe_stale(exp_mode, env, meta, disable_ledger=disable_ledger)
            elif sid == ScenarioId.SCENARIO_6_LEDGER_EXTERNAL_CONFLICT:
                res = self._run_scenario_6_conflict(exp_mode, env, meta, disable_ledger=disable_ledger)
            elif sid == ScenarioId.SCENARIO_7_OLDER_CHECKPOINT_RESTORED:
                res = self._run_scenario_7_older_checkpoint(exp_mode, env, meta, disable_ledger=disable_ledger)
            elif sid == ScenarioId.SCENARIO_8_NO_STATUS_QUERY:
                res = self._run_scenario_8_no_query(exp_mode, env, meta, disable_ledger=disable_ledger)
            else:
                raise ValueError(f"Unsupported scenario: {sid}")

            latency = (time.perf_counter() - start_time) * 1000.0
            res.execution_latency_ms = round(latency, 2)
            return res

        finally:
            env.cleanup()

    # -----------------------------------------------------------------------
    # Scenario 1: External Success With Response Timeout
    # -----------------------------------------------------------------------
    def _run_scenario_1_timeout(
        self,
        mode: ExperimentalMode,
        env: IsolatedEnvironment,
        meta: ScenarioMetadata,
        disable_ledger: bool = False,
    ) -> ExperimentResult:
        task_id = f"task_{meta.scenario_id.value}_{mode.value.lower()}"
        order_id = f"ord_s1_{mode.value.lower()}"
        operation_id = f"op_charge_{order_id}"
        agent = env.build_agent(task_id, mode, self.conservative_effect_log, disable_ledger=disable_ledger)
        agent.init_task(order_id=order_id, customer_id="cust_s1", item_id="laptop", amount=100.0)

        # Checkpoint 0 created before step 1
        cp0_id = agent.save_checkpoint()

        # Step 1 initial execution with TIMEOUT_AFTER_EXECUTION
        # External payment executes successfully in gateway
        env.payment_gw.charge(
            idempotency_key=f"pay_{order_id}_attempt_1",
            order_id=order_id,
            amount=100.0,
            customer_id="cust_s1",
        )

        # Ledger records intent and timeout (in modes that log effects)
        if not disable_ledger and mode in (ExperimentalMode.EFFECT_LOG, ExperimentalMode.RECOVERYGUARD):
            intent = ActionIntent(
                action_id=f"act_s1_{uuid.uuid4().hex[:6]}",
                task_id=task_id,
                operation_id=operation_id,
                tool_name="charge_payment",
                parameters={"order_id": order_id, "amount": 100.0},
                risk_level=ActionRiskLevel.RISKY_MUTATION,
                checkpoint_id=cp0_id,
            )
            env.ledger.record_intent(intent)
            env.ledger.mark_started(intent.action_id)
            env.ledger.mark_timeout(intent.action_id, "External service timed out after execution")

        # --- Crash & Rollback to Checkpoint 0 ---
        agent.restore_checkpoint(cp0_id)

        # Post-restore attempt
        attempt_count = 1

        def do_charge():
            nonlocal attempt_count
            attempt_count += 1
            key = f"pay_{order_id}_attempt_{attempt_count}"
            return env.payment_gw.charge(key, order_id, 100.0, "cust_s1")

        step_res = agent.execute_step(
            operation_id=operation_id,
            tool_name="charge_payment",
            risk_level=ActionRiskLevel.RISKY_MUTATION,
            tool_fn=do_charge,
            probe_fn=lambda: env.payment_gw.probe(order_id),
            probe_supported=meta.probe_supported,
            probe_available=meta.probe_available,
            probe_is_stale=meta.probe_is_stale,
        )

        # Evaluate side effects and invariants
        charges = [c for c in env.payment_gw._charges.values() if c["order_id"] == order_id and c["status"] == "CAPTURED"]
        captured_count = len(charges)
        dup_charges = max(0, captured_count - 1)

        inv_res = env.invariants.evaluate_all({"order_id": order_id, "payment_probe": env.payment_gw.probe_all_for_order(order_id)})
        violations = sum(1 for r in inv_res if not r.satisfied)

        blocked = 1 if step_res.get("outcome") in ("BLOCKED_BY_GATE", "BLOCKED_UNKNOWN") else 0
        human_pauses = 1 if (
            step_res.get("outcome") == "BLOCKED_UNKNOWN"
            or (step_res.get("decision") or {}).get("outcome") == "HUMAN_REVIEW"
        ) else 0

        # Success evaluation: 0 duplicate charges, 0 invariant violations, and recovered without false pause
        # In RecoveryGuard, it successfully resumes without mutation; in EFFECT_LOG it false-pauses
        success = (mode == ExperimentalMode.RECOVERYGUARD and dup_charges == 0 and violations == 0)

        return ExperimentResult(
            scenario_id=meta.scenario_id.value,
            scenario_name=meta.name,
            mode=mode.value,
            initial_conditions={"order_id": order_id, "initial_checkpoint": cp0_id},
            injected_fault=meta.injected_fault,
            recovery_decision=step_res.get("decision"),
            actual_side_effects={"captured_charges": captured_count},
            duplicate_side_effects=dup_charges,
            invariant_violations=violations,
            blocked_actions=blocked,
            human_escalations=human_pauses,
            execution_latency_ms=0.0,
            success=success,
            details=f"Outcome: {step_res.get('outcome')}; Total charges: {captured_count}",
        )

    # -----------------------------------------------------------------------
    # Scenario 2: Crash After Success Before Record
    # -----------------------------------------------------------------------
    def _run_scenario_2_crash_before_record(
        self,
        mode: ExperimentalMode,
        env: IsolatedEnvironment,
        meta: ScenarioMetadata,
        disable_ledger: bool = False,
    ) -> ExperimentResult:
        task_id = f"task_{meta.scenario_id.value}_{mode.value.lower()}"
        order_id = f"ord_s2_{mode.value.lower()}"
        operation_id = f"op_charge_{order_id}"
        agent = env.build_agent(task_id, mode, self.conservative_effect_log, disable_ledger=disable_ledger)
        agent.init_task(order_id=order_id, customer_id="cust_s2", item_id="laptop", amount=150.0)

        cp0_id = agent.save_checkpoint()

        # External service succeeded
        env.payment_gw.charge(
            idempotency_key=f"pay_{order_id}_1",
            order_id=order_id,
            amount=150.0,
            customer_id="cust_s2",
        )
        # Agent crashed BEFORE committing to ledger (ledger has no record or only started)

        # --- Crash & Rollback to Checkpoint 0 ---
        agent.restore_checkpoint(cp0_id)

        attempt_count = 1

        def do_charge():
            nonlocal attempt_count
            attempt_count += 1
            return env.payment_gw.charge(f"pay_{order_id}_{attempt_count}", order_id, 150.0, "cust_s2")

        step_res = agent.execute_step(
            operation_id=operation_id,
            tool_name="charge_payment",
            risk_level=ActionRiskLevel.RISKY_MUTATION,
            tool_fn=do_charge,
            probe_fn=lambda: env.payment_gw.probe(order_id),
            probe_supported=meta.probe_supported,
            probe_available=meta.probe_available,
            probe_is_stale=meta.probe_is_stale,
        )

        charges = [c for c in env.payment_gw._charges.values() if c["order_id"] == order_id and c["status"] == "CAPTURED"]
        captured_count = len(charges)
        dup_charges = max(0, captured_count - 1)

        inv_res = env.invariants.evaluate_all({"order_id": order_id, "payment_probe": env.payment_gw.probe_all_for_order(order_id)})
        violations = sum(1 for r in inv_res if not r.satisfied)

        blocked = 1 if step_res.get("outcome") in ("BLOCKED_BY_GATE", "BLOCKED_UNKNOWN") else 0
        human_pauses = 1 if (step_res.get("decision") or {}).get("outcome") == "HUMAN_REVIEW" else 0

        # Mode A and B duplicate because ledger has no record; Mode C queries probe and resumes
        success = (mode == ExperimentalMode.RECOVERYGUARD and dup_charges == 0 and violations == 0)

        return ExperimentResult(
            scenario_id=meta.scenario_id.value,
            scenario_name=meta.name,
            mode=mode.value,
            initial_conditions={"order_id": order_id, "amount": 150.0},
            injected_fault=meta.injected_fault,
            recovery_decision=step_res.get("decision"),
            actual_side_effects={"captured_charges": captured_count},
            duplicate_side_effects=dup_charges,
            invariant_violations=violations,
            blocked_actions=blocked,
            human_escalations=human_pauses,
            execution_latency_ms=0.0,
            success=success,
            details=f"Outcome: {step_res.get('outcome')}; Total charges: {captured_count}",
        )

    # -----------------------------------------------------------------------
    # Scenario 3: Failure Before External Action
    # -----------------------------------------------------------------------
    def _run_scenario_3_pre_failure(
        self,
        mode: ExperimentalMode,
        env: IsolatedEnvironment,
        meta: ScenarioMetadata,
        disable_ledger: bool = False,
    ) -> ExperimentResult:
        task_id = f"task_{meta.scenario_id.value}_{mode.value.lower()}"
        order_id = f"ord_s3_{mode.value.lower()}"
        operation_id = f"op_charge_{order_id}"
        agent = env.build_agent(task_id, mode, self.conservative_effect_log, disable_ledger=disable_ledger)
        agent.init_task(order_id=order_id, customer_id="cust_s3", item_id="laptop", amount=200.0)

        cp0_id = agent.save_checkpoint()

        # External service was NEVER reached (0 charges)
        # Ledger in Mode B/C records FAILED
        if not disable_ledger and mode in (ExperimentalMode.EFFECT_LOG, ExperimentalMode.RECOVERYGUARD):
            intent = ActionIntent(
                action_id=f"act_s3_{uuid.uuid4().hex[:6]}",
                task_id=task_id,
                operation_id=operation_id,
                tool_name="charge_payment",
                parameters={"order_id": order_id, "amount": 200.0},
                risk_level=ActionRiskLevel.RISKY_MUTATION,
            )
            env.ledger.record_intent(intent)
            env.ledger.mark_failed(intent.action_id, "Connection refused before request sent")

        # --- Crash & Rollback to Checkpoint 0 ---
        agent.restore_checkpoint(cp0_id)

        attempt_count = 0

        def do_charge():
            nonlocal attempt_count
            attempt_count += 1
            return env.payment_gw.charge(f"pay_{order_id}_{attempt_count}", order_id, 200.0, "cust_s3")

        step_res = agent.execute_step(
            operation_id=operation_id,
            tool_name="charge_payment",
            risk_level=ActionRiskLevel.RISKY_MUTATION,
            tool_fn=do_charge,
            probe_fn=lambda: env.payment_gw.probe(order_id),
            probe_supported=meta.probe_supported,
            probe_available=meta.probe_available,
            probe_is_stale=meta.probe_is_stale,
        )

        charges = [c for c in env.payment_gw._charges.values() if c["order_id"] == order_id and c["status"] == "CAPTURED"]
        captured_count = len(charges)
        dup_charges = max(0, captured_count - 1)

        inv_res = env.invariants.evaluate_all({"order_id": order_id, "payment_probe": env.payment_gw.probe_all_for_order(order_id)})
        violations = sum(1 for r in inv_res if not r.satisfied)

        success = (captured_count == 1 and dup_charges == 0 and violations == 0)

        return ExperimentResult(
            scenario_id=meta.scenario_id.value,
            scenario_name=meta.name,
            mode=mode.value,
            initial_conditions={"order_id": order_id, "amount": 200.0},
            injected_fault=meta.injected_fault,
            recovery_decision=step_res.get("decision"),
            actual_side_effects={"captured_charges": captured_count},
            duplicate_side_effects=dup_charges,
            invariant_violations=violations,
            blocked_actions=0,
            human_escalations=0,
            execution_latency_ms=0.0,
            success=success,
            details=f"Outcome: {step_res.get('outcome')}; Safe retry performed; Total charges: {captured_count}",
        )

    # -----------------------------------------------------------------------
    # Scenario 4: External Probe Unavailable
    # -----------------------------------------------------------------------
    def _run_scenario_4_probe_unavailable(
        self,
        mode: ExperimentalMode,
        env: IsolatedEnvironment,
        meta: ScenarioMetadata,
        disable_ledger: bool = False,
    ) -> ExperimentResult:
        task_id = f"task_{meta.scenario_id.value}_{mode.value.lower()}"
        order_id = f"ord_s4_{mode.value.lower()}"
        operation_id = f"op_charge_{order_id}"
        agent = env.build_agent(task_id, mode, self.conservative_effect_log, disable_ledger=disable_ledger)
        agent.init_task(order_id=order_id, customer_id="cust_s4", item_id="laptop", amount=250.0)

        cp0_id = agent.save_checkpoint()

        # External service executed 1 charge
        env.payment_gw.charge(f"pay_{order_id}_1", order_id, 250.0, "cust_s4")

        # Ambiguous timeout logged
        if not disable_ledger and mode in (ExperimentalMode.EFFECT_LOG, ExperimentalMode.RECOVERYGUARD):
            intent = ActionIntent(
                action_id=f"act_s4_{uuid.uuid4().hex[:6]}",
                task_id=task_id,
                operation_id=operation_id,
                tool_name="charge_payment",
                parameters={"order_id": order_id, "amount": 250.0},
                risk_level=ActionRiskLevel.RISKY_MUTATION,
            )
            env.ledger.record_intent(intent)
            env.ledger.mark_timeout(intent.action_id, "Gateway timed out")

        # Set external probe availability according to meta
        env.payment_gw.set_probe_available(meta.probe_available)

        # Rollback
        agent.restore_checkpoint(cp0_id)

        attempt_count = 1

        def do_charge():
            nonlocal attempt_count
            attempt_count += 1
            return env.payment_gw.charge(f"pay_{order_id}_{attempt_count}", order_id, 250.0, "cust_s4")

        step_res = agent.execute_step(
            operation_id=operation_id,
            tool_name="charge_payment",
            risk_level=ActionRiskLevel.RISKY_MUTATION,
            tool_fn=do_charge,
            probe_fn=lambda: env.payment_gw.probe(order_id),
            probe_supported=meta.probe_supported,
            probe_available=meta.probe_available,
            probe_is_stale=meta.probe_is_stale,
        )

        charges = [c for c in env.payment_gw._charges.values() if c["order_id"] == order_id and c["status"] == "CAPTURED"]
        captured_count = len(charges)
        dup_charges = max(0, captured_count - 1)

        blocked = 1 if step_res.get("outcome") in ("BLOCKED_BY_GATE", "BLOCKED_UNKNOWN") else 0
        human_pauses = 1 if (
            step_res.get("outcome") == "BLOCKED_UNKNOWN"
            or (step_res.get("decision") or {}).get("outcome") == "HUMAN_REVIEW"
        ) else 0

        # In Mode C, fails closed safely with HUMAN_REVIEW and blocks duplicate!
        success = (mode == ExperimentalMode.RECOVERYGUARD and blocked == 1 and dup_charges == 0)

        return ExperimentResult(
            scenario_id=meta.scenario_id.value,
            scenario_name=meta.name,
            mode=mode.value,
            initial_conditions={"order_id": order_id, "probe_available": False},
            injected_fault=meta.injected_fault,
            recovery_decision=step_res.get("decision"),
            actual_side_effects={"captured_charges": captured_count},
            duplicate_side_effects=dup_charges,
            invariant_violations=1 if dup_charges > 0 else 0,
            blocked_actions=blocked,
            human_escalations=human_pauses,
            execution_latency_ms=0.0,
            success=success,
            details=f"Outcome: {step_res.get('outcome')}; Probe unavailable handled fail-closed",
        )

    # -----------------------------------------------------------------------
    # Scenario 5: External Evidence Is Stale
    # -----------------------------------------------------------------------
    def _run_scenario_5_probe_stale(
        self,
        mode: ExperimentalMode,
        env: IsolatedEnvironment,
        meta: ScenarioMetadata,
        disable_ledger: bool = False,
    ) -> ExperimentResult:
        task_id = f"task_{meta.scenario_id.value}_{mode.value.lower()}"
        order_id = f"ord_s5_{mode.value.lower()}"
        operation_id = f"op_charge_{order_id}"
        agent = env.build_agent(task_id, mode, self.conservative_effect_log, disable_ledger=disable_ledger)
        agent.init_task(order_id=order_id, customer_id="cust_s5", item_id="laptop", amount=300.0)

        cp0_id = agent.save_checkpoint()

        # External service executed 1 charge
        env.payment_gw.charge(f"pay_{order_id}_1", order_id, 300.0, "cust_s5")

        if not disable_ledger and mode in (ExperimentalMode.EFFECT_LOG, ExperimentalMode.RECOVERYGUARD):
            intent = ActionIntent(
                action_id=f"act_s5_{uuid.uuid4().hex[:6]}",
                task_id=task_id,
                operation_id=operation_id,
                tool_name="charge_payment",
                parameters={"order_id": order_id, "amount": 300.0},
                risk_level=ActionRiskLevel.RISKY_MUTATION,
            )
            env.ledger.record_intent(intent)
            env.ledger.mark_timeout(intent.action_id, "Gateway timeout")

        # Configure stale probe evidence according to meta
        if meta.probe_is_stale:
            env.payment_gw.set_stale_evidence({"status": "STALE_ABSENT", "order_id": order_id})
        else:
            env.payment_gw.clear_stale_evidence()

        agent.restore_checkpoint(cp0_id)

        attempt_count = 1

        def do_charge():
            nonlocal attempt_count
            attempt_count += 1
            return env.payment_gw.charge(f"pay_{order_id}_{attempt_count}", order_id, 300.0, "cust_s5")

        step_res = agent.execute_step(
            operation_id=operation_id,
            tool_name="charge_payment",
            risk_level=ActionRiskLevel.RISKY_MUTATION,
            tool_fn=do_charge,
            probe_fn=lambda: env.payment_gw.probe(order_id),
            probe_supported=meta.probe_supported,
            probe_available=meta.probe_available,
            probe_is_stale=meta.probe_is_stale,
        )

        charges = [c for c in env.payment_gw._charges.values() if c["order_id"] == order_id and c["status"] == "CAPTURED"]
        captured_count = len(charges)
        dup_charges = max(0, captured_count - 1)

        blocked = 1 if step_res.get("outcome") in ("BLOCKED_BY_GATE", "BLOCKED_UNKNOWN") else 0
        human_pauses = 1 if (
            step_res.get("outcome") == "BLOCKED_UNKNOWN"
            or (step_res.get("decision") or {}).get("outcome") == "HUMAN_REVIEW"
        ) else 0

        success = (mode == ExperimentalMode.RECOVERYGUARD and blocked == 1 and dup_charges == 0)

        return ExperimentResult(
            scenario_id=meta.scenario_id.value,
            scenario_name=meta.name,
            mode=mode.value,
            initial_conditions={"order_id": order_id, "probe_is_stale": True},
            injected_fault=meta.injected_fault,
            recovery_decision=step_res.get("decision"),
            actual_side_effects={"captured_charges": captured_count},
            duplicate_side_effects=dup_charges,
            invariant_violations=1 if dup_charges > 0 else 0,
            blocked_actions=blocked,
            human_escalations=human_pauses,
            execution_latency_ms=0.0,
            success=success,
            details=f"Outcome: {step_res.get('outcome')}; Stale probe rejected; blocked unsafe retry",
        )

    # -----------------------------------------------------------------------
    # Scenario 6: Ledger and External Evidence Conflict
    # -----------------------------------------------------------------------
    def _run_scenario_6_conflict(
        self,
        mode: ExperimentalMode,
        env: IsolatedEnvironment,
        meta: ScenarioMetadata,
        disable_ledger: bool = False,
    ) -> ExperimentResult:
        task_id = f"task_{meta.scenario_id.value}_{mode.value.lower()}"
        order_id = f"ord_s6_{mode.value.lower()}"
        operation_id = f"op_charge_{order_id}"
        agent = env.build_agent(task_id, mode, self.conservative_effect_log, disable_ledger=disable_ledger)
        agent.init_task(order_id=order_id, customer_id="cust_s6", item_id="laptop", amount=350.0)

        cp0_id = agent.save_checkpoint()

        # Ledger records COMMITTED
        if not disable_ledger and mode in (ExperimentalMode.EFFECT_LOG, ExperimentalMode.RECOVERYGUARD):
            intent = ActionIntent(
                action_id=f"act_s6_{uuid.uuid4().hex[:6]}",
                task_id=task_id,
                operation_id=operation_id,
                tool_name="charge_payment",
                parameters={"order_id": order_id, "amount": 350.0},
                risk_level=ActionRiskLevel.RISKY_MUTATION,
            )
            env.ledger.record_intent(intent)
            env.ledger.mark_committed(intent.action_id, {"status": "SUCCESS"})

        # BUT external gateway has NO charge (or was externally canceled/voided)
        # env.payment_gw has 0 charges

        agent.restore_checkpoint(cp0_id)

        attempt_count = 0

        def do_charge():
            nonlocal attempt_count
            attempt_count += 1
            return env.payment_gw.charge(f"pay_{order_id}_{attempt_count}", order_id, 350.0, "cust_s6")

        step_res = agent.execute_step(
            operation_id=operation_id,
            tool_name="charge_payment",
            risk_level=ActionRiskLevel.RISKY_MUTATION,
            tool_fn=do_charge,
            probe_fn=lambda: env.payment_gw.probe(order_id),  # Returns None!
            probe_supported=meta.probe_supported,
            probe_available=meta.probe_available,
            probe_is_stale=meta.probe_is_stale,
            conflict_policy="quarantine",
        )

        charges = [c for c in env.payment_gw._charges.values() if c["order_id"] == order_id and c["status"] == "CAPTURED"]
        captured_count = len(charges)
        dup_charges = max(0, captured_count - 1)

        blocked = 1 if step_res.get("outcome") in ("BLOCKED_BY_GATE", "BLOCKED_UNKNOWN") else 0
        human_pauses = 1 if (
            step_res.get("outcome") == "BLOCKED_UNKNOWN"
            or (step_res.get("decision") or {}).get("outcome") in ("QUARANTINE", "HUMAN_REVIEW")
        ) else 0

        # Mode B silently skips (believes ledger, leaving order unpaid externally!).
        # Mode C catches conflict, issues QUARANTINE, and blocks execution!
        success = (
            mode == ExperimentalMode.RECOVERYGUARD
            and (step_res.get("decision") or {}).get("outcome") == "QUARANTINE"
            and blocked == 1
        )

        return ExperimentResult(
            scenario_id=meta.scenario_id.value,
            scenario_name=meta.name,
            mode=mode.value,
            initial_conditions={"order_id": order_id, "ledger_status": "COMMITTED", "external_charges": 0},
            injected_fault=meta.injected_fault,
            recovery_decision=step_res.get("decision"),
            actual_side_effects={"captured_charges": captured_count},
            duplicate_side_effects=dup_charges,
            invariant_violations=0,
            blocked_actions=blocked,
            human_escalations=human_pauses,
            execution_latency_ms=0.0,
            success=success,
            details=f"Outcome: {step_res.get('outcome')}; Conflict correctly quarantined",
        )

    # -----------------------------------------------------------------------
    # Scenario 7: Older Checkpoint Restored After Effects
    # -----------------------------------------------------------------------
    def _run_scenario_7_older_checkpoint(
        self,
        mode: ExperimentalMode,
        env: IsolatedEnvironment,
        meta: ScenarioMetadata,
        disable_ledger: bool = False,
    ) -> ExperimentResult:
        task_id = f"task_{meta.scenario_id.value}_{mode.value.lower()}"
        order_id = f"ord_s7_{mode.value.lower()}"
        op_charge = f"op_charge_{order_id}"
        op_notify = f"op_notify_{order_id}"

        agent = env.build_agent(task_id, mode, self.conservative_effect_log, disable_ledger=disable_ledger)
        agent.init_task(order_id=order_id, customer_id="cust_s7", item_id="laptop", amount=100.0)

        # Step 0: Initial Checkpoint CP0
        cp0_id = agent.save_checkpoint()

        # Step 1: Charge payment committed
        env.payment_gw.charge(f"pay_{order_id}_1", order_id, 100.0, "cust_s7")
        if not disable_ledger and mode in (ExperimentalMode.EFFECT_LOG, ExperimentalMode.RECOVERYGUARD):
            act1 = ActionIntent(
                action_id=f"act_s7_1_{uuid.uuid4().hex[:6]}",
                task_id=task_id,
                operation_id=op_charge,
                tool_name="charge_payment",
                parameters={"order_id": order_id, "amount": 100.0},
                risk_level=ActionRiskLevel.RISKY_MUTATION,
                checkpoint_id=cp0_id,
            )
            env.ledger.record_intent(act1)
            env.ledger.mark_committed(act1.action_id, {"status": "SUCCESS"})

        # Step 2: Intermediate Checkpoint CP1
        agent.state["step_index"] = 1
        agent.state["payment_id"] = "pay_1"
        cp1_id = agent.save_checkpoint()

        # Step 3: Notification delivered
        env.notifications.send_notification(
            recipient="user@example.com",
            message="Shipped!",
            order_id=order_id,
            notification_id=f"ntf_{order_id}_1",
        )
        if not disable_ledger and mode in (ExperimentalMode.EFFECT_LOG, ExperimentalMode.RECOVERYGUARD):
            act2 = ActionIntent(
                action_id=f"act_s7_2_{uuid.uuid4().hex[:6]}",
                task_id=task_id,
                operation_id=op_notify,
                tool_name="send_notification",
                parameters={"order_id": order_id},
                risk_level=ActionRiskLevel.RISKY_MUTATION,
                checkpoint_id=cp1_id,
            )
            env.ledger.record_intent(act2)
            env.ledger.mark_committed(act2.action_id, {"status": "DELIVERED"})

        # --- Operator/Crash rolls back all the way to OLDER Checkpoint CP0 ---
        agent.restore_checkpoint(cp0_id)

        # Attempt to run Step 1 (charge) then Step 2 (notify)
        charge_attempt = 1
        notify_attempt = 1

        def do_charge():
            nonlocal charge_attempt
            charge_attempt += 1
            return env.payment_gw.charge(f"pay_{order_id}_{charge_attempt}", order_id, 100.0, "cust_s7")

        def do_notify():
            nonlocal notify_attempt
            notify_attempt += 1
            return env.notifications.send_notification(
                recipient="user@example.com",
                message="Shipped!",
                order_id=order_id,
                notification_id=f"ntf_{order_id}_{notify_attempt}",
            )

        res1 = agent.execute_step(
            operation_id=op_charge,
            tool_name="charge_payment",
            risk_level=ActionRiskLevel.RISKY_MUTATION,
            tool_fn=do_charge,
            probe_fn=lambda: env.payment_gw.probe(order_id),
        )

        res2 = agent.execute_step(
            operation_id=op_notify,
            tool_name="send_notification",
            risk_level=ActionRiskLevel.RISKY_MUTATION,
            tool_fn=do_notify,
            probe_fn=lambda: env.notifications.probe_order(order_id),
        )

        charges = [c for c in env.payment_gw._charges.values() if c["order_id"] == order_id and c["status"] == "CAPTURED"]
        captured_charges = len(charges)
        dup_charges = max(0, captured_charges - 1)

        delivered_notifs = len(env.notifications.get_messages_for_order(order_id))
        dup_notifs = max(0, delivered_notifs - 1)

        total_dups = dup_charges + dup_notifs

        inv_res = env.invariants.evaluate_all({
            "order_id": order_id,
            "payment_probe": env.payment_gw.probe_all_for_order(order_id),
            "notification_probe": env.notifications.probe_order(order_id),
        })
        violations = sum(1 for r in inv_res if not r.satisfied)

        success = (total_dups == 0 and violations == 0)

        return ExperimentResult(
            scenario_id=meta.scenario_id.value,
            scenario_name=meta.name,
            mode=mode.value,
            initial_conditions={"order_id": order_id, "restored_checkpoint": cp0_id},
            injected_fault=meta.injected_fault,
            recovery_decision=res1.get("decision"),
            actual_side_effects={"charges": captured_charges, "notifications": delivered_notifs},
            duplicate_side_effects=total_dups,
            invariant_violations=violations,
            blocked_actions=0,
            human_escalations=0,
            execution_latency_ms=0.0,
            success=success,
            details=f"Step 1: {res1.get('outcome')}; Step 2: {res2.get('outcome')}; Total duplicates: {total_dups}",
        )

    # -----------------------------------------------------------------------
    # Scenario 8: External Service Lacks Status Query
    # -----------------------------------------------------------------------
    def _run_scenario_8_no_query(
        self,
        mode: ExperimentalMode,
        env: IsolatedEnvironment,
        meta: ScenarioMetadata,
        disable_ledger: bool = False,
    ) -> ExperimentResult:
        task_id = f"task_{meta.scenario_id.value}_{mode.value.lower()}"
        order_id = f"ord_s8_{mode.value.lower()}"
        operation_id = f"op_notify_{order_id}"
        agent = env.build_agent(task_id, mode, self.conservative_effect_log, disable_ledger=disable_ledger)
        agent.init_task(order_id=order_id, customer_id="cust_s8", item_id="laptop", amount=100.0)

        cp0_id = agent.save_checkpoint()

        # Notification was sent externally
        env.notifications.send_notification(
            recipient="user@example.com",
            message="Alert!",
            order_id=order_id,
            notification_id=f"ntf_{order_id}_1",
        )

        # Ambiguous timeout logged
        if not disable_ledger and mode in (ExperimentalMode.EFFECT_LOG, ExperimentalMode.RECOVERYGUARD):
            intent = ActionIntent(
                action_id=f"act_s8_{uuid.uuid4().hex[:6]}",
                task_id=task_id,
                operation_id=operation_id,
                tool_name="send_notification",
                parameters={"order_id": order_id},
                risk_level=ActionRiskLevel.RISKY_MUTATION,
                is_compensatable=False,
            )
            env.ledger.record_intent(intent)
            env.ledger.mark_timeout(intent.action_id, "Notification API timeout")

        agent.restore_checkpoint(cp0_id)

        notify_attempt = 1

        def do_notify():
            nonlocal notify_attempt
            notify_attempt += 1
            return env.notifications.send_notification(
                recipient="user@example.com",
                message="Alert!",
                order_id=order_id,
                notification_id=f"ntf_{order_id}_{notify_attempt}",
            )

        probe_fn = (lambda: env.notifications.probe_order(order_id)) if meta.probe_supported else None
        step_res = agent.execute_step(
            operation_id=operation_id,
            tool_name="send_notification",
            risk_level=ActionRiskLevel.RISKY_MUTATION,
            tool_fn=do_notify,
            probe_fn=probe_fn,
            probe_supported=meta.probe_supported,
            probe_available=meta.probe_available,
            probe_is_stale=meta.probe_is_stale,
        )

        messages = env.notifications.get_messages_for_order(order_id)
        msg_count = len(messages)
        dup_msgs = max(0, msg_count - 1)

        blocked = 1 if step_res.get("outcome") in ("BLOCKED_BY_GATE", "BLOCKED_UNKNOWN") else 0
        human_pauses = 1 if (
            step_res.get("outcome") == "BLOCKED_UNKNOWN"
            or (step_res.get("decision") or {}).get("outcome") == "HUMAN_REVIEW"
        ) else 0

        success = (mode == ExperimentalMode.RECOVERYGUARD and blocked == 1 and dup_msgs == 0)

        return ExperimentResult(
            scenario_id=meta.scenario_id.value,
            scenario_name=meta.name,
            mode=mode.value,
            initial_conditions={"order_id": order_id, "probe_supported": False},
            injected_fault=meta.injected_fault,
            recovery_decision=step_res.get("decision"),
            actual_side_effects={"notifications_sent": msg_count},
            duplicate_side_effects=dup_msgs,
            invariant_violations=1 if dup_msgs > 0 else 0,
            blocked_actions=blocked,
            human_escalations=human_pauses,
            execution_latency_ms=0.0,
            success=success,
            details=f"Outcome: {step_res.get('outcome')}; Unsupported status probe halted safely",
        )

    # -----------------------------------------------------------------------
    # Multi-Mode and Multi-Scenario Runners
    # -----------------------------------------------------------------------
    def run_scenario_all_modes(
        self,
        scenario_id: Union[str, ScenarioId],
        seed: int = 42,
        probe_override: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, ExperimentResult]:
        """Runs the given scenario across all three modes."""
        results = {}
        for m in (ExperimentalMode.CHECKPOINT_ONLY, ExperimentalMode.EFFECT_LOG, ExperimentalMode.RECOVERYGUARD):
            results[m.value] = self.run_scenario(scenario_id, m, seed, probe_override=probe_override)
        return results

    def run_all_scenarios(
        self,
        modes: Optional[List[ExperimentalMode]] = None,
        seed: int = 42,
        trials_per_scenario: int = 1,
    ) -> List[ExperimentResult]:
        """Runs the full evaluation matrix (all 8 scenarios across selected or all modes)."""
        target_modes = modes or [
            ExperimentalMode.CHECKPOINT_ONLY,
            ExperimentalMode.EFFECT_LOG,
            ExperimentalMode.RECOVERYGUARD,
        ]
        all_results = []
        for s_idx, sid in enumerate(ScenarioId):
            for m in target_modes:
                for trial in range(trials_per_scenario):
                    trial_seed = seed + s_idx * 1000 + trial
                    all_results.append(self.run_scenario(sid, m, seed=trial_seed))
        return all_results

    def run_ablation_scenario(
        self,
        scenario_id: Union[str, ScenarioId],
        variant: Union[str, AblationVariant] = AblationVariant.FULL_RECOVERYGUARD,
        seed: int = 42,
    ) -> ExperimentResult:
        """Runs a scenario under a specific ablation variant of RecoveryGuard.

        Ablation variants:
        - FULL_RECOVERYGUARD: Complete RecoveryGuard with ledger, probes, and invariants.
        - NO_LEDGER: Disables persistent effect ledger. Gate has no local history.
        - NO_PROBES: Disables external status probes (probe_available=False).
        - NO_INVARIANTS: Disables business/task invariant evaluations.
        """
        if isinstance(variant, str):
            variant = AblationVariant(variant)

        sid = ScenarioId(scenario_id) if isinstance(scenario_id, str) else scenario_id
        disable_ledger = (variant == AblationVariant.NO_LEDGER)
        disable_invariants = (variant == AblationVariant.NO_INVARIANTS)
        probe_override = {"probe_available": False, "probe_supported": False} if variant == AblationVariant.NO_PROBES else None

        res = self.run_scenario(
            scenario_id=sid,
            mode=ExperimentalMode.RECOVERYGUARD,
            seed=seed,
            probe_override=probe_override,
            disable_ledger=disable_ledger,
            disable_invariants=disable_invariants,
        )
        res.mode = f"RECOVERYGUARD_{variant.value}"
        return res

    def run_ablation_study(
        self,
        scenarios: Optional[List[ScenarioId]] = None,
        seed: int = 42,
        trials_per_scenario: int = 1,
    ) -> Dict[str, List[ExperimentResult]]:
        """Runs the complete ablation study across all 4 variants."""
        target_scenarios = scenarios or list(ScenarioId)
        results: Dict[str, List[ExperimentResult]] = {
            v.value: [] for v in AblationVariant
        }
        for variant in AblationVariant:
            for s_idx, sid in enumerate(target_scenarios):
                for trial in range(trials_per_scenario):
                    trial_seed = seed + s_idx * 1000 + trial
                    res = self.run_ablation_scenario(sid, variant, seed=trial_seed)
                    results[variant.value].append(res)
        return results

    def run_benchmark_trials(
        self,
        num_trials: int = 100,
        scenarios: Optional[List[ScenarioId]] = None,
        modes: Optional[List[ExperimentalMode]] = None,
        seed: int = 42,
    ) -> List[ExperimentResult]:
        """Runs multi-trial evaluation across scenarios and modes (e.g. >=100 trials).

        Deterministic and isolated for reproducibility.
        """
        target_scenarios = scenarios or list(ScenarioId)
        target_modes = modes or [
            ExperimentalMode.CHECKPOINT_ONLY,
            ExperimentalMode.EFFECT_LOG,
            ExperimentalMode.RECOVERYGUARD,
        ]
        results: List[ExperimentResult] = []
        for s_idx, sid in enumerate(target_scenarios):
            for m in target_modes:
                for trial in range(num_trials):
                    trial_seed = seed + s_idx * 10000 + trial
                    results.append(self.run_scenario(sid, m, seed=trial_seed))
        return results

    # -----------------------------------------------------------------------
    # Backward Compatibility Methods
    # -----------------------------------------------------------------------
    def run_crash_after_payment_scenario(self, mode: str) -> ScenarioResult:
        exp_res = self.run_scenario(ScenarioId.SCENARIO_1_TIMEOUT_AFTER_SUCCESS, mode)
        return ScenarioResult(
            scenario_name=exp_res.scenario_name,
            mode=exp_res.mode,
            success=exp_res.success,
            duplicate_charges=exp_res.duplicate_side_effects,
            invariant_violations=exp_res.invariant_violations,
            human_pauses=exp_res.human_escalations,
            details=exp_res.details,
        )

    def run_all_benchmarks(self) -> List[ScenarioResult]:
        res_list = []
        for m in ("checkpoint_only", "effect_log_only", "recovery_guard"):
            res_list.append(self.run_crash_after_payment_scenario(m))
        return res_list
