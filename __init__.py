"""RecoveryGuard: Cross-System Recovery Verification for AI Agents."""

__version__ = "0.1.0"

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
    BaseInvariant,
    DeploymentNoDuplicateInvariant,
    InvariantSuite,
    NoDuplicateChargeInvariant,
    NotificationSingleDeliveryInvariant,
    PaymentBeforeReservationInvariant,
    PaymentNoDuplicateInvariant,
    WorkflowCompletionCondition,
)
from recoveryguard.ledger import DurableEffectLedger
from recoveryguard.mock_services import (
    DeploymentRecordService,
    InventoryStore,
    NotificationService,
    PaymentGateway,
    ProbeUnavailableError,
)
from recoveryguard.models import (
    ActionIntent,
    ActionRiskLevel,
    CheckpointMetadata,
    EffectEvidence,
    EffectRecord,
    EffectStatus,
    EvidenceSource,
    EvidenceState,
    GateAuthorization,
    GateDeniedError,
    InvariantResult,
    InvariantStatus,
    RecoveryDecision,
    RecoveryOutcome,
    RecoveryVerificationError,
)
from recoveryguard.recovery_engine import RecoveryEngine
from recoveryguard.verifier import StateVerifier
from recoveryguard.experiments import (
    ExperimentHarness,
    ExperimentResult,
    ExperimentalMode,
    SCENARIO_REGISTRY,
    ScenarioId,
    ScenarioMetadata,
    ScenarioResult,
    normalize_mode,
)
from recoveryguard.evaluation import (
    METRIC_DEFINITIONS,
    MetricDefinition,
    ModeEvaluationSummary,
    QuantitativeEvaluator,
    classify_experiment_result,
)

__all__ = [
    "AgentWorkflow",
    "CheckpointManager",
    "DatabaseManager",
    "DurableEffectLedger",
    "NotificationService",
    "DeploymentRecordService",
    "PaymentGateway",
    "InventoryStore",
    "ProbeUnavailableError",
    "FaultInjector",
    "FaultType",
    "ServiceTimeoutError",
    "PreExecutionFailureError",
    "AgentCrashError",
    "ActionIntent",
    "ActionRiskLevel",
    "CheckpointMetadata",
    "EffectEvidence",
    "EffectRecord",
    "EffectStatus",
    "EvidenceSource",
    "EvidenceState",
    "InvariantResult",
    "InvariantStatus",
    "RecoveryDecision",
    "RecoveryOutcome",
    "RecoveryEngine",
    "StateVerifier",
    "BaseInvariant",
    "NotificationSingleDeliveryInvariant",
    "DeploymentNoDuplicateInvariant",
    "PaymentNoDuplicateInvariant",
    "NoDuplicateChargeInvariant",
    "PaymentBeforeReservationInvariant",
    "WorkflowCompletionCondition",
    "InvariantSuite",
    "GateAuthorization",
    "GateDeniedError",
    "RecoveryVerificationError",
    "ExperimentalMode",
    "ExperimentHarness",
    "ExperimentResult",
    "ScenarioResult",
    "ScenarioId",
    "ScenarioMetadata",
    "SCENARIO_REGISTRY",
    "normalize_mode",
    "QuantitativeEvaluator",
    "ModeEvaluationSummary",
    "MetricDefinition",
    "METRIC_DEFINITIONS",
    "classify_experiment_result",
]
