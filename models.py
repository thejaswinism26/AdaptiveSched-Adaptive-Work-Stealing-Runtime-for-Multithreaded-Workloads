"""Core data models and domain enums for RecoveryGuard.

Provides structured, immutable dataclasses for action intents, effect evidence,
checkpoint metadata, invariant evaluations, and recovery decisions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EvidenceState(str, Enum):
    """Epistemic evidence classification for external side effects."""
    CONFIRMED_SUCCESS = "CONFIRMED_SUCCESS"
    CONFIRMED_ABSENT = "CONFIRMED_ABSENT"
    UNKNOWN = "UNKNOWN"
    CONFLICTING = "CONFLICTING"


class EvidenceSource(str, Enum):
    """Origin of verification evidence."""
    LOCAL_LEDGER = "LOCAL_LEDGER"
    EXTERNAL_PROBE = "EXTERNAL_PROBE"
    INVARIANT_CHECK = "INVARIANT_CHECK"
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"


class RecoveryOutcome(str, Enum):
    """Deterministic recovery gate outcomes."""
    RESUME = "RESUME"                          # Prior effect committed; skip duplicate execution and resume
    RECONCILE = "RECONCILE"                    # State divergence detected; reconcile internal agent memory
    SAFE_RETRY = "SAFE_RETRY"                  # Backward-compatible alias for RETRY
    COMPENSATE = "COMPENSATE"                  # Invariant violated; trigger explicit rollback/compensation action
    QUARANTINE = "QUARANTINE"                  # Conflicting unresolvable state; isolate task without further mutation
    HUMAN_REVIEW = "HUMAN_REVIEW"              # Backward-compatible alias for HUMAN_INTERVENTION

    # Canonical aliases mapping to the 4 core recovery outcomes
    RETRY = "SAFE_RETRY"
    HUMAN_INTERVENTION = "HUMAN_REVIEW"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            val_upper = value.upper()
            if val_upper in ("RETRY", "SAFE_RETRY"):
                return cls.SAFE_RETRY
            if val_upper in ("HUMAN_INTERVENTION", "HUMAN_REVIEW"):
                return cls.HUMAN_REVIEW
        return super()._missing_(value)


class ActionRiskLevel(str, Enum):
    """Inherent risk classification of agent tools."""
    SAFE_READ = "SAFE_READ"                  # Read-only query, no side effects
    IDEMPOTENT_WRITE = "IDEMPOTENT_WRITE"    # Mutates state, but re-execution with same key is a no-op
    RISKY_MUTATION = "RISKY_MUTATION"        # Non-idempotent side effect (e.g., credit card charge, order dispatch)


class EffectStatus(str, Enum):
    """Lifecycle and epistemic statuses for ledger-recorded actions."""
    PENDING = "PENDING"
    STARTED = "STARTED"
    ATTEMPTED = "ATTEMPTED"
    COMMITTED = "COMMITTED"
    CONFIRMED_SUCCESS = "CONFIRMED_SUCCESS"
    CONFIRMED_ABSENT = "CONFIRMED_ABSENT"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"
    CONFLICTING = "CONFLICTING"


@dataclass(frozen=True)
class ActionIntent:
    """Represents an agent's intent to execute a tool.

    Correlates retries of the same logical action across checkpoints using `operation_id`,
    while assigning a unique `action_id` to each physical attempt.
    """
    action_id: str
    task_id: str
    operation_id: str
    tool_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    risk_level: ActionRiskLevel = ActionRiskLevel.RISKY_MUTATION
    checkpoint_id: Optional[str] = None
    timestamp: float = 0.0
    is_compensatable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ActionIntent:
        data_copy = dict(data)
        if isinstance(data_copy.get("risk_level"), str):
            data_copy["risk_level"] = ActionRiskLevel(data_copy["risk_level"])
        return cls(**data_copy)


@dataclass(frozen=True)
class EffectEvidence:
    """Structured evidence piece gathered about an action's side effects."""
    evidence_id: str
    action_id: str
    task_id: str
    operation_id: str
    source: EvidenceSource
    state: EvidenceState
    details: str
    observed_at: float
    raw_payload: Optional[Dict[str, Any]] = None
    checkpoint_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EffectEvidence:
        data_copy = dict(data)
        if isinstance(data_copy.get("source"), str):
            data_copy["source"] = EvidenceSource(data_copy["source"])
        if isinstance(data_copy.get("state"), str):
            data_copy["state"] = EvidenceState(data_copy["state"])
        return cls(**data_copy)


@dataclass(frozen=True)
class CheckpointMetadata:
    """Metadata recorded for an agent internal state checkpoint."""
    checkpoint_id: str
    task_id: str
    step_index: int
    created_at: float
    state_hash: str
    agent_state: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CheckpointMetadata:
        return cls(**data)


class InvariantStatus(str, Enum):
    """Evaluation status of a task invariant."""
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class InvariantResult:
    """Outcome of validating a task-specific business invariant."""
    invariant_id: str
    name: str
    satisfied: bool
    details: str
    evaluated_at: float
    affected_entities: List[str] = field(default_factory=list)
    status: InvariantStatus = InvariantStatus.PASS
    operation_id: Optional[str] = None
    is_safety_invariant: bool = True
    explanation: str = ""

    def __post_init__(self) -> None:
        if not self.explanation and self.details:
            object.__setattr__(self, "explanation", self.details)
        if self.status == InvariantStatus.PASS and not self.satisfied:
            object.__setattr__(self, "status", InvariantStatus.FAIL)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> InvariantResult:
        data_copy = dict(data)
        if isinstance(data_copy.get("status"), str):
            data_copy["status"] = InvariantStatus(data_copy["status"])
        return cls(**data_copy)


@dataclass(frozen=True)
class RecoveryDecision:
    """Formal decision rendered by the Recovery Gate before any post-restore tool invocation."""
    decision_id: str
    task_id: str
    action_id: str
    operation_id: str
    outcome: RecoveryOutcome
    is_safe_to_execute: bool
    rationale: str
    timestamp: float
    checkpoint_id: Optional[str] = None
    evidence_list: List[EffectEvidence] = field(default_factory=list)
    invariants_checked: List[InvariantResult] = field(default_factory=list)

    @property
    def standardized_outcome(self) -> str:
        """Standardizes outcome into one of the four core outcomes: RESUME, RETRY, QUARANTINE, HUMAN_INTERVENTION."""
        if self.outcome in (RecoveryOutcome.RETRY, RecoveryOutcome.SAFE_RETRY):
            return "RETRY"
        if self.outcome in (RecoveryOutcome.HUMAN_INTERVENTION, RecoveryOutcome.HUMAN_REVIEW):
            return "HUMAN_INTERVENTION"
        if self.outcome in (RecoveryOutcome.RECONCILE, RecoveryOutcome.COMPENSATE, RecoveryOutcome.QUARANTINE):
            return "QUARANTINE"
        return "RESUME"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RecoveryDecision:
        data_copy = dict(data)
        if isinstance(data_copy.get("outcome"), str):
            data_copy["outcome"] = RecoveryOutcome(data_copy["outcome"])
        if "evidence_list" in data_copy and data_copy["evidence_list"]:
            data_copy["evidence_list"] = [
                EffectEvidence.from_dict(e) if isinstance(e, dict) else e
                for e in data_copy["evidence_list"]
            ]
        if "invariants_checked" in data_copy and data_copy["invariants_checked"]:
            data_copy["invariants_checked"] = [
                InvariantResult.from_dict(i) if isinstance(i, dict) else i
                for i in data_copy["invariants_checked"]
            ]
        return cls(**data_copy)


@dataclass
class EffectRecord:
    """Row record stored in the durable SQLite effect ledger."""
    record_id: int
    action_id: str
    task_id: str
    operation_id: str
    tool_name: str
    status: EffectStatus
    params: Dict[str, Any]
    result: Optional[Dict[str, Any]]
    error_message: Optional[str]
    checkpoint_id: Optional[str]
    created_at: float
    updated_at: float
    evidence_list: List[EffectEvidence] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GateDeniedError(RuntimeError):
    """Raised when a side-effecting tool invocation is denied by the recovery gate."""
    pass


class RecoveryVerificationError(RuntimeError):
    """Raised when recovery verification encounters an unexpected failure and fails closed."""
    pass


@dataclass(frozen=True)
class GateAuthorization:
    """Scoped authorization token issued by the recovery gate.

    Guarantees authorization is strictly scoped to the verified task, operation,
    tool name, and outcome, preventing broad or stale future permissions.
    """
    task_id: str
    operation_id: str
    tool_name: str
    outcome: RecoveryOutcome
    decision_id: str
    timestamp: float
    is_reconciliation: bool = False
    is_compensation: bool = False
    evidence_signature: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

