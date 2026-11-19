"""Deterministic, zero-LLM Recovery Decision Engine.

Evaluates evidence states, task invariants, tool risk levels, and reversibility to output
one of six explicit recovery outcomes:
- RESUME
- RECONCILE
- SAFE_RETRY
- COMPENSATE
- QUARANTINE
- HUMAN_REVIEW

Strict Epistemic & Safety Principles:
1. Deterministic & Zero-LLM: Does not rely on natural language model judgment.
2. Confirmed Success: Do not repeat side effect; RESUME only if safety invariants pass.
3. Confirmed Absence: Allow SAFE_RETRY only if retry is demonstrably safe and all required
   safety conditions pass.
4. Unknown Evidence: Block risky mutations; request HUMAN_REVIEW or QUARANTINE.
5. Conflicting Evidence: Select RECONCILE or HUMAN_REVIEW. Never blindly retry.
6. Explicit Compensation: Action-specific; irreversible actions (e.g., notifications)
   cannot be undone and must never be marked COMPENSATE.
7. Conservative Defaults: Block execution whenever evidence is insufficient.

Note on Distributed Guarantees:
This prototype does NOT claim to guarantee exactly-once side effects across arbitrary
external black-box APIs without cooperation. Exactly-once semantics require external
service-side idempotency keys or authoritative read-only status query endpoints.
RecoveryGuard provides verification, uncertainty management, and cross-system recovery gating.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from recoveryguard.invariants import InvariantSuite
from recoveryguard.models import (
    ActionIntent,
    ActionRiskLevel,
    EffectEvidence,
    EvidenceState,
    InvariantResult,
    RecoveryDecision,
    RecoveryOutcome,
)
from recoveryguard.verifier import StateVerifier


class RecoveryEngine:
    """Deterministic Recovery Gate."""

    def __init__(
        self,
        verifier: StateVerifier,
        invariant_suite: Optional[InvariantSuite] = None,
        default_unknown_policy: str = "human_review",  # "human_review" or "quarantine"
    ) -> None:
        self.verifier = verifier
        self.invariant_suite = invariant_suite or InvariantSuite()
        self.default_unknown_policy = default_unknown_policy

    def is_action_reversible(self, intent: ActionIntent) -> bool:
        """Determines whether an action can be safely compensated/reversed.

        Notifications, emails, and SMS messages are inherently irreversible.
        Financial charges (refund) and inventory reservations (release) are compensatable.
        """
        if not intent.is_compensatable:
            return False
        tool_lower = intent.tool_name.lower()
        if any(term in tool_lower for term in ("notify", "notification", "email", "sms", "alert")):
            return False
        return True

    def evaluate_gate(
        self,
        intent: ActionIntent,
        probe_data: Optional[Dict[str, Any]] = None,
        probe_supported: bool = True,
        probe_available: bool = True,
        probe_is_stale: bool = False,
        invariant_context: Optional[Dict[str, Any]] = None,
        unknown_policy: Optional[str] = None,
        conflict_policy: Optional[str] = None,
    ) -> RecoveryDecision:
        """Deterministically evaluates whether it is safe for an agent to execute an action."""
        now = time.time()
        decision_id = f"dec_{uuid.uuid4().hex[:12]}"
        policy = unknown_policy or self.default_unknown_policy

        # ------------------------------------------------------------------
        # 1. Synthesize Multi-Source Evidence
        # ------------------------------------------------------------------
        evidences = self.verifier.verify_action(
            intent=intent,
            probe_data=probe_data,
            probe_supported=probe_supported,
            probe_available=probe_available,
            probe_is_stale=probe_is_stale,
        )
        synthesized_state = self.verifier.synthesize_state(evidences)

        # ------------------------------------------------------------------
        # 2. Check Task-Specific Invariants & Safety Conditions
        # ------------------------------------------------------------------
        ctx = dict(invariant_context or {})
        ctx.setdefault("operation_id", intent.operation_id)
        ctx.setdefault("task_id", intent.task_id)
        ctx.setdefault("probe_available", probe_available)
        ctx.setdefault("probe_is_stale", probe_is_stale)

        if "order_id" in intent.parameters:
            ctx.setdefault("order_id", intent.parameters["order_id"])

        if probe_data is not None:
            tool_lower = intent.tool_name.lower()
            if "charge" in tool_lower or "pay" in tool_lower:
                ctx.setdefault("payment_probe", probe_data)
            elif "reserve" in tool_lower or "inventory" in tool_lower:
                ctx.setdefault("reservation_probe", probe_data)
            elif "notify" in tool_lower or "notification" in tool_lower:
                ctx.setdefault("notification_probe", probe_data)
            elif "deploy" in tool_lower:
                ctx.setdefault("deployment_probe", probe_data)

        invariant_results = self.invariant_suite.evaluate_all(ctx)
        safety_passed = self.invariant_suite.safety_invariants_pass(invariant_results)
        failed_safety = self.invariant_suite.get_failed_safety_invariants(invariant_results)

        is_reversible = self.is_action_reversible(intent)

        # ------------------------------------------------------------------
        # 3. Deterministic Decision Matrix
        # ------------------------------------------------------------------

        # Rule 1: Safety Invariant Violations take precedence
        if not safety_passed:
            failed_names = [i.name for i in failed_safety]
            if is_reversible:
                outcome = RecoveryOutcome.COMPENSATE
                is_safe = False
                rationale = (
                    f"Execution blocked: Safety invariant violation detected ({', '.join(failed_names)}). "
                    f"Action '{intent.tool_name}' is compensatable; triggering explicit compensation workflow."
                )
            else:
                # Rule: Irreversible actions (notifications) must never be incorrectly marked COMPENSATE
                outcome = RecoveryOutcome.QUARANTINE
                is_safe = False
                rationale = (
                    f"Execution blocked: Safety invariant violation detected ({', '.join(failed_names)}). "
                    f"Action '{intent.tool_name}' is IRREVERSIBLE and cannot be compensated; isolating task under QUARANTINE."
                )

        # Rule 2: Conflicting Evidence
        elif synthesized_state == EvidenceState.CONFLICTING:
            # Rule: Conflicting evidence cannot authorize a blind retry
            is_safe = False
            if conflict_policy == "human_review":
                outcome = RecoveryOutcome.HUMAN_REVIEW
                rationale = (
                    "Execution blocked: Contradiction between durable effect ledger and external status probe. "
                    "Blind retry is strictly forbidden; escalating for HUMAN_REVIEW."
                )
            elif conflict_policy == "quarantine":
                outcome = RecoveryOutcome.QUARANTINE
                rationale = (
                    "Execution blocked: Contradiction between durable effect ledger and external status probe. "
                    "Blind retry is strictly forbidden; task quarantined."
                )
            else:
                outcome = RecoveryOutcome.RECONCILE
                rationale = (
                    "Execution blocked: Contradiction between durable effect ledger and external status probe. "
                    "Blind retry is strictly forbidden; task requires state RECONCILE to resolve divergence."
                )

        # Rule 3: Confirmed Success
        elif synthesized_state == EvidenceState.CONFIRMED_SUCCESS:
            # Rule: Do not repeat same logical side effect; resume workflow
            outcome = RecoveryOutcome.RESUME
            is_safe = True
            rationale = (
                "Action already confirmed committed externally. "
                "Bypassing duplicate tool mutation and resuming agent workflow (safety invariants satisfied)."
            )

        # Rule 4: Confirmed Absence
        elif synthesized_state == EvidenceState.CONFIRMED_ABSENT:
            # Rule: Allow SAFE_RETRY only if retry is demonstrably safe and all safety conditions pass
            outcome = RecoveryOutcome.SAFE_RETRY
            is_safe = True
            rationale = (
                "Action confirmed absent from external system via reliable probe and all safety invariants passed. "
                "Safe to execute tool."
            )

        # Rule 5: Unknown Evidence
        elif synthesized_state == EvidenceState.UNKNOWN:
            # Rule: Handle based on tool risk level
            if intent.risk_level == ActionRiskLevel.SAFE_READ:
                outcome = RecoveryOutcome.SAFE_RETRY
                is_safe = True
                rationale = (
                    "Execution status is unknown, but tool is read-only (SAFE_READ). "
                    "Safe to query without side effects."
                )
            elif intent.risk_level == ActionRiskLevel.IDEMPOTENT_WRITE:
                outcome = RecoveryOutcome.SAFE_RETRY
                is_safe = True
                rationale = (
                    "Execution status is unknown, but tool is IDEMPOTENT_WRITE with stable operation ID. "
                    "Safe to re-issue request."
                )
            else:
                # Rule: Block risky execution on unknown evidence
                is_safe = False
                if policy == "quarantine":
                    outcome = RecoveryOutcome.QUARANTINE
                    rationale = (
                        "Execution blocked: Status is UNKNOWN for RISKY_MUTATION. "
                        "Configured policy enforces immediate QUARANTINE to protect external state."
                    )
                else:
                    outcome = RecoveryOutcome.HUMAN_REVIEW
                    rationale = (
                        "Execution blocked: Status is UNKNOWN for RISKY_MUTATION (e.g. in-flight timeout, "
                        "stale probe, or missing external confirmation). Escalating for HUMAN_REVIEW."
                    )

        # Fallback defensive branch
        else:
            outcome = RecoveryOutcome.QUARANTINE
            is_safe = False
            rationale = f"Execution blocked: Unhandled epistemic state '{synthesized_state}'."

        decision = RecoveryDecision(
            decision_id=decision_id,
            task_id=intent.task_id,
            action_id=intent.action_id,
            operation_id=intent.operation_id,
            outcome=outcome,
            is_safe_to_execute=is_safe,
            rationale=rationale,
            timestamp=now,
            checkpoint_id=intent.checkpoint_id,
            evidence_list=evidences,
            invariants_checked=invariant_results,
        )

        # Record decision to durable effect ledger for auditability
        self.verifier.ledger.record_decision(decision)
        return decision
