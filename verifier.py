"""Evidence verification engine for RecoveryGuard.

Cross-references the durable effect ledger and external read-only probes to synthesize
an unambiguous EvidenceState (CONFIRMED_SUCCESS, CONFIRMED_ABSENT, UNKNOWN, CONFLICTING).

Strict Epistemic Integrity Rules:
1. A timeout must never be interpreted as proof of failure.
2. A missing ledger entry must not be interpreted as proof that an action never occurred.
3. An unavailable or stale external probe must not be treated as reliable confirmation.
4. Conflicting ledger and external evidence must remain explicitly conflicting until reconciled.
5. External probes must be read-only and must not trigger side effects.
6. Do not fabricate evidence when a service does not support verification.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from recoveryguard.ledger import DurableEffectLedger
from recoveryguard.models import (
    ActionIntent,
    EffectEvidence,
    EffectRecord,
    EffectStatus,
    EvidenceSource,
    EvidenceState,
)


class StateVerifier:
    """Synthesizes effect ledger history and active probe reports into structured evidence."""

    def __init__(self, ledger: DurableEffectLedger) -> None:
        self.ledger = ledger

    def verify_action(
        self,
        intent: ActionIntent,
        probe_data: Optional[Dict[str, Any]] = None,
        probe_supported: bool = True,
        probe_available: bool = True,
        probe_is_stale: bool = False,
    ) -> List[EffectEvidence]:
        """Collects and synthesizes structured evidence regarding a proposed or restored action.

        Does not trigger side effects.
        """
        evidences: List[EffectEvidence] = []
        now = time.time()

        # ------------------------------------------------------------------
        # 1. Inspect Durable Effect Ledger
        # ------------------------------------------------------------------
        prior_effects = self.ledger.get_effects_for_operation(intent.operation_id)
        has_committed_ledger = any(
            e.status in (EffectStatus.COMMITTED, EffectStatus.CONFIRMED_SUCCESS)
            for e in prior_effects
        )
        has_timeout_ledger = any(
            e.status in (EffectStatus.TIMEOUT, EffectStatus.UNKNOWN)
            for e in prior_effects
        )
        has_in_flight_or_attempted = any(
            e.status in (EffectStatus.PENDING, EffectStatus.STARTED, EffectStatus.ATTEMPTED)
            for e in prior_effects
        )
        has_explicit_absent = any(
            e.status == EffectStatus.CONFIRMED_ABSENT
            for e in prior_effects
        )

        if has_committed_ledger:
            evidences.append(
                EffectEvidence(
                    evidence_id=str(uuid.uuid4()),
                    action_id=intent.action_id,
                    task_id=intent.task_id,
                    operation_id=intent.operation_id,
                    source=EvidenceSource.LOCAL_LEDGER,
                    state=EvidenceState.CONFIRMED_SUCCESS,
                    details="Durable effect ledger confirms action previously COMMITTED.",
                    observed_at=now,
                    checkpoint_id=intent.checkpoint_id,
                )
            )
        elif has_timeout_ledger:
            # Rule: A timeout must NEVER be interpreted as proof of failure
            evidences.append(
                EffectEvidence(
                    evidence_id=str(uuid.uuid4()),
                    action_id=intent.action_id,
                    task_id=intent.task_id,
                    operation_id=intent.operation_id,
                    source=EvidenceSource.LOCAL_LEDGER,
                    state=EvidenceState.UNKNOWN,
                    details="Durable ledger records in-flight TIMEOUT. Status is UNKNOWN (never assumed failed).",
                    observed_at=now,
                    checkpoint_id=intent.checkpoint_id,
                )
            )
        elif has_in_flight_or_attempted:
            evidences.append(
                EffectEvidence(
                    evidence_id=str(uuid.uuid4()),
                    action_id=intent.action_id,
                    task_id=intent.task_id,
                    operation_id=intent.operation_id,
                    source=EvidenceSource.LOCAL_LEDGER,
                    state=EvidenceState.UNKNOWN,
                    details="Durable ledger records prior attempt in STARTED/ATTEMPTED state without resolution.",
                    observed_at=now,
                    checkpoint_id=intent.checkpoint_id,
                )
            )
        elif has_explicit_absent:
            evidences.append(
                EffectEvidence(
                    evidence_id=str(uuid.uuid4()),
                    action_id=intent.action_id,
                    task_id=intent.task_id,
                    operation_id=intent.operation_id,
                    source=EvidenceSource.LOCAL_LEDGER,
                    state=EvidenceState.CONFIRMED_ABSENT,
                    details="Durable ledger records prior verification of CONFIRMED_ABSENT.",
                    observed_at=now,
                    checkpoint_id=intent.checkpoint_id,
                )
            )
        else:
            # Rule: A missing ledger entry must NOT be interpreted as proof that an action never occurred!
            evidences.append(
                EffectEvidence(
                    evidence_id=str(uuid.uuid4()),
                    action_id=intent.action_id,
                    task_id=intent.task_id,
                    operation_id=intent.operation_id,
                    source=EvidenceSource.LOCAL_LEDGER,
                    state=EvidenceState.UNKNOWN,
                    details="No prior record found in durable ledger. (Missing ledger record does NOT prove external absence).",
                    observed_at=now,
                    checkpoint_id=intent.checkpoint_id,
                )
            )

        # ------------------------------------------------------------------
        # 2. Inspect Active External Probe (Read-only, non-mutating)
        # ------------------------------------------------------------------
        # Check if probe_data carries stale metadata or if probe is flagged stale
        is_stale = probe_is_stale or (
            isinstance(probe_data, dict) and bool(probe_data.get("is_stale", False))
        )

        if not probe_supported:
            # Rule: Do not fabricate evidence when a service does not support verification
            evidences.append(
                EffectEvidence(
                    evidence_id=str(uuid.uuid4()),
                    action_id=intent.action_id,
                    task_id=intent.task_id,
                    operation_id=intent.operation_id,
                    source=EvidenceSource.EXTERNAL_PROBE,
                    state=EvidenceState.UNKNOWN,
                    details="External service does not support read-only status verification.",
                    observed_at=now,
                    checkpoint_id=intent.checkpoint_id,
                )
            )
        elif not probe_available:
            # Rule: An unavailable external probe must not be treated as reliable confirmation
            evidences.append(
                EffectEvidence(
                    evidence_id=str(uuid.uuid4()),
                    action_id=intent.action_id,
                    task_id=intent.task_id,
                    operation_id=intent.operation_id,
                    source=EvidenceSource.EXTERNAL_PROBE,
                    state=EvidenceState.UNKNOWN,
                    details="External status probe is currently unavailable or unreachable.",
                    observed_at=now,
                    checkpoint_id=intent.checkpoint_id,
                )
            )
        elif is_stale:
            # Rule: Stale external probe must not be treated as reliable confirmation
            evidences.append(
                EffectEvidence(
                    evidence_id=str(uuid.uuid4()),
                    action_id=intent.action_id,
                    task_id=intent.task_id,
                    operation_id=intent.operation_id,
                    source=EvidenceSource.EXTERNAL_PROBE,
                    state=EvidenceState.UNKNOWN,
                    details="External probe evidence is stale and cannot be treated as reliable confirmation.",
                    observed_at=now,
                    raw_payload=probe_data,
                    checkpoint_id=intent.checkpoint_id,
                )
            )
        else:
            # Probe is supported, available, and fresh
            if probe_data is not None:
                if isinstance(probe_data, dict):
                    status_desc = probe_data.get("status", "FOUND")
                    raw_payload = probe_data
                elif isinstance(probe_data, list):
                    status_desc = f"{len(probe_data)} item(s) found"
                    raw_payload = {"items": probe_data}
                else:
                    status_desc = str(probe_data)
                    raw_payload = {"value": probe_data}

                evidences.append(
                    EffectEvidence(
                        evidence_id=str(uuid.uuid4()),
                        action_id=intent.action_id,
                        task_id=intent.task_id,
                        operation_id=intent.operation_id,
                        source=EvidenceSource.EXTERNAL_PROBE,
                        state=EvidenceState.CONFIRMED_SUCCESS,
                        details=f"Fresh external probe confirmed entity presence: {status_desc}",
                        observed_at=now,
                        raw_payload=raw_payload,
                        checkpoint_id=intent.checkpoint_id,
                    )
                )
            else:
                evidences.append(
                    EffectEvidence(
                        evidence_id=str(uuid.uuid4()),
                        action_id=intent.action_id,
                        task_id=intent.task_id,
                        operation_id=intent.operation_id,
                        source=EvidenceSource.EXTERNAL_PROBE,
                        state=EvidenceState.CONFIRMED_ABSENT,
                        details="Fresh external probe affirmatively confirmed entity is absent.",
                        observed_at=now,
                        checkpoint_id=intent.checkpoint_id,
                    )
                )

        return evidences

    def synthesize_state(self, evidences: List[EffectEvidence]) -> EvidenceState:
        """Deterministically synthesizes multiple evidence items into a unified EvidenceState.

        Epistemic Synthesis Hierarchy:
        1. CONFLICTING:
           - Ledger claims COMMITTED (CONFIRMED_SUCCESS), but fresh external probe
             affirmatively proves the entity is absent (CONFIRMED_ABSENT).
           - These must remain explicitly conflicting until reconciled.
        2. CONFIRMED_SUCCESS:
           - External probe confirms presence (ground truth).
           - OR durable ledger confirms committed with no contradicting probe.
        3. CONFIRMED_ABSENT:
           - ONLY when a fresh, reliable external probe affirmatively verifies absence,
             and the ledger does not claim committed success.
           - Missing ledger records or unavailable/stale probes CANNOT establish confirmed absence!
        4. UNKNOWN:
           - Default conservative epistemic state under uncertainty (in-flight timeout,
             unsupported probe, unavailable probe, stale probe, or unverified missing records).
        """
        ledger_ev = next((e for e in evidences if e.source == EvidenceSource.LOCAL_LEDGER), None)
        probe_ev = next((e for e in evidences if e.source == EvidenceSource.EXTERNAL_PROBE), None)

        ledger_state = ledger_ev.state if ledger_ev else EvidenceState.UNKNOWN
        probe_state = probe_ev.state if probe_ev else EvidenceState.UNKNOWN

        # 1. Contradiction: Ledger claims committed, but fresh external probe confirms absent
        if ledger_state == EvidenceState.CONFIRMED_SUCCESS and probe_state == EvidenceState.CONFIRMED_ABSENT:
            return EvidenceState.CONFLICTING

        # 2. External probe affirmatively proves success (ground truth in external reality)
        if probe_state == EvidenceState.CONFIRMED_SUCCESS:
            return EvidenceState.CONFIRMED_SUCCESS

        # 3. Durable ledger recorded commit and probe did not contradict (unsupported/unavailable probe)
        if ledger_state == EvidenceState.CONFIRMED_SUCCESS and probe_state != EvidenceState.CONFIRMED_ABSENT:
            return EvidenceState.CONFIRMED_SUCCESS

        # 4. Confirmed absence: REQUIRES fresh reliable probe confirming absence
        if probe_state == EvidenceState.CONFIRMED_ABSENT:
            # If ledger does not contradict with committed success, absence is established
            if ledger_state != EvidenceState.CONFIRMED_SUCCESS:
                return EvidenceState.CONFIRMED_ABSENT

        # 5. All other cases represent uncertainty (unknown)
        return EvidenceState.UNKNOWN
