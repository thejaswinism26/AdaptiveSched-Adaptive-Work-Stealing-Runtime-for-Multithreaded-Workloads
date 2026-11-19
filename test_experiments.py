"""Tests for RecoveryGuard experimental modes, failure scenarios, and benchmarking harness.

Verifies:
1. All three modes execute the same scenario definitions.
2. Experimental runs have strictly isolated state (no leak between runs/modes).
3. Repeated runs with identical settings produce identical, deterministic outcomes.
4. The CHECKPOINT_ONLY baseline reproduces the duplicate-side-effect failure.
5. RecoveryGuard blocks unsafe retries when evidence remains unknown.
6. Probe unavailability and conflicting evidence are recorded correctly.
7. Effect-Log baseline policies and limitations are accurately evaluated.
8. Backward compatibility helper methods function as expected.
"""

import pytest

from recoveryguard import (
    ExperimentHarness,
    ExperimentResult,
    ExperimentalMode,
    RecoveryOutcome,
    SCENARIO_REGISTRY,
    ScenarioId,
    ScenarioResult,
)


@pytest.fixture
def harness(tmp_path):
    """Provides an ExperimentHarness with isolated base temp dir."""
    return ExperimentHarness(base_temp_dir=tmp_path)


# ---------------------------------------------------------------------------
# Test 1: All three modes execute the same scenario definitions
# ---------------------------------------------------------------------------
def test_all_three_modes_execute_same_scenario_definitions(harness):
    """Verifies all 8 scenario definitions execute under CHECKPOINT_ONLY, EFFECT_LOG, and RECOVERYGUARD."""
    modes = [
        ExperimentalMode.CHECKPOINT_ONLY,
        ExperimentalMode.EFFECT_LOG,
        ExperimentalMode.RECOVERYGUARD,
    ]

    for sid in ScenarioId:
        meta = SCENARIO_REGISTRY[sid]
        assert meta.scenario_id == sid

        for m in modes:
            res = harness.run_scenario(sid, m)
            assert isinstance(res, ExperimentResult)
            assert res.scenario_id == sid.value
            assert res.scenario_name == meta.name
            assert res.mode == m.value
            assert res.injected_fault == meta.injected_fault
            assert isinstance(res.actual_side_effects, dict)
            assert res.execution_latency_ms >= 0.0


# ---------------------------------------------------------------------------
# Test 2: Experimental runs have isolated state
# ---------------------------------------------------------------------------
def test_experimental_runs_have_isolated_state(harness):
    """Verifies runs do not leak external service state, ledgers, or checkpoints."""
    # Run 1: Mutates payment gateway in CHECKPOINT_ONLY mode (creates 2 charges)
    res1 = harness.run_scenario(ScenarioId.SCENARIO_1_TIMEOUT_AFTER_SUCCESS, ExperimentalMode.CHECKPOINT_ONLY)
    assert res1.actual_side_effects["captured_charges"] == 2

    # Run 2: Fresh run of Scenario 3 (pre-execution failure) in another mode
    # Should start with an empty payment gateway and 0 charges prior to retry
    res2 = harness.run_scenario(ScenarioId.SCENARIO_3_FAILURE_BEFORE_EXECUTION, ExperimentalMode.RECOVERYGUARD)
    assert res2.actual_side_effects["captured_charges"] == 1
    assert res2.duplicate_side_effects == 0

    # Run 3: Direct check on a freshly constructed environment
    env = harness.create_isolated_environment("test_leak_check")
    assert len(env.payment_gw._charges) == 0
    assert len(env.notifications._sent_messages) == 0
    assert len(env.deployments._deployments) == 0
    assert len(env.ledger.get_effects_for_task("any_task")) == 0
    env.cleanup()


# ---------------------------------------------------------------------------
# Test 3: Repeated runs with identical settings produce identical outcomes
# ---------------------------------------------------------------------------
def test_repeated_runs_with_identical_settings_produce_identical_outcomes(harness):
    """Verifies determinism: repeated runs with the same settings yield identical metrics."""
    modes = [
        ExperimentalMode.CHECKPOINT_ONLY,
        ExperimentalMode.EFFECT_LOG,
        ExperimentalMode.RECOVERYGUARD,
    ]

    for sid in (
        ScenarioId.SCENARIO_1_TIMEOUT_AFTER_SUCCESS,
        ScenarioId.SCENARIO_2_CRASH_BEFORE_RECORD,
        ScenarioId.SCENARIO_4_PROBE_UNAVAILABLE,
        ScenarioId.SCENARIO_6_LEDGER_EXTERNAL_CONFLICT,
    ):
        for m in modes:
            run_a = harness.run_scenario(sid, m, seed=100)
            run_b = harness.run_scenario(sid, m, seed=100)

            assert run_a.success == run_b.success, f"Failed on {sid} {m}"
            assert run_a.duplicate_side_effects == run_b.duplicate_side_effects
            assert run_a.invariant_violations == run_b.invariant_violations
            assert run_a.blocked_actions == run_b.blocked_actions
            assert run_a.human_escalations == run_b.human_escalations
            assert run_a.actual_side_effects == run_b.actual_side_effects
            assert run_a.details == run_b.details


# ---------------------------------------------------------------------------
# Test 4: Checkpoint-only baseline reproduces duplicate side effects
# ---------------------------------------------------------------------------
def test_checkpoint_only_reproduces_duplicate_side_effect_failure(harness):
    """Verifies CHECKPOINT_ONLY reproduces duplicate actions upon rollback."""
    # Scenario 1: Timeout after external success
    s1 = harness.run_scenario(ScenarioId.SCENARIO_1_TIMEOUT_AFTER_SUCCESS, ExperimentalMode.CHECKPOINT_ONLY)
    assert s1.duplicate_side_effects == 1
    assert s1.invariant_violations == 1
    assert s1.success is False

    # Scenario 2: Crash before recording success
    s2 = harness.run_scenario(ScenarioId.SCENARIO_2_CRASH_BEFORE_RECORD, ExperimentalMode.CHECKPOINT_ONLY)
    assert s2.duplicate_side_effects == 1
    assert s2.invariant_violations == 1
    assert s2.success is False

    # Scenario 7: Rollback to older checkpoint after multiple effects
    s7 = harness.run_scenario(ScenarioId.SCENARIO_7_OLDER_CHECKPOINT_RESTORED, ExperimentalMode.CHECKPOINT_ONLY)
    assert s7.duplicate_side_effects == 2  # 1 duplicate charge + 1 duplicate notification
    assert s7.invariant_violations == 2
    assert s7.success is False


# ---------------------------------------------------------------------------
# Test 5: RecoveryGuard blocks unsafe retries when evidence remains unknown
# ---------------------------------------------------------------------------
def test_recoveryguard_blocks_unsafe_retries_when_evidence_unknown(harness):
    """Verifies RecoveryGuard halts closed on unknown evidence for risky mutations."""
    # Scenario 4: Probe unavailable -> HUMAN_REVIEW
    s4 = harness.run_scenario(ScenarioId.SCENARIO_4_PROBE_UNAVAILABLE, ExperimentalMode.RECOVERYGUARD)
    assert s4.blocked_actions == 1
    assert s4.human_escalations == 1
    assert s4.duplicate_side_effects == 0
    assert s4.recovery_decision["outcome"] == RecoveryOutcome.HUMAN_REVIEW.value
    assert s4.recovery_decision["is_safe_to_execute"] is False
    assert s4.success is True  # Safely prevented duplicate mutation!

    # Scenario 5: Stale evidence -> HUMAN_REVIEW
    s5 = harness.run_scenario(ScenarioId.SCENARIO_5_PROBE_STALE, ExperimentalMode.RECOVERYGUARD)
    assert s5.blocked_actions == 1
    assert s5.human_escalations == 1
    assert s5.duplicate_side_effects == 0
    assert s5.recovery_decision["outcome"] == RecoveryOutcome.HUMAN_REVIEW.value
    assert s5.success is True

    # Scenario 8: Unsupported status probe -> HUMAN_REVIEW
    s8 = harness.run_scenario(ScenarioId.SCENARIO_8_NO_STATUS_QUERY, ExperimentalMode.RECOVERYGUARD)
    assert s8.blocked_actions == 1
    assert s8.human_escalations == 1
    assert s8.duplicate_side_effects == 0
    assert s8.recovery_decision["outcome"] == RecoveryOutcome.HUMAN_REVIEW.value
    assert s8.success is True


# ---------------------------------------------------------------------------
# Test 6: Probe unavailability and conflicting evidence recorded correctly
# ---------------------------------------------------------------------------
def test_probe_unavailability_and_conflicting_evidence_recorded_correctly(harness):
    """Verifies metadata and initial conditions are recorded for probe unavailability and conflicts."""
    # Scenario 4: Probe unavailable
    s4 = harness.run_scenario(ScenarioId.SCENARIO_4_PROBE_UNAVAILABLE, ExperimentalMode.RECOVERYGUARD)
    assert s4.initial_conditions.get("probe_available") is False
    assert s4.recovery_decision is not None
    assert s4.recovery_decision["outcome"] == RecoveryOutcome.HUMAN_REVIEW.value

    # Scenario 6: Conflicting evidence (Ledger COMMITTED vs External absent)
    s6 = harness.run_scenario(ScenarioId.SCENARIO_6_LEDGER_EXTERNAL_CONFLICT, ExperimentalMode.RECOVERYGUARD)
    assert s6.initial_conditions.get("ledger_status") == "COMMITTED"
    assert s6.initial_conditions.get("external_charges") == 0
    assert s6.recovery_decision is not None
    assert s6.recovery_decision["outcome"] == RecoveryOutcome.QUARANTINE.value
    assert s6.blocked_actions == 1
    assert s6.human_escalations == 1
    assert s6.success is True


# ---------------------------------------------------------------------------
# Test 7: Effect-Log baseline policies and limitations
# ---------------------------------------------------------------------------
def test_effect_log_baseline_limitations(harness):
    """Verifies Effect-Log limitations: blind to uncommitted crashes and external conflicts."""
    # Limitation 1: In Scenario 2, crash happened before writing commit record to ledger.
    # Effect-Log cannot prevent duplicate charge because client ledger has no record!
    s2 = harness.run_scenario(ScenarioId.SCENARIO_2_CRASH_BEFORE_RECORD, ExperimentalMode.EFFECT_LOG)
    assert s2.duplicate_side_effects == 1
    assert s2.invariant_violations == 1
    assert s2.success is False

    # Limitation 2: In Scenario 6, ledger records committed, but external service has nothing.
    # Effect-Log blindly skips (replay protection), believing ledger over external reality!
    s6 = harness.run_scenario(ScenarioId.SCENARIO_6_LEDGER_EXTERNAL_CONFLICT, ExperimentalMode.EFFECT_LOG)
    assert s6.actual_side_effects["captured_charges"] == 0  # Order was NEVER paid externally!
    assert s6.duplicate_side_effects == 0
    # Effect-Log failed to detect contradiction (it has no external probes)


# ---------------------------------------------------------------------------
# Test 8: Backward compatibility helper methods
# ---------------------------------------------------------------------------
def test_backward_compatibility_benchmark_methods(harness):
    """Verifies backward compatibility methods for benchmarks and dashboards."""
    bench = harness.run_crash_after_payment_scenario("recovery_guard")
    assert isinstance(bench, ScenarioResult)
    assert bench.mode == "RECOVERYGUARD"
    assert bench.duplicate_charges == 0
    assert bench.invariant_violations == 0

    all_bench = harness.run_all_benchmarks()
    assert len(all_bench) == 3
    assert all_bench[0].mode == "CHECKPOINT_ONLY"
    assert all_bench[1].mode == "EFFECT_LOG"
    assert all_bench[2].mode == "RECOVERYGUARD"
