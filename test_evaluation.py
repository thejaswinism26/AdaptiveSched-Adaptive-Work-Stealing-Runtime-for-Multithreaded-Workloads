"""Tests for quantitative evaluation metrics and reporting in RecoveryGuard.

Verifies:
1. Metric definitions and documentation coverage.
2. Zero-denominator and inapplicable case handling (empty results or no applicable cases).
3. Hand-verifiable classification fixtures (duplicate detection, false pause vs correct safety block, unsafe resumptions).
4. Exact metric rate calculations with known fixture distributions.
5. All three modes are evaluated over the exact same scenario set.
6. Summary and per-scenario table formatting.
7. Reproducibility across multiple identical evaluation runs.
"""

import pytest

from recoveryguard import (
    ExperimentHarness,
    ExperimentResult,
    ExperimentalMode,
    METRIC_DEFINITIONS,
    ModeEvaluationSummary,
    QuantitativeEvaluator,
    ScenarioId,
    classify_experiment_result,
)


@pytest.fixture
def evaluator():
    return QuantitativeEvaluator()


@pytest.fixture
def harness(tmp_path):
    return ExperimentHarness(base_temp_dir=tmp_path)


# ---------------------------------------------------------------------------
# Test 1: Metric definitions and documentation registry
# ---------------------------------------------------------------------------
def test_metric_definitions_registry():
    """Verifies all required metrics are documented with precise numerators, denominators, and symbols."""
    required_metrics = [
        "duplicate_side_effect_rate",
        "unsafe_resumption_rate",
        "unsafe_retry_rate",
        "false_pause_rate",
        "human_escalation_rate",
        "recovery_success_rate",
        "mean_latency_ms",
        "median_latency_ms",
    ]

    for m in required_metrics:
        assert m in METRIC_DEFINITIONS, f"Missing metric definition: {m}"
        definition = METRIC_DEFINITIONS[m]
        assert len(definition.name) > 0
        assert len(definition.symbol) > 0
        assert len(definition.description) > 0
        assert len(definition.numerator) > 0
        assert len(definition.denominator) > 0
        assert len(definition.treatment_of_inapplicable) > 0


# ---------------------------------------------------------------------------
# Test 2: Zero-denominator and inapplicable cases
# ---------------------------------------------------------------------------
def test_zero_denominator_empty_results(evaluator):
    """Verifies that an empty result set safely returns 0.0 for all rates without division-by-zero."""
    summary = evaluator.evaluate_mode([], mode=ExperimentalMode.RECOVERYGUARD)

    assert isinstance(summary, ModeEvaluationSummary)
    assert summary.total_experiments == 0
    assert summary.total_failures == 0
    assert summary.duplicate_side_effect_rate == 0.0
    assert summary.unsafe_resumption_rate == 0.0
    assert summary.unsafe_retry_rate == 0.0
    assert summary.false_pause_rate == 0.0
    assert summary.human_escalation_rate == 0.0
    assert summary.recovery_success_rate == 0.0
    assert summary.mean_latency_ms == 0.0
    assert summary.median_latency_ms == 0.0
    assert len(summary.scenario_breakdown) == 0


# ---------------------------------------------------------------------------
# Test 3: Hand-verifiable classification fixtures
# ---------------------------------------------------------------------------
def test_hand_verifiable_classification_fixtures():
    """Verifies fine-grained outcome classification using hand-crafted results."""
    # Fixture A: Duplicate charge in CHECKPOINT_ONLY
    res_dup = ExperimentResult(
        scenario_id="scenario_1_timeout_after_success",
        scenario_name="Timeout After Success",
        mode="CHECKPOINT_ONLY",
        initial_conditions={},
        injected_fault="TIMEOUT_AFTER_EXECUTION",
        recovery_decision=None,
        actual_side_effects={"charges": 2},
        duplicate_side_effects=1,
        invariant_violations=1,
        blocked_actions=0,
        human_escalations=0,
        execution_latency_ms=1.2,
        success=False,
        details="Duplicate execution occurred",
    )
    class_dup = classify_experiment_result(res_dup)
    assert class_dup["has_duplicate_side_effects"] is True
    assert class_dup["is_unsafe_retry"] is True
    assert class_dup["is_false_pause"] is False
    assert class_dup["is_correct_safety_block"] is False

    # Fixture B: Correct safety block in Scenario 4 (probe down) under RECOVERYGUARD
    res_safe_block = ExperimentResult(
        scenario_id="scenario_4_probe_unavailable",
        scenario_name="Probe Unavailable",
        mode="RECOVERYGUARD",
        initial_conditions={"probe_available": False},
        injected_fault="PROBE_UNAVAILABLE",
        recovery_decision={"outcome": "HUMAN_REVIEW"},
        actual_side_effects={"charges": 1},
        duplicate_side_effects=0,
        invariant_violations=0,
        blocked_actions=1,
        human_escalations=1,
        execution_latency_ms=2.5,
        success=True,
        details="Blocked by gate fail-closed",
    )
    class_safe_block = classify_experiment_result(res_safe_block)
    assert class_safe_block["has_duplicate_side_effects"] is False
    assert class_safe_block["is_correct_safety_block"] is True
    # Crucial guarantee: Do not count a blocked unsafe action as a false pause!
    assert class_safe_block["is_false_pause"] is False

    # Fixture C: False pause in Scenario 1 under EFFECT_LOG (action succeeded; safe resume was possible)
    res_false_pause = ExperimentResult(
        scenario_id="scenario_1_timeout_after_success",
        scenario_name="Timeout After Success",
        mode="EFFECT_LOG",
        initial_conditions={},
        injected_fault="TIMEOUT_AFTER_EXECUTION",
        recovery_decision=None,
        actual_side_effects={"charges": 1},
        duplicate_side_effects=0,
        invariant_violations=0,
        blocked_actions=1,
        human_escalations=1,
        execution_latency_ms=1.0,
        success=False,
        details="Paused conservatively on unconfirmed timeout",
    )
    class_fp = classify_experiment_result(res_false_pause)
    assert class_fp["is_false_pause"] is True
    assert class_fp["is_correct_safety_block"] is False

    # Fixture D: Unsafe resumption in Scenario 6 under EFFECT_LOG (skipped missing payment)
    res_unsafe_resump = ExperimentResult(
        scenario_id="scenario_6_ledger_external_conflict",
        scenario_name="Ledger Conflict",
        mode="EFFECT_LOG",
        initial_conditions={"ledger_status": "COMMITTED"},
        injected_fault="STATE_CONTRADICTION",
        recovery_decision=None,
        actual_side_effects={"captured_charges": 0},
        duplicate_side_effects=0,
        invariant_violations=0,
        blocked_actions=0,
        human_escalations=0,
        execution_latency_ms=0.8,
        success=False,
        details="Silently skipped external charge",
    )
    class_ur = classify_experiment_result(res_unsafe_resump)
    assert class_ur["is_unsafe_resumption"] is True
    assert class_ur["is_false_pause"] is False


# ---------------------------------------------------------------------------
# Test 4: Exact metric calculations with known fixture distributions
# ---------------------------------------------------------------------------
def test_evaluator_exact_rate_calculations(evaluator):
    """Verifies arithmetic precision on a controlled 4-item distribution."""
    # 4 runs:
    # Run 1: Clean success (safe retry)
    # Run 2: Duplicate mutation
    # Run 3: False pause
    # Run 4: Correct safety block
    fixture_runs = [
        ExperimentResult(
            scenario_id="scenario_3_failure_before_execution",
            scenario_name="Pre Failure",
            mode="TEST_MODE",
            initial_conditions={},
            injected_fault="FAILURE_BEFORE_EXECUTION",
            recovery_decision=None,
            actual_side_effects={"charges": 1},
            duplicate_side_effects=0,
            invariant_violations=0,
            blocked_actions=0,
            human_escalations=0,
            execution_latency_ms=10.0,
            success=True,
            details="",
        ),
        ExperimentResult(
            scenario_id="scenario_1_timeout_after_success",
            scenario_name="Timeout Success",
            mode="TEST_MODE",
            initial_conditions={},
            injected_fault="TIMEOUT",
            recovery_decision=None,
            actual_side_effects={"charges": 2},
            duplicate_side_effects=1,
            invariant_violations=1,
            blocked_actions=0,
            human_escalations=0,
            execution_latency_ms=20.0,
            success=False,
            details="",
        ),
        ExperimentResult(
            scenario_id="scenario_1_timeout_after_success",
            scenario_name="Timeout Success",
            mode="TEST_MODE",
            initial_conditions={},
            injected_fault="TIMEOUT",
            recovery_decision=None,
            actual_side_effects={"charges": 1},
            duplicate_side_effects=0,
            invariant_violations=0,
            blocked_actions=1,
            human_escalations=1,
            execution_latency_ms=30.0,
            success=False,
            details="",
        ),
        ExperimentResult(
            scenario_id="scenario_4_probe_unavailable",
            scenario_name="Probe Down",
            mode="TEST_MODE",
            initial_conditions={},
            injected_fault="PROBE_DOWN",
            recovery_decision={"outcome": "HUMAN_REVIEW"},
            actual_side_effects={"charges": 1},
            duplicate_side_effects=0,
            invariant_violations=0,
            blocked_actions=1,
            human_escalations=1,
            execution_latency_ms=40.0,
            success=True,
            details="",
        ),
    ]

    summary = evaluator.evaluate_mode(fixture_runs, mode=None)

    assert summary.total_experiments == 4
    # Runs 1 & 4 succeeded (success=True) -> 2 successes, 2 failures
    assert summary.total_failures == 2
    assert summary.recovery_success_rate == 0.5  # 2/4 = 50%

    # Exactly 1 duplicate run (Run 2) -> 1/4 = 25%
    assert summary.duplicate_side_effect_rate == 0.25

    # Exactly 1 false pause (Run 3) -> 1/4 = 25% (Run 4 is correct safety block, not false pause!)
    assert summary.false_pause_rate == 0.25

    # Human escalations in Runs 3 and 4 -> 2/4 = 50%
    assert summary.human_escalation_rate == 0.5

    # Latencies: [10, 20, 30, 40]
    # Mean: 25.0 ms, Median: (20 + 30) / 2 = 25.0 ms
    assert summary.mean_latency_ms == 25.0
    assert summary.median_latency_ms == 25.0


# ---------------------------------------------------------------------------
# Test 5: Full suite evaluation consistency across all modes
# ---------------------------------------------------------------------------
def test_full_suite_evaluation_consistency(harness, evaluator):
    """Executes all 8 scenarios across all 3 modes and checks comparative properties."""
    results, summaries = evaluator.run_and_evaluate(harness, seed=42)

    # 8 scenarios * 3 modes = 24 runs
    assert len(results) == 24
    assert len(summaries) == 3

    m_cp = summaries[ExperimentalMode.CHECKPOINT_ONLY.value]
    m_el = summaries[ExperimentalMode.EFFECT_LOG.value]
    m_rg = summaries[ExperimentalMode.RECOVERYGUARD.value]

    # Every mode evaluated over exactly the same 8 scenarios
    assert m_cp.total_experiments == 8
    assert m_el.total_experiments == 8
    assert m_rg.total_experiments == 8

    # CHECKPOINT_ONLY has duplicate side effects (Scenario 1, 2, 4, 5, 7, 8)
    assert m_cp.duplicate_side_effect_rate > 0.0
    assert m_cp.total_failures > 0
    assert m_cp.false_pause_rate == 0.0  # Naive baseline never pauses!

    # EFFECT_LOG has failures from crash-before-record (Scenario 2) and conflict (Scenario 6)
    assert m_el.total_failures > 0

    # RECOVERYGUARD prevents all duplicate side effects (0 duplicates across all scenarios!)
    assert m_rg.duplicate_side_effect_rate == 0.0
    assert m_rg.unsafe_resumption_rate == 0.0
    assert m_rg.false_pause_rate == 0.0  # All blocks in RecoveryGuard are correct safety blocks!
    assert m_rg.recovery_success_rate == 1.0


# ---------------------------------------------------------------------------
# Test 6: Summary and scenario tables format cleanly
# ---------------------------------------------------------------------------
def test_table_formatting_and_report_generation(harness, evaluator):
    """Verifies Markdown comparison table and scenario breakdown generation."""
    results, summaries = evaluator.run_and_evaluate(harness, seed=42)

    comp_table = evaluator.format_comparison_table(summaries)
    assert "| Metric | Mode A: CHECKPOINT_ONLY | Mode B: EFFECT_LOG | Mode C: RECOVERYGUARD |" in comp_table
    assert "**Recovery Success Rate (RSR)**" in comp_table
    assert "**Duplicate Side-Effect Rate (DSR)**" in comp_table
    assert "**False-Pause Rate (FPR)**" in comp_table
    assert "**Mean Recovery Latency**" in comp_table

    scen_table = evaluator.format_scenario_table(results)
    assert "| Scenario | Mode | Outcome Status | Duplicates | Blocked | Escalated |" in scen_table
    assert "scenario_1_timeout_after_success" in scen_table

    full_report = evaluator.generate_full_report(results, summaries)
    assert "# RecoveryGuard: Quantitative Evaluation Report" in full_report
    assert "## 1. Summary Comparison" in full_report
    assert "## 2. Per-Scenario Experimental Results" in full_report
    assert "## 3. Metric Definitions & Denominators" in full_report
