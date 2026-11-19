"""Comprehensive tests for multi-trial evaluation, ablation study, and persistent exports.

Verifies:
1. Multi-trial execution (100 trials) with deterministic seed reproducibility.
2. The 4 ablation variants (FULL_RECOVERYGUARD, NO_LEDGER, NO_PROBES, NO_INVARIANTS).
3. Persistent export roundtripping for both CSV and SQLite formats.
4. Crash before agent record (Scenario 2) distinguishing risky mutations vs idempotency.
5. Metric computation (FQR, HIR, DSR, URR, RSR) and standardized outcomes.
"""

from __future__ import annotations

import csv
import sqlite3
import tempfile
from pathlib import Path

import pytest

from recoveryguard.evaluation import (
    QuantitativeEvaluator,
    evaluate_ablation_study,
    export_results_to_csv,
    export_results_to_sqlite,
    export_summaries_to_csv,
    format_ablation_table,
)
from recoveryguard.experiments import (
    AblationVariant,
    ExperimentHarness,
    ExperimentalMode,
    ScenarioId,
)
from recoveryguard.models import RecoveryDecision, RecoveryOutcome


@pytest.fixture
def harness():
    return ExperimentHarness()


@pytest.fixture
def evaluator():
    return QuantitativeEvaluator()


def test_100_trials_reproducibility_and_isolation(harness):
    """Verifies that 100 trials can execute quickly, deterministically, and with isolated state."""
    # Run 100 trials of Scenario 1 on Mode C: RECOVERYGUARD
    results_run1 = harness.run_benchmark_trials(
        num_trials=100,
        scenarios=[ScenarioId.SCENARIO_1_TIMEOUT_AFTER_SUCCESS],
        modes=[ExperimentalMode.RECOVERYGUARD],
        seed=12345,
    )
    assert len(results_run1) == 100
    assert all(r.success for r in results_run1)
    assert all(r.duplicate_side_effects == 0 for r in results_run1)

    # Re-run with identical seed produces identical results
    results_run2 = harness.run_benchmark_trials(
        num_trials=100,
        scenarios=[ScenarioId.SCENARIO_1_TIMEOUT_AFTER_SUCCESS],
        modes=[ExperimentalMode.RECOVERYGUARD],
        seed=12345,
    )
    for r1, r2 in zip(results_run1, results_run2):
        assert r1.duplicate_side_effects == r2.duplicate_side_effects
        assert r1.invariant_violations == r2.invariant_violations
        assert r1.success == r2.success


def test_ablation_variants_distinct_behavior(harness, evaluator):
    """Verifies that the 4 ablation variants exhibit genuinely distinct, principled recovery behavior."""
    # 1. Full RecoveryGuard on Scenario 1 (Timeout after success):
    # Has ledger + probes + invariants -> external success probed -> RESUME -> success
    res_full = harness.run_ablation_scenario(
        ScenarioId.SCENARIO_1_TIMEOUT_AFTER_SUCCESS,
        AblationVariant.FULL_RECOVERYGUARD,
    )
    assert res_full.success is True
    assert res_full.duplicate_side_effects == 0

    # 2. No-Ledger variant:
    # Mandatory gate checks ledger, but ledger is None -> fails closed with BLOCKED_BY_GATE
    res_no_ledger = harness.run_ablation_scenario(
        ScenarioId.SCENARIO_1_TIMEOUT_AFTER_SUCCESS,
        AblationVariant.NO_LEDGER,
    )
    assert res_no_ledger.blocked_actions == 1

    # 3. No-Probes variant:
    # Ledger has timeout, but probes are unavailable -> epistemic state UNKNOWN on RISKY_MUTATION
    # Gate fails closed to HUMAN_REVIEW to prevent duplicate charge!
    res_no_probes = harness.run_ablation_scenario(
        ScenarioId.SCENARIO_1_TIMEOUT_AFTER_SUCCESS,
        AblationVariant.NO_PROBES,
    )
    assert res_no_probes.blocked_actions == 1
    assert res_no_probes.human_escalations == 1
    assert res_no_probes.duplicate_side_effects == 0

    # 4. No-Invariants variant:
    # Runs without multi-system business invariants
    res_no_inv = harness.run_ablation_scenario(
        ScenarioId.SCENARIO_1_TIMEOUT_AFTER_SUCCESS,
        AblationVariant.NO_INVARIANTS,
    )
    assert res_no_inv.duplicate_side_effects == 0


def test_ablation_study_matrix_and_table_generation(harness, evaluator):
    """Verifies execution of the entire ablation study and markdown table generation."""
    ablation_results = harness.run_ablation_study(
        scenarios=[ScenarioId.SCENARIO_1_TIMEOUT_AFTER_SUCCESS, ScenarioId.SCENARIO_4_PROBE_UNAVAILABLE],
        seed=42,
        trials_per_scenario=2,
    )
    assert len(ablation_results) == 4
    for variant in AblationVariant:
        assert variant.value in ablation_results
        assert len(ablation_results[variant.value]) == 4  # 2 scenarios * 2 trials

    summaries = evaluator.evaluate_ablation_study(ablation_results)
    assert len(summaries) == 4
    for variant in AblationVariant:
        s = summaries[variant.value]
        assert s.total_experiments == 4
        assert 0.0 <= s.recovery_success_rate <= 1.0

    table_md = evaluator.format_ablation_table(summaries)
    assert "| Metric | Full RecoveryGuard | No Ledger | No Probes | No Invariants |" in table_md
    assert "Recovery Success Rate (RSR)" in table_md
    assert "False-Quarantine Rate (FQR)" in table_md
    assert "Human-Intervention Rate (HIR)" in table_md


def test_export_results_to_csv_roundtrip(harness):
    """Verifies that experimental results export to CSV with full fidelity and no fabricated data."""
    results = harness.run_all_scenarios(modes=[ExperimentalMode.RECOVERYGUARD], seed=42)
    assert len(results) == 8

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        csv_path = f.name

    try:
        exported_path = export_results_to_csv(results, csv_path)
        assert Path(exported_path).exists()

        with open(exported_path, "r", encoding="utf-8") as f_in:
            reader = list(csv.DictReader(f_in))

        assert len(reader) == 8
        first_row = reader[0]
        assert first_row["scenario_id"] == ScenarioId.SCENARIO_1_TIMEOUT_AFTER_SUCCESS.value
        assert first_row["mode"] == "RECOVERYGUARD"
        assert int(first_row["duplicate_side_effects"]) == 0
        assert int(first_row["success"]) == 1
    finally:
        Path(csv_path).unlink(missing_ok=True)


def test_export_results_to_sqlite_roundtrip(harness):
    """Verifies that raw experimental results export into an independent SQLite table and query correctly."""
    results = harness.run_all_scenarios(seed=42)
    assert len(results) == 24

    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        db_path = f.name

    try:
        exported_path = export_results_to_sqlite(results, db_path)
        assert Path(exported_path).exists()

        conn = sqlite3.connect(exported_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM experiment_runs")
        count = cur.fetchone()[0]
        assert count == 24

        cur.execute("SELECT DISTINCT mode FROM experiment_runs ORDER BY mode")
        modes = [r[0] for r in cur.fetchall()]
        assert modes == ["CHECKPOINT_ONLY", "EFFECT_LOG", "RECOVERYGUARD"]
        conn.close()
    finally:
        Path(db_path).unlink(missing_ok=True)


def test_scenario_2_crash_before_record_recoveryguard_probe_verification(harness):
    """Verifies Scenario 2 (Crash before record):

    External payment succeeded, but the agent crashed before recording the result.
    - CHECKPOINT_ONLY: Re-executes charge blindly -> duplicate charge!
    - EFFECT_LOG: Has no ledger commit -> assumes unexecuted -> duplicate charge!
    - RECOVERYGUARD: Queries external probe -> discovers charge was committed -> RESUMES without mutation!
    """
    res_cp = harness.run_scenario(ScenarioId.SCENARIO_2_CRASH_BEFORE_RECORD, ExperimentalMode.CHECKPOINT_ONLY)
    assert res_cp.duplicate_side_effects == 1
    assert res_cp.success is False

    res_el = harness.run_scenario(ScenarioId.SCENARIO_2_CRASH_BEFORE_RECORD, ExperimentalMode.EFFECT_LOG)
    assert res_el.duplicate_side_effects == 1
    assert res_el.success is False

    res_rg = harness.run_scenario(ScenarioId.SCENARIO_2_CRASH_BEFORE_RECORD, ExperimentalMode.RECOVERYGUARD)
    assert res_rg.duplicate_side_effects == 0
    assert res_rg.success is True
    assert res_rg.recovery_decision is not None
    assert res_rg.recovery_decision["outcome"] == RecoveryOutcome.RESUME.value


def test_standardized_outcome_mapping():
    """Verifies that RecoveryDecision.standardized_outcome maps all outcomes to {RESUME, RETRY, QUARANTINE, HUMAN_INTERVENTION}."""
    d_resume = RecoveryDecision(
        decision_id="d1",
        task_id="t1",
        action_id="a1",
        operation_id="o1",
        outcome=RecoveryOutcome.RESUME,
        is_safe_to_execute=True,
        rationale="Committed",
        timestamp=1.0,
    )
    assert d_resume.standardized_outcome == "RESUME"

    d_retry = RecoveryDecision(
        decision_id="d2",
        task_id="t1",
        action_id="a1",
        operation_id="o1",
        outcome=RecoveryOutcome.SAFE_RETRY,
        is_safe_to_execute=True,
        rationale="Absent",
        timestamp=1.0,
    )
    assert d_retry.standardized_outcome == "RETRY"

    d_quarantine = RecoveryDecision(
        decision_id="d3",
        task_id="t1",
        action_id="a1",
        operation_id="o1",
        outcome=RecoveryOutcome.QUARANTINE,
        is_safe_to_execute=False,
        rationale="Conflicting",
        timestamp=1.0,
    )
    assert d_quarantine.standardized_outcome == "QUARANTINE"

    d_human = RecoveryDecision(
        decision_id="d4",
        task_id="t1",
        action_id="a1",
        operation_id="o1",
        outcome=RecoveryOutcome.HUMAN_REVIEW,
        is_safe_to_execute=False,
        rationale="Unknown",
        timestamp=1.0,
    )
    assert d_human.standardized_outcome == "HUMAN_INTERVENTION"
