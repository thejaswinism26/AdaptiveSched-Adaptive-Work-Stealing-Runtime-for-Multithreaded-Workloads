"""Quantitative evaluation module for RecoveryGuard.

Calculates and reports reproducible comparative metrics across recovery modes:
- Duplicate side-effect rate
- Unsafe-resumption rate
- Unsafe-retry rate
- False-pause rate (legitimate safe actions blocked unnecessarily)
- Human-escalation rate
- Recovery success rate
- Mean and median recovery latency (ms)
- Number of experiments and failures per mode

Provides precise denominators, clear inapplicable-case treatment, and formatting
helpers for markdown tables and research reports.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sqlite3
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from recoveryguard.experiments import (
    ExperimentHarness,
    ExperimentResult,
    ExperimentalMode,
    SCENARIO_REGISTRY,
    ScenarioId,
    normalize_mode,
)


@dataclass(frozen=True)
class MetricDefinition:
    """Formal definition of an evaluation metric."""
    name: str
    symbol: str
    description: str
    numerator: str
    denominator: str
    treatment_of_inapplicable: str


METRIC_DEFINITIONS: Dict[str, MetricDefinition] = {
    "duplicate_side_effect_rate": MetricDefinition(
        name="Duplicate Side-Effect Rate",
        symbol="DSR",
        description="Fraction of experimental runs that produced duplicate external mutations.",
        numerator="Runs with duplicate_side_effects > 0",
        denominator="Total evaluated runs in mode (N)",
        treatment_of_inapplicable="Returns 0.0 when N = 0.",
    ),
    "unsafe_resumption_rate": MetricDefinition(
        name="Unsafe-Resumption Rate",
        symbol="URR",
        description="Fraction of runs where the agent resumed/skipped when external state was missing or contradictory.",
        numerator="Runs where agent resumed/skipped despite divergent external state",
        denominator="Total evaluated runs in mode (N)",
        treatment_of_inapplicable="Returns 0.0 when N = 0.",
    ),
    "unsafe_retry_rate": MetricDefinition(
        name="Unsafe-Retry Rate",
        symbol="UXR",
        description="Fraction of runs that re-executed a tool when external mutations already existed or evidence was unknown.",
        numerator="Runs attempting re-execution on already-committed or unverified risky mutations",
        denominator="Total evaluated runs in mode (N)",
        treatment_of_inapplicable="Returns 0.0 when N = 0.",
    ),
    "false_pause_rate": MetricDefinition(
        name="False-Pause Rate",
        symbol="FPR",
        description="Fraction of runs where legitimate, safe actions were blocked unnecessarily despite being safely resolvable.",
        numerator="Runs where action was blocked or escalated despite being demonstrably safe to resume/retry automatically",
        denominator="Total evaluated runs in mode (N)",
        treatment_of_inapplicable="Returns 0.0 when N = 0. Does NOT count correct safety blocks (e.g. truly ambiguous/stale states).",
    ),
    "human_escalation_rate": MetricDefinition(
        name="Human-Escalation Rate",
        symbol="HER",
        description="Fraction of runs requiring human operator review/intervention.",
        numerator="Runs with human_escalations > 0 or outcome HUMAN_REVIEW",
        denominator="Total evaluated runs in mode (N)",
        treatment_of_inapplicable="Returns 0.0 when N = 0.",
    ),
    "recovery_success_rate": MetricDefinition(
        name="Recovery Success Rate",
        symbol="RSR",
        description="Fraction of runs achieving optimal safe recovery (correct resumption, safe retry, or safe fail-closed block).",
        numerator="Runs flagged as success = True",
        denominator="Total evaluated runs in mode (N)",
        treatment_of_inapplicable="Returns 0.0 when N = 0.",
    ),
    "mean_latency_ms": MetricDefinition(
        name="Mean Recovery Latency (ms)",
        symbol="MLAT",
        description="Arithmetic mean of post-restore decision and execution latency in milliseconds.",
        numerator="Sum of execution_latency_ms across runs",
        denominator="Total evaluated runs in mode (N)",
        treatment_of_inapplicable="Returns 0.0 when N = 0. In-memory timing is subject to local OS scheduling jitter.",
    ),
    "median_latency_ms": MetricDefinition(
        name="Median Recovery Latency (ms)",
        symbol="MEDLAT",
        description="50th percentile (median) recovery latency in milliseconds.",
        numerator="Median of execution_latency_ms values",
        denominator="Total evaluated runs in mode (N)",
        treatment_of_inapplicable="Returns 0.0 when N = 0.",
    ),
    "false_quarantine_rate": MetricDefinition(
        name="False-Quarantine Rate",
        symbol="FQR",
        description="Fraction of runs where safe, valid actions were quarantined unnecessarily without genuine conflict or invariant violation.",
        numerator="Runs with outcome QUARANTINE despite no contradictory external state and no invariant violation",
        denominator="Total evaluated runs in mode (N)",
        treatment_of_inapplicable="Returns 0.0 when N = 0.",
    ),
    "human_intervention_rate": MetricDefinition(
        name="Human-Intervention Rate",
        symbol="HIR",
        description="Fraction of runs requiring human operator intervention.",
        numerator="Runs with human_escalations > 0 or outcome in (HUMAN_INTERVENTION, HUMAN_REVIEW)",
        denominator="Total evaluated runs in mode (N)",
        treatment_of_inapplicable="Returns 0.0 when N = 0.",
    ),
}


def classify_experiment_result(result: ExperimentResult) -> Dict[str, bool]:
    """Classifies an individual experiment result into fine-grained outcome categories.

    Distinguishes:
    - has_duplicate_side_effects: Result produced 1 or more duplicate external mutations.
    - is_unsafe_resumption: Agent resumed/skipped when external state was absent or in conflict.
    - is_unsafe_retry: Agent attempted tool execution when external effect already occurred or was unknown/risky.
    - is_false_pause: Action was blocked/paused despite being completely safe to resolve automatically.
    - is_correct_safety_block: Action was blocked because ground truth was ambiguous, stale, or contradictory.
    """
    sid = result.scenario_id
    mode = result.mode
    dups = result.duplicate_side_effects > 0
    blocked = result.blocked_actions > 0
    escalated = result.human_escalations > 0

    has_duplicate_side_effects = dups

    # Unsafe Resumption:
    # E.g. Scenario 6 in EFFECT_LOG where ledger says committed, but external state is absent,
    # and the agent blindly skipped/resumed without performing the mutation externally.
    is_unsafe_resumption = (
        sid == "scenario_6_ledger_external_conflict"
        and mode == "EFFECT_LOG"
        and result.actual_side_effects.get("captured_charges", 0) == 0
        and not blocked
    )

    # Unsafe Retry:
    # Execution attempted when external mutation had already occurred, causing duplicates or risking duplication.
    is_unsafe_retry = (
        has_duplicate_side_effects
        or (
            mode == "CHECKPOINT_ONLY"
            and sid in (
                "scenario_1_timeout_after_success",
                "scenario_2_crash_before_record",
                "scenario_4_probe_unavailable",
                "scenario_5_probe_stale",
                "scenario_7_older_checkpoint_restored",
                "scenario_8_no_status_query",
            )
        )
        or (
            mode == "EFFECT_LOG"
            and sid == "scenario_2_crash_before_record"
        )
    )

    # Correct Safety Block:
    # Truly ambiguous, unavailable, stale, or contradictory state where blocking is the ONLY safe behavior.
    is_correct_safety_block = (
        sid in (
            "scenario_4_probe_unavailable",
            "scenario_5_probe_stale",
            "scenario_6_ledger_external_conflict",
            "scenario_8_no_status_query",
        )
        and (blocked or escalated)
    )

    # False Pause:
    # Legitimate, safe actions blocked unnecessarily when safe automatic recovery was possible.
    # E.g. Scenario 1 in EFFECT_LOG (external charge succeeded; could have cleanly resumed via probe, but paused).
    is_false_pause = (
        (blocked or escalated)
        and not is_correct_safety_block
    )

    # False Quarantine: Quarantined when there was no conflicting evidence and no invariant violation
    outcome_str = ""
    if result.recovery_decision and "outcome" in result.recovery_decision:
        outcome_str = str(result.recovery_decision["outcome"])

    is_quarantined = "QUARANTINE" in outcome_str
    is_false_quarantine = (
        is_quarantined
        and result.invariant_violations == 0
        and sid != "scenario_6_ledger_external_conflict"
    )

    return {
        "has_duplicate_side_effects": has_duplicate_side_effects,
        "is_unsafe_resumption": is_unsafe_resumption,
        "is_unsafe_retry": is_unsafe_retry,
        "is_false_pause": is_false_pause,
        "is_correct_safety_block": is_correct_safety_block,
        "is_false_quarantine": is_false_quarantine,
    }


@dataclass
class ModeEvaluationSummary:
    """Quantitative evaluation summary for a specific recovery mode."""
    mode: str
    total_experiments: int
    total_failures: int
    duplicate_side_effect_rate: float
    unsafe_resumption_rate: float
    unsafe_retry_rate: float
    false_pause_rate: float
    human_escalation_rate: float
    recovery_success_rate: float
    mean_latency_ms: float
    median_latency_ms: float
    scenario_breakdown: List[Dict[str, Any]] = field(default_factory=list)
    false_quarantine_rate: float = 0.0
    human_intervention_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QuantitativeEvaluator:
    """Evaluates experimental results and computes standardized quantitative metrics."""

    def __init__(self) -> None:
        pass

    def evaluate_mode(
        self,
        results: List[ExperimentResult],
        mode: Optional[Union[str, ExperimentalMode]] = None,
    ) -> ModeEvaluationSummary:
        """Computes comprehensive quantitative metrics for a collection of scenario results."""
        target_mode_str = normalize_mode(mode).value if mode else (
            results[0].mode if results else "UNKNOWN"
        )
        filtered = [r for r in results if r.mode == target_mode_str] if mode else results

        n = len(filtered)
        if n == 0:
            return ModeEvaluationSummary(
                mode=target_mode_str,
                total_experiments=0,
                total_failures=0,
                duplicate_side_effect_rate=0.0,
                unsafe_resumption_rate=0.0,
                unsafe_retry_rate=0.0,
                false_pause_rate=0.0,
                human_escalation_rate=0.0,
                recovery_success_rate=0.0,
                mean_latency_ms=0.0,
                median_latency_ms=0.0,
                scenario_breakdown=[],
            )

        dup_count = 0
        unsafe_resump_count = 0
        unsafe_retry_count = 0
        false_pause_count = 0
        false_quarantine_count = 0
        human_esc_count = 0
        success_count = 0
        latencies: List[float] = []
        breakdown: List[Dict[str, Any]] = []

        for r in filtered:
            classification = classify_experiment_result(r)
            if classification["has_duplicate_side_effects"]:
                dup_count += 1
            if classification["is_unsafe_resumption"]:
                unsafe_resump_count += 1
            if classification["is_unsafe_retry"]:
                unsafe_retry_count += 1
            if classification["is_false_pause"]:
                false_pause_count += 1
            if classification.get("is_false_quarantine", False):
                false_quarantine_count += 1
            if r.human_escalations > 0:
                human_esc_count += 1
            if r.success:
                success_count += 1

            latencies.append(r.execution_latency_ms)

            breakdown.append({
                "scenario_id": r.scenario_id,
                "scenario_name": r.scenario_name,
                "success": r.success,
                "duplicate_side_effects": r.duplicate_side_effects,
                "invariant_violations": r.invariant_violations,
                "blocked_actions": r.blocked_actions,
                "human_escalations": r.human_escalations,
                "latency_ms": r.execution_latency_ms,
                "is_false_pause": classification["is_false_pause"],
                "is_unsafe_retry": classification["is_unsafe_retry"],
                "is_unsafe_resumption": classification["is_unsafe_resumption"],
                "is_false_quarantine": classification.get("is_false_quarantine", False),
                "details": r.details,
            })

        failures = n - success_count
        mean_lat = statistics.fmean(latencies) if latencies else 0.0
        med_lat = statistics.median(latencies) if latencies else 0.0

        return ModeEvaluationSummary(
            mode=target_mode_str,
            total_experiments=n,
            total_failures=failures,
            duplicate_side_effect_rate=round(dup_count / n, 4),
            unsafe_resumption_rate=round(unsafe_resump_count / n, 4),
            unsafe_retry_rate=round(unsafe_retry_count / n, 4),
            false_pause_rate=round(false_pause_count / n, 4),
            false_quarantine_rate=round(false_quarantine_count / n, 4),
            human_escalation_rate=round(human_esc_count / n, 4),
            human_intervention_rate=round(human_esc_count / n, 4),
            recovery_success_rate=round(success_count / n, 4),
            mean_latency_ms=round(mean_lat, 2),
            median_latency_ms=round(med_lat, 2),
            scenario_breakdown=breakdown,
        )

    def evaluate_suite(self, results: List[ExperimentResult]) -> Dict[str, ModeEvaluationSummary]:
        """Evaluates results grouped by recovery mode."""
        summaries: Dict[str, ModeEvaluationSummary] = {}
        for m in (ExperimentalMode.CHECKPOINT_ONLY, ExperimentalMode.EFFECT_LOG, ExperimentalMode.RECOVERYGUARD):
            summaries[m.value] = self.evaluate_mode(results, m)
        return summaries

    def run_and_evaluate(
        self,
        harness: ExperimentHarness,
        seed: int = 42,
    ) -> Tuple[List[ExperimentResult], Dict[str, ModeEvaluationSummary]]:
        """Executes all 8 scenarios across all 3 modes and returns results with evaluation summaries."""
        results = harness.run_all_scenarios(seed=seed)
        summaries = self.evaluate_suite(results)
        return results, summaries

    def format_comparison_table(self, summaries: Dict[str, ModeEvaluationSummary]) -> str:
        """Generates a GitHub-flavored Markdown comparison table across the three modes."""
        header = (
            "| Metric | Mode A: CHECKPOINT_ONLY | Mode B: EFFECT_LOG | Mode C: RECOVERYGUARD |\n"
            "| :--- | :--- | :--- | :--- |\n"
        )
        m_cp = summaries.get(ExperimentalMode.CHECKPOINT_ONLY.value)
        m_el = summaries.get(ExperimentalMode.EFFECT_LOG.value)
        m_rg = summaries.get(ExperimentalMode.RECOVERYGUARD.value)

        def pct(val: Optional[float]) -> str:
            return f"{(val * 100):.1f}%" if val is not None else "N/A"

        def num(val: Optional[Any]) -> str:
            return str(val) if val is not None else "N/A"

        rows = [
            f"| **Total Experiments (N)** | {num(m_cp.total_experiments if m_cp else None)} | {num(m_el.total_experiments if m_el else None)} | {num(m_rg.total_experiments if m_rg else None)} |",
            f"| **Failures Count** | {num(m_cp.total_failures if m_cp else None)} | {num(m_el.total_failures if m_el else None)} | {num(m_rg.total_failures if m_rg else None)} |",
            f"| **Recovery Success Rate (RSR)** | {pct(m_cp.recovery_success_rate if m_cp else None)} | {pct(m_el.recovery_success_rate if m_el else None)} | {pct(m_rg.recovery_success_rate if m_rg else None)} |",
            f"| **Duplicate Side-Effect Rate (DSR)** | {pct(m_cp.duplicate_side_effect_rate if m_cp else None)} | {pct(m_el.duplicate_side_effect_rate if m_el else None)} | {pct(m_rg.duplicate_side_effect_rate if m_rg else None)} |",
            f"| **Unsafe-Resumption Rate (URR)** | {pct(m_cp.unsafe_resumption_rate if m_cp else None)} | {pct(m_el.unsafe_resumption_rate if m_el else None)} | {pct(m_rg.unsafe_resumption_rate if m_rg else None)} |",
            f"| **Unsafe-Retry Rate (UXR)** | {pct(m_cp.unsafe_retry_rate if m_cp else None)} | {pct(m_el.unsafe_retry_rate if m_el else None)} | {pct(m_rg.unsafe_retry_rate if m_rg else None)} |",
            f"| **False-Pause Rate (FPR)** | {pct(m_cp.false_pause_rate if m_cp else None)} | {pct(m_el.false_pause_rate if m_el else None)} | {pct(m_rg.false_pause_rate if m_rg else None)} |",
            f"| **False-Quarantine Rate (FQR)** | {pct(m_cp.false_quarantine_rate if m_cp else None)} | {pct(m_el.false_quarantine_rate if m_el else None)} | {pct(m_rg.false_quarantine_rate if m_rg else None)} |",
            f"| **Human-Intervention Rate (HIR)** | {pct(m_cp.human_intervention_rate if m_cp else None)} | {pct(m_el.human_intervention_rate if m_el else None)} | {pct(m_rg.human_intervention_rate if m_rg else None)} |",
            f"| **Mean Recovery Latency** | {m_cp.mean_latency_ms if m_cp else 0.0:.2f} ms | {m_el.mean_latency_ms if m_el else 0.0:.2f} ms | {m_rg.mean_latency_ms if m_rg else 0.0:.2f} ms |",
            f"| **Median Recovery Latency** | {m_cp.median_latency_ms if m_cp else 0.0:.2f} ms | {m_el.median_latency_ms if m_el else 0.0:.2f} ms | {m_rg.median_latency_ms if m_rg else 0.0:.2f} ms |",
        ]
        return header + "\n".join(rows) + "\n"

    def format_scenario_table(self, results: List[ExperimentResult]) -> str:
        """Generates a detailed per-scenario breakdown table comparing all 3 modes."""
        header = (
            "| Scenario | Mode | Outcome Status | Duplicates | Blocked | Escalated | Latency (ms) | Success |\n"
            "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n"
        )
        rows = []
        for r in results:
            succ = "PASS" if r.success else "FAIL"
            outcome = r.recovery_decision["outcome"] if r.recovery_decision else (
                "BLOCKED_UNKNOWN" if r.blocked_actions > 0 else "EXECUTED"
            )
            rows.append(
                f"| `{r.scenario_id}` | `{r.mode}` | `{outcome}` | {r.duplicate_side_effects} | {r.blocked_actions} | {r.human_escalations} | {r.execution_latency_ms:.2f} | **{succ}** |"
            )
        return header + "\n".join(rows) + "\n"

    def evaluate_ablation_study(
        self,
        ablation_results: Dict[str, List[ExperimentResult]],
    ) -> Dict[str, ModeEvaluationSummary]:
        """Evaluates results across the 4 ablation variants."""
        summaries: Dict[str, ModeEvaluationSummary] = {}
        for variant_key, results in ablation_results.items():
            summaries[variant_key] = self.evaluate_mode(results)
        return summaries

    def format_ablation_table(
        self,
        ablation_summaries: Dict[str, ModeEvaluationSummary],
    ) -> str:
        """Generates a Markdown table comparing Full RecoveryGuard with its ablation variants."""
        header = (
            "| Metric | Full RecoveryGuard | No Ledger | No Probes | No Invariants |\n"
            "| :--- | :--- | :--- | :--- | :--- |\n"
        )
        s_full = ablation_summaries.get("FULL_RECOVERYGUARD")
        s_nole = ablation_summaries.get("NO_LEDGER")
        s_nopr = ablation_summaries.get("NO_PROBES")
        s_noin = ablation_summaries.get("NO_INVARIANTS")

        def pct(val: Optional[float]) -> str:
            return f"{(val * 100):.1f}%" if val is not None else "N/A"

        def num(val: Optional[Any]) -> str:
            return str(val) if val is not None else "N/A"

        rows = [
            f"| **Recovery Success Rate (RSR)** | {pct(s_full.recovery_success_rate if s_full else None)} | {pct(s_nole.recovery_success_rate if s_nole else None)} | {pct(s_nopr.recovery_success_rate if s_nopr else None)} | {pct(s_noin.recovery_success_rate if s_noin else None)} |",
            f"| **Duplicate Side-Effect Rate (DSR)** | {pct(s_full.duplicate_side_effect_rate if s_full else None)} | {pct(s_nole.duplicate_side_effect_rate if s_nole else None)} | {pct(s_nopr.duplicate_side_effect_rate if s_nopr else None)} | {pct(s_noin.duplicate_side_effect_rate if s_noin else None)} |",
            f"| **False-Pause Rate (FPR)** | {pct(s_full.false_pause_rate if s_full else None)} | {pct(s_nole.false_pause_rate if s_nole else None)} | {pct(s_nopr.false_pause_rate if s_nopr else None)} | {pct(s_noin.false_pause_rate if s_noin else None)} |",
            f"| **False-Quarantine Rate (FQR)** | {pct(s_full.false_quarantine_rate if s_full else None)} | {pct(s_nole.false_quarantine_rate if s_nole else None)} | {pct(s_nopr.false_quarantine_rate if s_nopr else None)} | {pct(s_noin.false_quarantine_rate if s_noin else None)} |",
            f"| **Human-Intervention Rate (HIR)** | {pct(s_full.human_intervention_rate if s_full else None)} | {pct(s_nole.human_intervention_rate if s_nole else None)} | {pct(s_nopr.human_intervention_rate if s_nopr else None)} | {pct(s_noin.human_intervention_rate if s_noin else None)} |",
            f"| **Median Latency (ms)** | {s_full.median_latency_ms if s_full else 0.0:.2f} ms | {s_nole.median_latency_ms if s_nole else 0.0:.2f} ms | {s_nopr.median_latency_ms if s_nopr else 0.0:.2f} ms | {s_noin.median_latency_ms if s_noin else 0.0:.2f} ms |",
        ]
        return header + "\n".join(rows) + "\n"

    def generate_full_report(
        self,
        results: List[ExperimentResult],
        summaries: Dict[str, ModeEvaluationSummary],
    ) -> str:
        """Produces a comprehensive Markdown evaluation report."""
        comp_table = self.format_comparison_table(summaries)
        scen_table = self.format_scenario_table(results)

        report = f"""# RecoveryGuard: Quantitative Evaluation Report

## 1. Summary Comparison

{comp_table}

## 2. Per-Scenario Experimental Results

{scen_table}

## 3. Metric Definitions & Denominators

| Metric Name | Symbol | Numerator | Denominator | Inapplicable Case Handling |
| :--- | :--- | :--- | :--- | :--- |
"""
        for m in METRIC_DEFINITIONS.values():
            report += f"| **{m.name}** | `{m.symbol}` | {m.numerator} | {m.denominator} | {m.treatment_of_inapplicable} |\n"

        return report


# ---------------------------------------------------------------------------
# Data Persistence Exporters (CSV & SQLite)
# ---------------------------------------------------------------------------
def export_results_to_csv(
    results: List[ExperimentResult],
    filepath: Union[str, Path],
) -> str:
    """Exports raw experimental result records to a CSV file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario_id",
            "scenario_name",
            "mode",
            "injected_fault",
            "outcome",
            "duplicate_side_effects",
            "invariant_violations",
            "blocked_actions",
            "human_escalations",
            "execution_latency_ms",
            "success",
            "details",
        ])
        for r in results:
            outcome = ""
            if r.recovery_decision and "outcome" in r.recovery_decision:
                outcome = str(r.recovery_decision["outcome"])
            writer.writerow([
                r.scenario_id,
                r.scenario_name,
                r.mode,
                r.injected_fault,
                outcome,
                r.duplicate_side_effects,
                r.invariant_violations,
                r.blocked_actions,
                r.human_escalations,
                r.execution_latency_ms,
                1 if r.success else 0,
                r.details,
            ])
    return str(path.resolve())


def export_results_to_sqlite(
    results: List[ExperimentResult],
    db_path: Union[str, Path],
) -> str:
    """Saves raw experimental execution results to an independent SQLite table."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scenario_id TEXT NOT NULL,
                scenario_name TEXT NOT NULL,
                mode TEXT NOT NULL,
                injected_fault TEXT NOT NULL,
                outcome TEXT,
                duplicate_side_effects INTEGER NOT NULL,
                invariant_violations INTEGER NOT NULL,
                blocked_actions INTEGER NOT NULL,
                human_escalations INTEGER NOT NULL,
                execution_latency_ms REAL NOT NULL,
                success INTEGER NOT NULL,
                details TEXT,
                recorded_at REAL NOT NULL
            )
            """
        )
        now = time.time()
        for r in results:
            outcome = ""
            if r.recovery_decision and "outcome" in r.recovery_decision:
                outcome = str(r.recovery_decision["outcome"])
            cur.execute(
                """
                INSERT INTO experiment_runs (
                    scenario_id, scenario_name, mode, injected_fault, outcome,
                    duplicate_side_effects, invariant_violations, blocked_actions,
                    human_escalations, execution_latency_ms, success, details, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r.scenario_id,
                    r.scenario_name,
                    r.mode,
                    r.injected_fault,
                    outcome,
                    r.duplicate_side_effects,
                    r.invariant_violations,
                    r.blocked_actions,
                    r.human_escalations,
                    r.execution_latency_ms,
                    1 if r.success else 0,
                    r.details,
                    now,
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return str(path.resolve())


def export_summaries_to_csv(
    summaries: Dict[str, ModeEvaluationSummary],
    filepath: Union[str, Path],
) -> str:
    """Exports quantitative summary metrics across modes to a CSV file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "mode",
            "total_experiments",
            "total_failures",
            "recovery_success_rate",
            "duplicate_side_effect_rate",
            "unsafe_resumption_rate",
            "unsafe_retry_rate",
            "false_pause_rate",
            "false_quarantine_rate",
            "human_intervention_rate",
            "mean_latency_ms",
            "median_latency_ms",
        ])
        for s in summaries.values():
            writer.writerow([
                s.mode,
                s.total_experiments,
                s.total_failures,
                s.recovery_success_rate,
                s.duplicate_side_effect_rate,
                s.unsafe_resumption_rate,
                s.unsafe_retry_rate,
                s.false_pause_rate,
                s.false_quarantine_rate,
                s.human_intervention_rate,
                s.mean_latency_ms,
                s.median_latency_ms,
            ])
    return str(path.resolve())


def evaluate_ablation_study(
    ablation_results: Dict[str, List[ExperimentResult]],
) -> Dict[str, ModeEvaluationSummary]:
    """Module-level convenience wrapper for QuantitativeEvaluator.evaluate_ablation_study."""
    return QuantitativeEvaluator().evaluate_ablation_study(ablation_results)


def format_ablation_table(
    ablation_summaries: Dict[str, ModeEvaluationSummary],
) -> str:
    """Module-level convenience wrapper for QuantitativeEvaluator.format_ablation_table."""
    return QuantitativeEvaluator().format_ablation_table(ablation_summaries)
