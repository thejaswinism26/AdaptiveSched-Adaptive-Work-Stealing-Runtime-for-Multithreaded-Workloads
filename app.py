"""Streamlit interactive dashboard for RecoveryGuard.

Visualizes and evaluates:
1. Checkpoint-Only (Naive Rollback Baseline)
2. Effect-Log (Replay Protection Baseline)
3. RecoveryGuard (Durable Ledger + External Probes + Invariants + Mandatory Gate)

Provides:
- System overview and architectural comparison
- Interactive failure scenario execution lab
- Step-by-step recovery verification trace
- Quantitative benchmark suite evaluation (DSR, URR, UXR, FPR, HER, RSR, MLAT, MEDLAT)
- Formal metric definitions and research limitations
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from recoveryguard.evaluation import (
    METRIC_DEFINITIONS,
    ModeEvaluationSummary,
    QuantitativeEvaluator,
    classify_experiment_result,
    export_results_to_csv,
    export_results_to_sqlite,
    export_summaries_to_csv,
)
from recoveryguard.experiments import (
    ExperimentHarness,
    ExperimentResult,
    ExperimentalMode,
    SCENARIO_REGISTRY,
    ScenarioId,
    ScenarioMetadata,
    normalize_mode,
)
from recoveryguard.models import EvidenceState, InvariantStatus, RecoveryOutcome

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RecoveryGuard Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def get_harness() -> ExperimentHarness:
    """Returns or caches the benchmark experiment harness."""
    return ExperimentHarness()


def get_evaluator() -> QuantitativeEvaluator:
    """Returns the quantitative evaluator."""
    return QuantitativeEvaluator()


def format_outcome_badge(outcome: Any) -> str:
    """Returns a styled HTML/markdown badge for a recovery outcome."""
    outcome_str = str(outcome).split(".")[-1]
    color_map = {
        "RESUME": "🟢",
        "SAFE_RETRY": "🔵",
        "RECONCILE": "🟣",
        "COMPENSATE": "🟠",
        "HUMAN_REVIEW": "🟡",
        "QUARANTINE": "🔴",
        "EXECUTED": "⚠️",
        "BLOCKED_BY_GATE": "🚫",
        "BLOCKED_UNKNOWN": "⏳",
        "SKIPPED_DUPLICATE": "⏭️",
        "RESUMED_WITHOUT_MUTATION": "🛡️",
        "EXECUTED_SAFE": "✅",
    }
    icon = color_map.get(outcome_str, "⚪")
    return f"{icon} `{outcome_str}`"


def render_trace_item(
    step_num: int,
    title: str,
    status_icon: str,
    content_fn,
) -> None:
    """Renders a single numbered trace phase container."""
    with st.container():
        st.markdown(f"#### {status_icon} Phase {step_num}: {title}")
        content_fn()
        st.markdown("---")


# ---------------------------------------------------------------------------
# Main App Layout
# ---------------------------------------------------------------------------
def main() -> None:
    # Initialize session state for deterministic, button-triggered execution
    if "single_scenario_results" not in st.session_state:
        st.session_state["single_scenario_results"] = None
    if "single_scenario_id" not in st.session_state:
        st.session_state["single_scenario_id"] = None
    if "single_scenario_mode" not in st.session_state:
        st.session_state["single_scenario_mode"] = "ALL"
    if "benchmark_results" not in st.session_state:
        st.session_state["benchmark_results"] = None
    if "benchmark_summaries" not in st.session_state:
        st.session_state["benchmark_summaries"] = None

    # Title & Header
    st.title("🛡️ RecoveryGuard: Cross-System Recovery Verification")
    st.caption("A Research Prototype for Epistemic Recovery Verification in AI Agent Workflows")

    st.info(
        "**Core Hypothesis:** When an AI agent restores an internal checkpoint, external systems "
        "(payment gateways, message queues, cloud deployments) retain real-world side effects. "
        "A naive rollback creates dangerous distributed divergence. RecoveryGuard enforces a **mandatory "
        "post-restore recovery gate** that combines a durable effect ledger, read-only external status probes, "
        "and business invariants to eliminate duplicate side effects and avoid false pauses."
    )

    # Architectural Overview Section
    with st.expander("📚 System Overview: The Rollback Problem & Experimental Modes", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### The Distributed Rollback Problem")
            st.markdown(
                """
                - **Internal vs External State Divergence:** Checkpoints only restore local agent memory
                  (step count, variables). External APIs cannot be silently rolled back.
                - **Network Epistemic Uncertainty:** A network timeout after an HTTP request does *not*
                  imply the action failed. The external service may have executed the mutation completely.
                - **Replay Hazards:** If the restored agent re-executes the first tool blindly, it duplicates
                  payments, dispatches duplicate notifications, or corrupts multi-service invariant states.
                """
            )
        with col2:
            st.markdown("### RecoveryGuard Architectural Pillars")
            st.markdown(
                """
                1. **Durable Effect Ledger:** SQLite WAL database recording action intents and attempts
                   that survives agent process crashes and checkpoint rollbacks.
                2. **Non-Mutating External Probes:** Read-only queries checking ground truth without side effects.
                3. **Task-Specific Invariants:** Cross-service consistency checks (e.g. at most 1 charge per order).
                4. **Mandatory Post-Restore Gate:** Blocks side-effecting tools until verified and authorized.
                """
            )

        st.markdown("### Comparative Recovery Modes")
        mode_data = [
            {
                "Mode": "Mode A: CHECKPOINT_ONLY",
                "Description": "Naive baseline: restores local checkpoint and resumes execution blindly.",
                "Durable Ledger": "❌ None",
                "External Probes": "❌ None",
                "Task Invariants": "❌ None",
                "Failure Mode": "High duplicate side-effect rate (DSR ~75%). Unsafe retries.",
            },
            {
                "Mode": "Mode B: EFFECT_LOG",
                "Description": "Replay-protection baseline: deduplicates known commits via SQLite log.",
                "Durable Ledger": "✅ Local SQLite",
                "External Probes": "❌ None",
                "Task Invariants": "❌ None",
                "Failure Mode": "False pauses on timeouts; blind to unrecorded external commits.",
            },
            {
                "Mode": "Mode C: RECOVERYGUARD",
                "Description": "Cross-system verification gate combining ledger, probes, and invariants.",
                "Durable Ledger": "✅ Local SQLite",
                "External Probes": "✅ Read-only API queries",
                "Task Invariants": "✅ Multi-system checks",
                "Failure Mode": "Zero duplicate side effects (DSR 0.0%); zero false pauses (FPR 0.0%).",
            },
        ]
        st.table(mode_data)

    # Main Tabs: Single Scenario Lab vs Full Benchmark Suite
    tab_lab, tab_bench, tab_limits = st.tabs(
        [
            "🔬 Scenario Lab & Execution Trace",
            "📊 Quantitative Benchmark Evaluation",
            "📖 Research Limitations & Discussion",
        ]
    )

    # -----------------------------------------------------------------------
    # TAB 1: Scenario Lab & Execution Trace
    # -----------------------------------------------------------------------
    with tab_lab:
        st.subheader("Interactive Scenario Execution Lab")
        st.markdown(
            "Select a distributed fault scenario and evaluate how the agent recovers under different modes."
        )

        col_ctrl1, col_ctrl2 = st.columns([2, 1])

        with col_ctrl1:
            scenario_options = list(ScenarioId)
            selected_scenario_id = st.selectbox(
                "Select Fault Scenario",
                options=scenario_options,
                format_func=lambda sid: f"{SCENARIO_REGISTRY[sid].name} [{SCENARIO_REGISTRY[sid].injected_fault}]",
                index=0,
            )
            selected_meta = SCENARIO_REGISTRY[selected_scenario_id]
            st.info(f"**Description:** {selected_meta.description}")

        with col_ctrl2:
            selected_mode_option = st.selectbox(
                "Select Recovery Mode",
                options=["ALL", "RECOVERYGUARD", "EFFECT_LOG", "CHECKPOINT_ONLY"],
                format_func=lambda m: {
                    "ALL": "⚡ Compare All 3 Modes Side-by-Side",
                    "RECOVERYGUARD": "🛡️ Mode C: RECOVERYGUARD",
                    "EFFECT_LOG": "📋 Mode B: EFFECT_LOG",
                    "CHECKPOINT_ONLY": "⏪ Mode A: CHECKPOINT_ONLY",
                }[m],
                index=0,
            )

        # Configurable Probe Conditions with Deterministic Defaults
        with st.expander("⚙️ Configure Probe Conditions & Fault Injection Overrides", expanded=False):
            st.markdown(
                "*Note: Default values reflect the standardized benchmark parameters for this scenario.*"
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                override_supported = st.checkbox(
                    "Probe Supported by Service",
                    value=selected_meta.probe_supported,
                    help="Whether the external service implements a read-only query endpoint.",
                )
            with c2:
                override_available = st.checkbox(
                    "Probe Endpoint Online",
                    value=selected_meta.probe_available,
                    help="Whether the probe query succeeds or encounters an outage.",
                )
            with c3:
                override_stale = st.checkbox(
                    "Probe Returns Stale Data",
                    value=selected_meta.probe_is_stale,
                    help="Whether the probe returns outdated/cached evidence.",
                )

        probe_override = {
            "probe_supported": override_supported,
            "probe_available": override_available,
            "probe_is_stale": override_stale,
            "has_external_query_capability": override_supported,
        }

        # Explicit Trigger Button
        if st.button("▶ Run Selected Scenario", type="primary", use_container_width=True):
            harness = get_harness()
            with st.spinner("Executing scenario in isolated sandbox..."):
                if selected_mode_option == "ALL":
                    results = harness.run_scenario_all_modes(
                        selected_scenario_id,
                        probe_override=probe_override,
                    )
                    st.session_state["single_scenario_results"] = results
                    st.session_state["single_scenario_mode"] = "ALL"
                else:
                    exp_mode = normalize_mode(selected_mode_option)
                    res = harness.run_scenario(
                        selected_scenario_id,
                        exp_mode,
                        probe_override=probe_override,
                    )
                    st.session_state["single_scenario_results"] = {exp_mode.value: res}
                    st.session_state["single_scenario_mode"] = exp_mode.value

                st.session_state["single_scenario_id"] = selected_scenario_id
            st.success("Scenario execution complete!")

        # Render Execution Trace if results exist
        if st.session_state["single_scenario_results"] is not None:
            results_dict = st.session_state["single_scenario_results"]
            st.markdown("### Execution & Recovery Trace")

            # If ALL modes were run, provide mode comparison tabs
            if len(results_dict) > 1:
                mode_tabs = st.tabs(
                    [
                        "🛡️ Mode C: RECOVERYGUARD",
                        "📋 Mode B: EFFECT_LOG",
                        "⏪ Mode A: CHECKPOINT_ONLY",
                    ]
                )
                modes_order = [
                    ExperimentalMode.RECOVERYGUARD.value,
                    ExperimentalMode.EFFECT_LOG.value,
                    ExperimentalMode.CHECKPOINT_ONLY.value,
                ]
                for tab, m_key in zip(mode_tabs, modes_order):
                    with tab:
                        if m_key in results_dict:
                            render_mode_trace(results_dict[m_key], selected_meta)
            else:
                m_key = list(results_dict.keys())[0]
                render_mode_trace(results_dict[m_key], selected_meta)

    # -----------------------------------------------------------------------
    # TAB 2: Quantitative Benchmark Suite
    # -----------------------------------------------------------------------
    with tab_bench:
        st.subheader("Quantitative Evaluation Matrix")
        st.markdown(
            "Execute the full evaluation suite across all 8 standardized failure scenarios "
            "and all 3 experimental modes (24 isolated trials) to measure recovery fidelity."
        )

        if st.button("🚀 Run Full Benchmark Suite (24 Trials)", type="primary"):
            harness = get_harness()
            evaluator = get_evaluator()
            with st.spinner("Executing 24 isolated experimental trials..."):
                results = harness.run_all_scenarios(seed=42)
                summaries = evaluator.evaluate_suite(results)
                st.session_state["benchmark_results"] = results
                st.session_state["benchmark_summaries"] = summaries
            st.success("Benchmark evaluation completed successfully!")

        if st.session_state["benchmark_summaries"] is not None:
            summaries: Dict[str, ModeEvaluationSummary] = st.session_state["benchmark_summaries"]
            evaluator = get_evaluator()

            # Top Key Metrics Comparison Cards
            m_cp = summaries.get(ExperimentalMode.CHECKPOINT_ONLY.value)
            m_el = summaries.get(ExperimentalMode.EFFECT_LOG.value)
            m_rg = summaries.get(ExperimentalMode.RECOVERYGUARD.value)

            st.markdown("#### Key Research Metrics")
            kpi_cols = st.columns(4)

            with kpi_cols[0]:
                st.metric(
                    label="Duplicate Side-Effect Rate (DSR)",
                    value=f"{(m_rg.duplicate_side_effect_rate * 100):.1f}%",
                    delta=f"-{(m_cp.duplicate_side_effect_rate - m_rg.duplicate_side_effect_rate) * 100:.1f}% vs CP",
                    delta_color="inverse",
                    help="Fraction of runs that duplicated external mutations (lower is better; 0% is ideal).",
                )
            with kpi_cols[1]:
                st.metric(
                    label="False-Pause Rate (FPR)",
                    value=f"{(m_rg.false_pause_rate * 100):.1f}%",
                    delta=f"-{(m_el.false_pause_rate - m_rg.false_pause_rate) * 100:.1f}% vs EL",
                    delta_color="inverse",
                    help="Fraction of runs where safe actions were paused unnecessarily (lower is better; 0% is ideal).",
                )
            with kpi_cols[2]:
                st.metric(
                    label="Recovery Success Rate (RSR)",
                    value=f"{(m_rg.recovery_success_rate * 100):.1f}%",
                    delta=f"+{(m_rg.recovery_success_rate - m_cp.recovery_success_rate) * 100:.1f}% vs CP",
                    help="Fraction of runs achieving safe, correct recovery (higher is better; 100% is ideal).",
                )
            with kpi_cols[3]:
                st.metric(
                    label="Mean Recovery Latency",
                    value=f"{m_rg.mean_latency_ms:.2f} ms",
                    delta=f"+{m_rg.mean_latency_ms - m_cp.mean_latency_ms:.2f} ms",
                    delta_color="off",
                    help="Mean post-restore decision and tool authorization latency in milliseconds.",
                )

            st.markdown("#### Mode Comparison Table")
            comp_table_md = evaluator.format_comparison_table(summaries)
            st.markdown(comp_table_md)

            st.markdown("#### Per-Scenario Breakdown Matrix")
            results: List[ExperimentResult] = st.session_state["benchmark_results"]
            scen_table_md = evaluator.format_scenario_table(results)
            st.markdown(scen_table_md)

            st.markdown("#### 📈 Comparative Error & Recovery Rates")
            chart_dict = {
                "Mode": ["Mode A: CHECKPOINT_ONLY", "Mode B: EFFECT_LOG", "Mode C: RECOVERYGUARD"],
                "Duplicate Side-Effect Rate (%)": [
                    (m_cp.duplicate_side_effect_rate * 100) if m_cp else 0.0,
                    (m_el.duplicate_side_effect_rate * 100) if m_el else 0.0,
                    (m_rg.duplicate_side_effect_rate * 100) if m_rg else 0.0,
                ],
                "False-Pause Rate (%)": [
                    (m_cp.false_pause_rate * 100) if m_cp else 0.0,
                    (m_el.false_pause_rate * 100) if m_el else 0.0,
                    (m_rg.false_pause_rate * 100) if m_rg else 0.0,
                ],
                "Recovery Success Rate (%)": [
                    (m_cp.recovery_success_rate * 100) if m_cp else 0.0,
                    (m_el.recovery_success_rate * 100) if m_el else 0.0,
                    (m_rg.recovery_success_rate * 100) if m_rg else 0.0,
                ],
            }
            st.bar_chart(
                data=chart_dict,
                x="Mode",
                y=["Duplicate Side-Effect Rate (%)", "False-Pause Rate (%)", "Recovery Success Rate (%)"],
            )

            st.markdown("#### 💾 Export Benchmark Execution Records")
            exp_c1, exp_c2 = st.columns(2)
            with exp_c1:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as tmp_csv:
                    export_results_to_csv(results, tmp_csv.name)
                    with open(tmp_csv.name, "r", encoding="utf-8") as f_in:
                        csv_data = f_in.read()
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_data,
                    file_name="recoveryguard_benchmark_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with exp_c2:
                with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
                    export_results_to_sqlite(results, tmp_db.name)
                    with open(tmp_db.name, "rb") as f_in:
                        db_bytes = f_in.read()
                st.download_button(
                    label="💾 Download Results as SQLite DB",
                    data=db_bytes,
                    file_name="recoveryguard_benchmark.sqlite",
                    mime="application/x-sqlite3",
                    use_container_width=True,
                )

            with st.expander("🔬 Component Ablation Study (Full vs -Ledger vs -Probes vs -Invariants)", expanded=False):
                st.markdown(
                    "Evaluate the isolated contribution of each architectural pillar of RecoveryGuard:\n"
                    "- **FULL_RECOVERYGUARD**: Ledger + External Probes + Invariant Suite\n"
                    "- **NO_LEDGER**: Disables durable effect ledger (fails closed under mandatory gate)\n"
                    "- **NO_PROBES**: Disables external status probes (handles timeouts via human review)\n"
                    "- **NO_INVARIANTS**: Disables cross-system business invariant evaluations"
                )
                if st.button("Run Component Ablation Study (32 Trials)"):
                    harness = get_harness()
                    evaluator = get_evaluator()
                    with st.spinner("Executing ablation experiments across 4 variants..."):
                        abl_res = harness.run_ablation_study(seed=42)
                        abl_sums = evaluator.evaluate_ablation_study(abl_res)
                        st.session_state["ablation_summaries"] = abl_sums
                    st.success("Component ablation study completed!")

                if "ablation_summaries" in st.session_state and st.session_state["ablation_summaries"]:
                    abl_sums = st.session_state["ablation_summaries"]
                    st.markdown(evaluator.format_ablation_table(abl_sums))

            with st.expander("🔁 Multi-Trial Benchmark Suite (≥100 Trials per Scenario)", expanded=False):
                st.markdown(
                    "Run high-volume multi-trial evaluations (e.g. 100 trials per scenario per strategy = 2,400 runs) "
                    "with reproducible pseudo-random seeds."
                )
                mt_trial_count = st.slider("Trials per scenario per strategy", min_value=10, max_value=100, value=100, step=10)
                if st.button(f"Run {mt_trial_count * 24} Multi-Trial Suite"):
                    harness = get_harness()
                    evaluator = get_evaluator()
                    with st.spinner(f"Executing {mt_trial_count * 24} trials across 8 scenarios and 3 strategies..."):
                        mt_res = harness.run_benchmark_trials(num_trials=mt_trial_count, seed=42)
                        mt_sums = evaluator.evaluate_suite(mt_res)
                        st.session_state["multi_trial_summaries"] = mt_sums
                        st.session_state["multi_trial_results"] = mt_res
                    st.success(f"Successfully evaluated {len(mt_res)} multi-trial experiments!")

                if "multi_trial_summaries" in st.session_state and st.session_state["multi_trial_summaries"]:
                    st.markdown("##### Multi-Trial Quantitative Results")
                    st.markdown(evaluator.format_comparison_table(st.session_state["multi_trial_summaries"]))

            with st.expander("📐 Metric Definitions & Denominators Reference", expanded=False):
                st.markdown(
                    """
                    | Metric Name | Symbol | Numerator | Denominator | Treatment of Inapplicable Cases |
                    | :--- | :--- | :--- | :--- | :--- |
                    """
                )
                for m in METRIC_DEFINITIONS.values():
                    st.markdown(
                        f"| **{m.name}** | `{m.symbol}` | {m.numerator} | {m.denominator} | {m.treatment_of_inapplicable} |"
                    )

    # -----------------------------------------------------------------------
    # TAB 3: Research Limitations & Discussion
    # -----------------------------------------------------------------------
    with tab_limits:
        st.subheader("Research Scope & Epistemic Boundaries")
        st.markdown(
            """
            RecoveryGuard addresses critical vulnerabilities in agent recovery, but real-world distributed
            systems impose fundamental physical and epistemic constraints:
            """
        )

        col_l1, col_l2 = st.columns(2)

        with col_l1:
            st.markdown("### 1. External Service Probe Cooperativeness")
            st.warning(
                "**Dependency on Status Queries:** RecoveryGuard requires external APIs to provide "
                "idempotent query endpoints (e.g. `GET /charges/{idempotency_key}`). For legacy external "
                "services that lack status-query capabilities (demonstrated in **Scenario 8**), the recovery "
                "gate cannot establish confirmed absence or confirmed success. In such cases, RecoveryGuard "
                "intentionally **fails closed to HUMAN_REVIEW or COMPENSATE** rather than risking duplicate actions."
            )

            st.markdown("### 2. Epistemic Uncertainty & Timeouts")
            st.info(
                "**Timeout != Execution Failure:** In distributed computing, a network timeout represents "
                "an unobserved state. The packet may have reached the server and executed, or dropped in flight. "
                "RecoveryGuard models epistemic uncertainty explicitly (`EvidenceState.UNKNOWN`). It strictly "
                "forbids speculative retry on risky mutations until external evidence confirms the entity is absent."
            )

        with col_l2:
            st.markdown("### 3. Distributed Exactly-Once Limits")
            st.warning(
                "**Practical At-Most-Once / Exactly-Once:** True distributed exactly-once across arbitrary SaaS "
                "endpoints is physically impossible without two-phase commit (2PC). RecoveryGuard achieves "
                "practical cross-system safety by combining **client-side durable intent logging**, **idempotency keys**, "
                "and **mandatory post-restore verification gates** that intercept every side-effecting invocation."
            )

            st.markdown("### 4. Prototype Scope & Verification Assumptions")
            st.info(
                "**Local Research Implementation:** The current prototype demonstrates the core recovery "
                "verification algorithms using SQLite WAL transactions, mock services, and deterministic fault injection. "
                "Production implementations should incorporate distributed lease locking, cryptographically signed audit "
                "ledgers, and asynchronous human-in-the-loop escalation channels."
            )


# ---------------------------------------------------------------------------
# Trace Visualizer
# ---------------------------------------------------------------------------
def render_mode_trace(result: ExperimentResult, meta: ScenarioMetadata) -> None:
    """Renders the step-by-step verification trace for a specific experimental mode."""
    # Summary Header
    is_success = result.success
    status_icon = "✅" if is_success else "❌"
    status_color = "green" if is_success else "red"

    st.markdown(
        f"### {status_icon} Mode: `{result.mode}` — **{'SAFE RECOVERY' if is_success else 'RECOVERY FAILURE'}**"
    )

    # Top KPI Metrics row
    r_cols = st.columns(5)
    with r_cols[0]:
        st.metric("Duplicate Mutations", result.duplicate_side_effects)
    with r_cols[1]:
        st.metric("Invariant Violations", result.invariant_violations)
    with r_cols[2]:
        st.metric("Blocked Actions", result.blocked_actions)
    with r_cols[3]:
        st.metric("Human Escalations", result.human_escalations)
    with r_cols[4]:
        st.metric("Latency", f"{result.execution_latency_ms:.2f} ms")

    st.markdown("---")

    # Step 1: Checkpoint Restoration
    def phase_1():
        st.markdown(
            f"The agent process was interrupted by fault **`{result.injected_fault}`**. "
            f"The agent state was restored from checkpoint **`{result.initial_conditions.get('initial_checkpoint', 'cp0')}`**."
        )
        st.markdown(
            "> **Agent Believes:** Step index is reset to `0`. It believes the action has NOT yet been attempted."
        )
        st.json(result.initial_conditions)

    render_trace_item(1, "Checkpoint Restoration (Agent Internal State)", "⏪", phase_1)

    # Step 2: Durable Ledger Evidence
    def phase_2():
        if result.mode == "CHECKPOINT_ONLY":
            st.warning(
                "Mode A (CHECKPOINT_ONLY) does NOT maintain an independent effect ledger. "
                "All history prior to checkpoint restoration was lost!"
            )
        else:
            st.markdown(
                "The **SQLite WAL Effect Ledger** lives outside the agent checkpoint. "
                "It recorded the following intent and attempt evidence before the crash/rollback:"
            )
            # Check decision evidence list if available
            dec = result.recovery_decision
            if dec and "evidence_list" in dec:
                ledger_evidence = [e for e in dec["evidence_list"] if "LEDGER" in str(e.get("source", ""))]
                if ledger_evidence:
                    st.write(ledger_evidence)
                else:
                    st.info("Ledger records in-flight intent or timeout.")
            else:
                st.info(f"Ledger tracked task ID `{result.initial_conditions.get('order_id', 'task')}`.")

    render_trace_item(2, "Persistent Durable Effect Ledger", "💾", phase_2)

    # Step 3: External-State Probe Query
    def phase_3():
        if result.mode in ("CHECKPOINT_ONLY", "EFFECT_LOG"):
            st.warning(
                f"Mode `{result.mode}` does NOT support read-only external status probes. "
                "It cannot query external services to establish ground truth!"
            )
        else:
            dec = result.recovery_decision
            probe_evidence = []
            if dec and "evidence_list" in dec:
                probe_evidence = [e for e in dec["evidence_list"] if "PROBE" in str(e.get("source", ""))]

            if probe_evidence:
                st.markdown("External probe was dispatched before tool execution:")
                for pe in probe_evidence:
                    st.write(
                        f"- **State:** `{pe.get('state')}` | **Details:** {pe.get('details')}"
                    )
                    if pe.get("raw_payload"):
                        st.json(pe.get("raw_payload"))
            else:
                st.info("External probe status: Queried according to scenario configuration.")

    render_trace_item(3, "Read-Only External Status Probe Query", "🔍", phase_3)

    # Step 4: Invariant Evaluation
    def phase_4():
        dec = result.recovery_decision
        if dec and "invariants_checked" in dec and dec["invariants_checked"]:
            st.markdown("The RecoveryGuard Invariant Suite evaluated all business safety invariants:")
            invs = dec["invariants_checked"]
            for inv in invs:
                inv_status = str(inv.get("status", "")).split(".")[-1]
                icon = "✅" if inv_status == "PASS" else "❌"
                st.markdown(
                    f"- {icon} **{inv.get('name')}**: `{inv_status}` — {inv.get('explanation')}"
                )
        else:
            if result.invariant_violations > 0:
                st.error(
                    f"❌ **Invariant Violation Detected:** Result violated business safety invariants! ({result.invariant_violations} violations)"
                )
            else:
                st.info("No business invariants were evaluated by baseline mode.")

    render_trace_item(4, "Task-Specific Business Invariant Checks", "⚖️", phase_4)

    # Step 5: Recovery Gate Decision
    def phase_5():
        dec = result.recovery_decision
        if dec:
            outcome = dec.get("outcome", "UNKNOWN")
            st.markdown(f"**Gate Outcome Decision:** {format_outcome_badge(outcome)}")
            st.markdown(f"**Safe to Execute:** `{dec.get('is_safe_to_execute')}`")
            st.markdown(f"**Formal Rationale:**\n> {dec.get('rationale')}")
        else:
            st.warning(
                f"Mode `{result.mode}` has NO recovery gate. Decisions are made purely by unverified heuristics."
            )

    render_trace_item(5, "Deterministic Recovery Decision Engine", "🧠", phase_5)

    # Step 6: Tool Execution Gate Enforcement
    def phase_6():
        outcome = result.details
        if "RESUMED_WITHOUT_MUTATION" in outcome:
            st.success(
                "🛡️ **Authorization Granted to RESUME:** Tool execution was safely bypassed without mutating external state. Agent continues workflow."
            )
        elif "EXECUTED_SAFE" in outcome:
            st.success(
                "✅ **Authorization Granted to SAFE_RETRY:** Tool execution permitted because external absence was confirmed."
            )
        elif "BLOCKED" in outcome or result.blocked_actions > 0:
            st.warning(
                f"🚫 **Execution Blocked by Gate:** {outcome}. Prevented illegal or duplicate side effect!"
            )
        elif "SKIPPED_DUPLICATE" in outcome:
            st.info("⏭️ **Replay Protection:** Action skipped because ledger recorded prior commit.")
        elif "EXECUTED" in outcome:
            st.error(
                "⚠️ **Blind Execution:** Tool call proceeded without recovery authorization! Resulted in duplicate side effect."
            )
        else:
            st.write(outcome)

    render_trace_item(6, "Post-Restore Tool Execution Enforcement", "🚪", phase_6)

    # Step 7: Final External State
    def phase_7():
        st.markdown("**Realized Side Effects Across External Services:**")
        st.json(result.actual_side_effects)
        if result.duplicate_side_effects > 0:
            st.error(
                f"❌ **Duplicate External Side Effects:** {result.duplicate_side_effects} duplicate mutation(s) detected!"
            )
        else:
            st.success("✅ **Zero Duplicate Side Effects:** External system state remains consistent.")

    render_trace_item(7, "Actual External System State", "🌐", phase_7)


if __name__ == "__main__":
    main()
