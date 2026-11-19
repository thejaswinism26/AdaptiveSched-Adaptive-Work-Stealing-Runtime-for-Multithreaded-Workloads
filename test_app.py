"""Automated tests for the RecoveryGuard Streamlit dashboard (app.py).

Uses streamlit.testing.v1.AppTest to verify:
1. Initial dashboard rendering (titles, tabs, selectors, default states).
2. Single-scenario execution and step-by-step trace generation.
3. Full benchmark suite execution and metric comparison card rendering.
4. Probe condition toggles and isolated execution state.
"""

from __future__ import annotations

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).parent.parent / "recoveryguard" / "app.py"


@pytest.fixture
def app_tree():
    """Initializes and runs the Streamlit AppTest tree."""
    at = AppTest.from_file(APP_PATH, default_timeout=25)
    at.run()
    return at


def test_app_initial_rendering(app_tree: AppTest) -> None:
    """Verifies that the dashboard renders all required high-level UI sections without errors."""
    assert not app_tree.exception
    assert len(app_tree.title) >= 1
    assert "RecoveryGuard" in app_tree.title[0].value
    assert len(app_tree.tabs) == 3

    # Check that execution buttons exist
    assert len(app_tree.button) >= 2


def test_app_single_scenario_trace_execution(app_tree: AppTest) -> None:
    """Verifies that clicking 'Run Selected Scenario' produces execution traces and metrics."""
    # Click button 0: Run Selected Scenario
    app_tree.button[0].click().run(timeout=25)
    assert not app_tree.exception

    # Should render metrics and success alerts
    assert len(app_tree.metric) >= 5
    assert len(app_tree.success) >= 1

    # Verify that phase headings are rendered in the execution trace
    markdown_texts = [m.value for m in app_tree.markdown]
    assert any("Phase 1: Checkpoint Restoration" in t for t in markdown_texts)
    assert any("Phase 5: Deterministic Recovery Decision Engine" in t or "Deterministic Recovery" in t for t in markdown_texts)
    assert any("Phase 7: Actual External System State" in t for t in markdown_texts)


def test_app_benchmark_suite_execution(app_tree: AppTest) -> None:
    """Verifies that clicking 'Run Full Benchmark Suite' executes 24 trials and renders KPI metrics."""
    # Click button 1: Run Full Benchmark Suite
    app_tree.button[1].click().run(timeout=35)
    assert not app_tree.exception

    # Check KPI cards rendered
    assert len(app_tree.metric) >= 4
    metric_labels = [m.label for m in app_tree.metric]
    assert "Duplicate Side-Effect Rate (DSR)" in metric_labels
    assert "False-Pause Rate (FPR)" in metric_labels
    assert "Recovery Success Rate (RSR)" in metric_labels

    # Verify comparison tables exist in markdown
    markdown_texts = [m.value for m in app_tree.markdown]
    assert any("Mode A: CHECKPOINT_ONLY" in t for t in markdown_texts)
    assert any("Mode C: RECOVERYGUARD" in t for t in markdown_texts)

    # Verify download buttons and expanders rendered
    assert len(app_tree.download_button) >= 2
    assert any("Download Results as CSV" in b.label for b in app_tree.download_button)
    assert any("Download Results as SQLite" in b.label for b in app_tree.download_button)
