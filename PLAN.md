# RecoveryGuard: Implementation Plan

## 1. Project Overview & Architecture

RecoveryGuard is a lightweight, rigorous research prototype investigating cross-system recovery verification for AI agents. When an AI agent crashes or is rolled back to a previous checkpoint, internal agent state reverts, but external side effects (e.g., API calls, financial transactions, database mutations) persist. 

RecoveryGuard introduces a **deterministic, uncertainty-aware Recovery Gate** positioned strictly before any post-restore tool invocation. The gate cross-references:
1. A durable effect ledger (persisted independently of agent memory checkpoints).
2. External read-only state probes (verifying actual external system reality).
3. Task-specific invariants (validating cross-system business constraints).

```
[Agent Runtime] ---> (Crash / Rollback to Checkpoint)
       │
       ▼
[Mandatory Recovery Gate]  <--- Deterministic Decision Engine (NO LLM)
   ├── Query Durable SQLite Effect Ledger
   ├── Execute Read-Only External Probes
   └── Evaluate Task-Specific Invariants
       │
       ├── Evidence: CONFIRMED_SUCCESS | CONFIRMED_ABSENT | UNKNOWN | CONFLICTING
       ▼
   Outcome: RESUME | RECONCILE | SAFE_RETRY | COMPENSATE | QUARANTINE | HUMAN_REVIEW
```

---

## 2. Directory Structure

```text
os_project/
├── PLAN.md                   # Implementation plan and milestone roadmap
├── README.md                 # System documentation, research hypothesis, usage guide
├── requirements.txt          # Python dependencies (streamlit, pytest)
├── .gitignore                # Git ignore rules for Python, SQLite, checkpoints
├── recoveryguard/
│   ├── __init__.py           # Package initialization
│   ├── models.py             # Data models (EvidenceState, RecoveryOutcome, ActionIntent, EffectRecord, etc.)
│   ├── database.py           # SQLite WAL connection manager and transaction context
│   ├── ledger.py             # Durable effect ledger (persists across agent rollbacks)
│   ├── mock_services.py      # Simulated external services (PaymentGateway, InventoryStore, NotificationService)
│   ├── checkpoints.py        # File-based agent memory & execution checkpoint manager
│   ├── invariants.py         # Task-specific consistency invariants (e.g. no duplicate charges)
│   ├── verifier.py           # Multi-source evidence synthesis engine
│   ├── recovery_engine.py    # Deterministic Recovery Gate decision engine
│   ├── agent.py              # Agent workflow runner supporting 3 recovery modes
│   ├── experiments.py        # Automated benchmark runner across fault matrices
│   └── app.py                # Interactive Streamlit dashboard
└── tests/
    ├── __init__.py
    ├── test_models.py        # Unit tests for domain models, enums, and immutability
    ├── test_ledger.py        # Unit tests for SQLite persistence, WAL transactions, and timeouts
    └── test_engine.py        # Unit tests for verifier evidence states and gate outcomes
```

---

## 3. Main Data Models & Enums

### Explicit Evidence States
- `CONFIRMED_SUCCESS`: Effect log and/or external probe confirms the action took effect.
- `CONFIRMED_ABSENT`: Active external probe confirms the action definitely did not occur.
- `UNKNOWN`: Ambiguous outcome (e.g. in-flight timeout, unsupported probe). Never assumed to have failed.
- `CONFLICTING`: Contradiction between durable effect ledger and external observed state.

### Explicit Recovery Outcomes
- `RESUME`: Prior effect committed externally; bypass duplicate mutation and resume agent.
- `RECONCILE`: State divergence detected; reconcile internal agent memory with external state.
- `SAFE_RETRY`: Effect confirmed absent or tool is idempotent; safe to execute.
- `COMPENSATE`: Invariant violation detected; execute compensating action (e.g. refund/release).
- `QUARANTINE`: Conflicting unresolvable state; isolate task without further mutations.
- `HUMAN_REVIEW`: Status unknown on risky non-idempotent mutation; halt for human operator approval.

### Structured Dataclasses
- `ActionIntent`: Unique `action_id`, logical `task_id`, stable `operation_id`, tool name, parameters, `risk_level`, `checkpoint_id`, timestamp.
- `EffectEvidence`: Source (`LOCAL_LEDGER`, `EXTERNAL_PROBE`, `INVARIANT_CHECK`), `EvidenceState`, observation timestamp, raw payload.
- `CheckpointMetadata`: Checkpoint ID, step index, state hash, agent memory snapshot.
- `InvariantResult`: Invariant ID, satisfied flag, violation details, affected entities.
- `RecoveryDecision`: Outcome, `is_safe_to_execute` boolean, rationale, evidence list, invariant list.
- `EffectRecord`: SQLite database row model (`PENDING`, `STARTED`, `COMMITTED`, `FAILED`, `TIMEOUT`).

---

## 4. Implementation Status & Roadmap

- [x] **Phase 1: Project Skeleton, Core Models, Durable Ledger & Foundations (COMPLETED)**
  - Typed domain models and enums in `recoveryguard/models.py`.
  - SQLite database management with WAL mode and transaction rollback in `recoveryguard/database.py`.
  - Durable effect ledger in `recoveryguard/ledger.py` with stable operation ID correlation.
  - Isolated file-based checkpoint manager in `recoveryguard/checkpoints.py`.
  - Simulated external services with independent state and read-only probes in `recoveryguard/mock_services.py`.
  - Invariants, verifier, and deterministic recovery engine in `recoveryguard/invariants.py`, `recoveryguard/verifier.py`, `recoveryguard/recovery_engine.py`.
  - Multi-mode agent workflow in `recoveryguard/agent.py`.
  - Benchmark experiment harness in `recoveryguard/experiments.py`.

- [x] **Phase 2: Persistence & Failure-Simulation Layer (COMPLETED)**
  - SQLite effect ledger tracks action intent prior to execution, attempt history, timestamps, stable operation IDs, checkpoint IDs, and structured result evidence without overwriting.
  - Independent mock external services (`NotificationService`, `DeploymentRecordService`, `PaymentGateway`) persisting state independently of agent checkpoints.
  - Read-only status probes supporting configurable probe-unavailable and stale-evidence scenarios.
  - Checkpoint manager saving/restoring agent state to JSON without modifying SQLite or external service state.
  - Deterministic fault injector simulating `TIMEOUT_AFTER_EXECUTION`, `CRASH_BEFORE_AGENT_RECORD`, and `FAILURE_BEFORE_EXECUTION`.
  - Comprehensive automated test suite in `tests/test_persistence_and_faults.py` proving guarantees A through E (25/25 tests passing).

- [x] **Phase 3: Core Recovery Decision Engine & Invariant Verification (COMPLETED)**
  - Deterministic invariants implemented for single notification delivery, deployment duplication prevention, and duplicate payment prevention.
  - Distinguishes safety invariants from workflow completion conditions (`is_safety_invariant`).
  - Synthesizes effect history, read-only external probes, and invariant results.
  - Implements strict epistemic rules: timeouts are never assumed failures, missing ledger records never prove absence, stale/unavailable probes cannot establish absence.
  - Recovery decision engine maps evidence to `RESUME`, `RECONCILE`, `SAFE_RETRY`, `COMPENSATE`, `QUARANTINE`, and `HUMAN_REVIEW`.
  - Enforces that irreversible actions (notifications) are never marked as compensatable (yielding `QUARANTINE`).
  - Conflicting evidence forbids blind retries (yielding `RECONCILE` or `HUMAN_REVIEW`).

- [x] **Phase 3.5: Mandatory Post-Restore Recovery Gate (COMPLETED)**
  - Gate behavior strictly enforced at agent tool-execution interface (`AgentWorkflow.execute_tool`).
  - Restoring any checkpoint marks agent as requiring verification (`_requires_verification = True`) and resets authorizations.
  - Authorization tokens (`GateAuthorization`) are strictly scoped by task, operation, tool, and outcome.
  - Prevents ordinary side-effecting tools under `RECONCILE` (only allows reconciliation operations) and `COMPENSATE` (only allows approved compensations).
  - Fully blocks ordinary execution under `QUARANTINE` and `HUMAN_REVIEW`.
  - Fails closed on verification exceptions, unreachable ledger, unresolvable external state on risky mutations, and detected evidence drift.
  - 8 automated tests in `tests/test_mandatory_gate.py` (42/42 total tests passing across all suites).

- [x] **Phase 4: Quantitative Evaluation Benchmark Suite (COMPLETED)**
  - Three selectable experimental modes (`CHECKPOINT_ONLY`, `EFFECT_LOG`, `RECOVERYGUARD`) implemented in `recoveryguard/experiments.py`.
  - Eight deterministic, repeatable distributed failure scenarios implemented with strict state isolation.
  - Reproducible comparative metrics: actual side effects, duplicate mutations, invariant violations, blocked actions, human escalations, and latency.
  - 14 automated tests across `tests/test_experiments.py` and `tests/test_evaluation.py` (56/56 total tests passing across all suites).

- [x] **Phase 5: Interactive Streamlit Research Dashboard (COMPLETED)**
  - Local interactive research demo in `recoveryguard/app.py`.
  - Section 1: System overview explaining the rollback problem, research hypothesis, and 3 comparative recovery modes.
  - Section 2: Interactive scenario controls supporting 8 standardized failure scenarios, mode selection (single or side-by-side comparison), configurable probe toggles (supported, online, stale), and explicit button execution.
  - Section 3: Detailed 7-phase step-by-step execution trace displaying checkpoint restoration, durable ledger records, read-only probe responses, business invariant checks, formal recovery decision rationale, gate enforcement (permit/block), and external realized side effects.
  - Section 4: Quantitative benchmark evaluation runner executing 24 isolated trials across all modes, displaying top KPI cards, summary comparison table, fine-grained per-scenario breakdown matrix, and formal metric formulas.
  - Section 5: Rigorous research limitations discussing probe cooperativeness, epistemic uncertainty (timeout != failure), distributed exactly-once limits, and prototype scope.
  - 3 automated tests in `tests/test_app.py` verifying headless rendering, single scenario execution, and benchmark execution.

- [x] **Phase 6: Multi-Trial Benchmarks, Component Ablation Study & Persistent Data Exports (COMPLETED)**
  - Multi-trial stochastic evaluation supporting $\ge 100$ trials per scenario per strategy with reproducible seeds.
  - Component ablation study implementing 4 variants: `FULL_RECOVERYGUARD`, `NO_LEDGER`, `NO_PROBES`, `NO_INVARIANTS`.
  - Export capabilities for raw and summarized experimental data to independent CSV and SQLite stores (`export_results_to_csv`, `export_results_to_sqlite`, `export_summaries_to_csv`).
  - Standardized recovery outcome mappings: `RESUME`, `RETRY`, `QUARANTINE`, `HUMAN_INTERVENTION`.
  - Measured performance bar chart and one-click export download buttons integrated cleanly into the Streamlit dashboard without modifying existing visual layouts.
  - 7 comprehensive tests in `tests/test_ablation_and_trials.py` (66/66 total automated tests passing).

---

## 5. Key Risks & Mitigations

1. **Uncertainty under Network Timeouts:**
   - *Risk:* In distributed tool execution, an HTTP timeout may mean the request was dropped before execution or succeeded right before dropping.
   - *Mitigation:* Explicitly label timed-out states as `UNKNOWN` rather than `FAILED`. If a read-only probe is available, use it to resolve `UNKNOWN` to `CONFIRMED_SUCCESS` or `CONFIRMED_ABSENT`. If unresolvable and the action is `RISKY_MUTATION`, force `HUMAN_REVIEW` rather than blind re-execution.
2. **Time-of-Check to Time-of-Use (TOCTOU) External Drift:**
   - *Risk:* External system state may change between probe verification and tool action.
   - *Mitigation:* Keep the recovery gate strictly adjacent to execution (blocking gate invocation immediately prior to tool dispatch) and verify entity version/timestamp tags during invariant checks.
3. **False Pause / Excessive Interventions:**
   - *Risk:* Being overly cautious could pause agent workflows when simple idempotent replay or safe progression was possible.
   - *Mitigation:* Classify tool risks (`SAFE_READ`, `IDEMPOTENT_WRITE`, `RISKY_MUTATION`). Idempotent tools with verified `CONFIRMED_SUCCESS` can be safely resolved to `RESUME` or replayed if idempotent keys match, reserving pauses only for true `UNKNOWN` or `CONFLICTING` states on risky mutations.
4. **LLM Non-Determinism in Recovery:**
   - *Risk:* Using an LLM to decide whether recovery is safe could lead to hallucinated recovery decisions.
   - *Mitigation:* The recovery gate is strictly rule-based, deterministic, and algorithmic, requiring 0 LLM calls for the safety gate decision.
