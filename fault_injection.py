"""Deterministic, reproducible fault injection framework for RecoveryGuard.

Simulates three fundamental distributed systems failure modes:
1. FAILURE_BEFORE_EXECUTION:
   A normal network or client failure occurs before the external service mutates state.
   (External state is NOT modified).

2. TIMEOUT_AFTER_EXECUTION:
   The external service successfully executes the mutation, but the network connection
   drops or times out before the client receives the acknowledgment.
   (External state IS modified; client perceives TIMEOUT/UNKNOWN).

3. CRASH_BEFORE_AGENT_RECORD:
   The external service successfully executes and responds, but the agent crashes
   abruptly before it can record the committed result to its ledger or advance its checkpoint.
   (External state IS modified; local ledger only has PENDING/STARTED or nothing).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar

T = TypeVar("T")


class FaultType(str, Enum):
    """Supported deterministic fault types."""
    NONE = "NONE"
    FAILURE_BEFORE_EXECUTION = "FAILURE_BEFORE_EXECUTION"
    TIMEOUT_AFTER_EXECUTION = "TIMEOUT_AFTER_EXECUTION"
    CRASH_BEFORE_AGENT_RECORD = "CRASH_BEFORE_AGENT_RECORD"


class ServiceTimeoutError(TimeoutError):
    """Simulates an external service timeout where caller does not receive response."""
    pass


class PreExecutionFailureError(RuntimeError):
    """Simulates an error occurring before the external service executes."""
    pass


class AgentCrashError(RuntimeError):
    """Simulates an abrupt agent crash occurring before recording committed state."""
    pass


class FaultInjector:
    """Manages deterministic fault injection rules keyed by target operation or tool name."""

    def __init__(self) -> None:
        # Key: target (e.g. tool_name or operation_id) -> (fault_type, remaining_count)
        self._rules: Dict[str, Dict[str, Any]] = {}

    def set_fault(self, target: str, fault_type: FaultType, count: int = 1) -> None:
        """Sets a deterministic fault rule for a target to fire `count` times."""
        self._rules[target] = {
            "type": fault_type,
            "remaining": count,
        }

    def trigger_failure_before_execution(self, target: str, count: int = 1) -> None:
        """Convenience method to inject pre-execution failure."""
        self.set_fault(target, FaultType.FAILURE_BEFORE_EXECUTION, count)

    def trigger_timeout_after_execution(self, target: str, count: int = 1) -> None:
        """Convenience method to inject post-execution timeout."""
        self.set_fault(target, FaultType.TIMEOUT_AFTER_EXECUTION, count)

    def trigger_crash_before_record(self, target: str, count: int = 1) -> None:
        """Convenience method to inject crash before ledger commit."""
        self.set_fault(target, FaultType.CRASH_BEFORE_AGENT_RECORD, count)

    def clear(self) -> None:
        """Clears all active fault rules."""
        self._rules.clear()

    def execute_with_faults(
        self,
        target: str,
        action_fn: Callable[[], T],
        before_record_hook: Optional[Callable[[], None]] = None,
    ) -> T:
        """Executes action_fn while evaluating registered fault rules for target.

        Deterministic sequence:
        1. Check if FAILURE_BEFORE_EXECUTION rule applies. If so, decrement rule and raise PreExecutionFailureError.
        2. Execute action_fn() (external service mutates state).
        3. Check if TIMEOUT_AFTER_EXECUTION rule applies. If so, decrement rule and raise ServiceTimeoutError.
        4. If before_record_hook is provided, execute it.
        5. Check if CRASH_BEFORE_AGENT_RECORD rule applies. If so, decrement rule and raise AgentCrashError.
        6. Return the execution result.
        """
        rule = self._rules.get(target)
        if not rule or rule.get("remaining", 0) <= 0:
            result = action_fn()
            if before_record_hook:
                before_record_hook()
            return result

        fault_type = rule["type"]

        # 1. Pre-execution failure
        if fault_type == FaultType.FAILURE_BEFORE_EXECUTION:
            rule["remaining"] -= 1
            if rule["remaining"] <= 0:
                self._rules.pop(target, None)
            raise PreExecutionFailureError(
                f"[Fault Injection] Pre-execution failure injected for '{target}'"
            )

        # 2. Execute external service action
        result = action_fn()

        # 3. Post-execution timeout
        if fault_type == FaultType.TIMEOUT_AFTER_EXECUTION:
            rule["remaining"] -= 1
            if rule["remaining"] <= 0:
                self._rules.pop(target, None)
            raise ServiceTimeoutError(
                f"[Fault Injection] Post-execution timeout injected for '{target}'"
            )

        # 4. Optional hook
        if before_record_hook:
            before_record_hook()

        # 5. Crash before agent records response
        if fault_type == FaultType.CRASH_BEFORE_AGENT_RECORD:
            rule["remaining"] -= 1
            if rule["remaining"] <= 0:
                self._rules.pop(target, None)
            raise AgentCrashError(
                f"[Fault Injection] Agent crash before recording injected for '{target}'"
            )

        return result
