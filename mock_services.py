"""Simulated external services with independent state stores and read-only probes.

Models real-world distributed systems:
1. NotificationService (persists delivered customer messages)
2. DeploymentRecordService (persists production deployment records)
3. PaymentGateway (fake, non-monetary transactions & refunds)
4. InventoryStore (stock reservations & releases)

Service state lives completely independently of agent checkpoints and the SQLite ledger.
Provides read-only status probes when supported, with configurable probe-unavailable
and stale-evidence scenarios.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional


class ProbeUnavailableError(Exception):
    """Raised when an external service status probe is temporarily unavailable."""
    pass


class NotificationService:
    """Simulated external customer notification service.

    Persists delivered messages independently of agent memory or checkpoint files.
    Supports read-only status probes with configurable availability and stale evidence.
    """

    def __init__(self) -> None:
        # Key: notification_id or order_id -> record
        self._sent_messages: List[Dict[str, Any]] = []
        self._by_id: Dict[str, Dict[str, Any]] = {}
        self.probe_available: bool = True
        self.stale_evidence: Optional[Any] = None

    def send_notification(
        self,
        recipient: str,
        message: str,
        order_id: Optional[str] = None,
        notification_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delivers and persistently records a notification."""
        nid = notification_id or f"ntf_{uuid.uuid4().hex[:10]}"
        record = {
            "notification_id": nid,
            "order_id": order_id or "general",
            "recipient": recipient,
            "message": message,
            "status": "DELIVERED",
            "sent_at": time.time(),
        }
        self._sent_messages.append(record)
        self._by_id[nid] = record
        return {"notification_id": nid, "status": "DELIVERED"}

    def send(self, order_id: str, recipient: str, message: str) -> Dict[str, Any]:
        """Backward-compatible send method keyed by order_id."""
        msg_id = f"msg_{order_id}_{len(self._sent_messages) + 1}"
        record = {
            "notification_id": msg_id,
            "message_id": msg_id,
            "order_id": order_id,
            "recipient": recipient,
            "message": message,
            "status": "DELIVERED",
            "sent_at": time.time(),
        }
        self._sent_messages.append(record)
        self._by_id[msg_id] = record
        return {"notification_id": msg_id, "message_id": msg_id, "status": "DELIVERED"}

    def probe_notification(self, notification_id: str) -> Optional[Dict[str, Any]]:
        """Read-only probe: Queries a specific notification by ID."""
        if not self.probe_available:
            return None
        if self.stale_evidence is not None:
            return self.stale_evidence
        rec = self._by_id.get(notification_id)
        return dict(rec) if rec else None

    def probe(self, order_id: str) -> Optional[List[Dict[str, Any]]]:
        """Read-only probe: Queries sent messages for an order."""
        if not self.probe_available:
            return None
        if self.stale_evidence is not None:
            return self.stale_evidence if isinstance(self.stale_evidence, list) else [self.stale_evidence]
        matches = [dict(m) for m in self._sent_messages if m["order_id"] == order_id]
        return matches if matches else None

    def probe_order(self, order_id: str) -> Optional[List[Dict[str, Any]]]:
        """Read-only probe alias for probing all messages for an order."""
        return self.probe(order_id)

    def get_messages_for_order(self, order_id: str) -> List[Dict[str, Any]]:
        """Returns all delivered message records for a specific order."""
        return [dict(m) for m in self._sent_messages if m.get("order_id") == order_id]

    def set_probe_available(self, available: bool) -> None:
        """Configures probe availability."""
        self.probe_available = available

    def set_stale_evidence(self, stale_data: Optional[Any]) -> None:
        """Configures simulated stale or cached evidence for probes."""
        self.stale_evidence = stale_data

    def get_delivered_count(self) -> int:
        """Returns total delivered notifications count."""
        return len(self._sent_messages)

    def reset(self) -> None:
        """Resets service state."""
        self._sent_messages.clear()
        self._by_id.clear()
        self.probe_available = True
        self.stale_evidence = None


class DeploymentRecordService:
    """Simulated external deployment tracking service.

    Persists release records independently of agent memory or checkpoint files.
    Supports read-only status probes with configurable availability and stale evidence.
    """

    def __init__(self) -> None:
        self._deployments: Dict[str, Dict[str, Any]] = {}
        self.probe_available: bool = True
        self.stale_evidence: Optional[Any] = None

    def create_deployment(
        self,
        deployment_id: str,
        service_name: str,
        version: str,
        environment: str = "production",
        config_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Records a new deployment."""
        now = time.time()
        record = {
            "deployment_id": deployment_id,
            "service_name": service_name,
            "version": version,
            "environment": environment,
            "config_hash": config_hash or "default",
            "status": "DEPLOYED",
            "deployed_at": now,
        }
        self._deployments[deployment_id] = record
        return {"deployment_id": deployment_id, "status": "DEPLOYED", "version": version}

    def rollback_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """Marks a deployment as rolled back."""
        if deployment_id in self._deployments:
            self._deployments[deployment_id]["status"] = "ROLLED_BACK"
            return {"deployment_id": deployment_id, "status": "ROLLED_BACK"}
        return {"deployment_id": deployment_id, "status": "NOT_FOUND"}

    def probe_deployment(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Read-only probe: Queries a deployment record without modifying state."""
        if not self.probe_available:
            return None
        if self.stale_evidence is not None:
            return dict(self.stale_evidence) if isinstance(self.stale_evidence, dict) else self.stale_evidence
        rec = self._deployments.get(deployment_id)
        return dict(rec) if rec else None

    def probe(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Alias for probe_deployment."""
        return self.probe_deployment(deployment_id)

    def set_probe_available(self, available: bool) -> None:
        """Configures probe availability."""
        self.probe_available = available

    def set_stale_evidence(self, stale_data: Optional[Any]) -> None:
        """Configures simulated stale or cached evidence for probes."""
        self.stale_evidence = stale_data

    def reset(self) -> None:
        """Resets service state."""
        self._deployments.clear()
        self.probe_available = True
        self.stale_evidence = None


class PaymentGateway:
    """Simulated external payment processor using fake, non-monetary transactions."""

    def __init__(self) -> None:
        self._charges: Dict[str, Dict[str, Any]] = {}
        self.probe_available: bool = True
        self.stale_evidence: Optional[Any] = None

    def charge(
        self,
        idempotency_key: str,
        order_id: str,
        amount: float,
        customer_id: str,
    ) -> Dict[str, Any]:
        """Charges a customer's simulated payment method idempotently."""
        if idempotency_key in self._charges:
            existing = self._charges[idempotency_key]
            return {
                "charge_id": existing["charge_id"],
                "status": existing["status"],
                "is_duplicate_replay": True,
                "amount": existing["amount"],
            }

        charge_id = f"ch_{order_id}_{int(time.time() * 1000)}"
        record = {
            "charge_id": charge_id,
            "order_id": order_id,
            "amount": amount,
            "customer_id": customer_id,
            "status": "CAPTURED",
            "created_at": time.time(),
        }
        self._charges[idempotency_key] = record
        return {
            "charge_id": charge_id,
            "status": "CAPTURED",
            "is_duplicate_replay": False,
            "amount": amount,
        }

    def refund(self, order_id: str) -> Dict[str, Any]:
        """Refunds any active charge associated with an order."""
        for key, record in self._charges.items():
            if record["order_id"] == order_id and record["status"] == "CAPTURED":
                record["status"] = "REFUNDED"
                return {"order_id": order_id, "status": "REFUNDED", "amount": record["amount"]}
        return {"order_id": order_id, "status": "NO_CHARGE_FOUND"}

    def probe(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Read-only probe: Queries charge status for an order without mutating state."""
        if not self.probe_available:
            return None
        if self.stale_evidence is not None:
            return dict(self.stale_evidence) if isinstance(self.stale_evidence, dict) else self.stale_evidence
        for record in self._charges.values():
            if record["order_id"] == order_id:
                return dict(record)
        return None

    def probe_all_for_order(self, order_id: str) -> Optional[List[Dict[str, Any]]]:
        """Queries all charges for an order, supporting list-based invariant checking."""
        if not self.probe_available:
            return None
        if self.stale_evidence is not None:
            return self.stale_evidence if isinstance(self.stale_evidence, list) else [self.stale_evidence]
        records = [dict(r) for r in self._charges.values() if r.get("order_id") == order_id]
        return records if records else None

    def get_charges_for_order(self, order_id: str) -> List[Dict[str, Any]]:
        """Returns all charge records for an order."""
        return [dict(r) for r in self._charges.values() if r.get("order_id") == order_id]

    def set_probe_available(self, available: bool) -> None:
        self.probe_available = available

    def set_stale_evidence(self, stale_data: Optional[Any]) -> None:
        self.stale_evidence = stale_data

    def reset(self) -> None:
        """Resets service state."""
        self._charges.clear()
        self.probe_available = True
        self.stale_evidence = None


class InventoryStore:
    """Simulated external warehouse inventory system."""

    def __init__(self, initial_stock: Optional[Dict[str, int]] = None) -> None:
        self._stock: Dict[str, int] = dict(initial_stock) if initial_stock else {"item_laptop": 10}
        self._reservations: Dict[str, Dict[str, Any]] = {}
        self.probe_available: bool = True
        self.stale_evidence: Optional[Any] = None

    def reserve(self, order_id: str, item_id: str, quantity: int = 1) -> Dict[str, Any]:
        """Reserves inventory for an order."""
        if order_id in self._reservations:
            existing = self._reservations[order_id]
            return {
                "reservation_id": existing["reservation_id"],
                "status": existing["status"],
                "is_duplicate_replay": True,
            }

        current = self._stock.get(item_id, 0)
        if current < quantity:
            return {"status": "OUT_OF_STOCK", "requested": quantity, "available": current}

        self._stock[item_id] = current - quantity
        res_id = f"res_{order_id}_{item_id}"
        record = {
            "reservation_id": res_id,
            "order_id": order_id,
            "item_id": item_id,
            "quantity": quantity,
            "status": "RESERVED",
            "created_at": time.time(),
        }
        self._reservations[order_id] = record
        return {"reservation_id": res_id, "status": "RESERVED", "is_duplicate_replay": False}

    def release(self, order_id: str) -> Dict[str, Any]:
        """Releases an existing reservation back to stock."""
        if order_id in self._reservations:
            record = self._reservations[order_id]
            if record["status"] == "RESERVED":
                item_id = record["item_id"]
                qty = record["quantity"]
                self._stock[item_id] = self._stock.get(item_id, 0) + qty
                record["status"] = "RELEASED"
                return {"order_id": order_id, "status": "RELEASED", "returned": qty}
        return {"order_id": order_id, "status": "NO_RESERVATION_FOUND"}

    def probe(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Read-only probe: Queries reservation status without mutating stock."""
        if not self.probe_available:
            return None
        if self.stale_evidence is not None:
            return dict(self.stale_evidence) if isinstance(self.stale_evidence, dict) else self.stale_evidence
        if order_id in self._reservations:
            return dict(self._reservations[order_id])
        return None

    def set_probe_available(self, available: bool) -> None:
        self.probe_available = available

    def set_stale_evidence(self, stale_data: Optional[Any]) -> None:
        self.stale_evidence = stale_data

    def get_stock(self, item_id: str) -> int:
        """Returns current stock level."""
        return self._stock.get(item_id, 0)

    def reset(self) -> None:
        """Resets service state."""
        self._reservations.clear()
        self.probe_available = True
        self.stale_evidence = None
