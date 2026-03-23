"""
AI Hedge Fund — Part 2: Multi-Agent System
============================================
message_bus.py — Agent-to-Agent Communication Protocol

This is the nervous system of the multi-agent hedge fund.

Every decision in a real fund involves multiple people:
  PM wants to buy → Risk Manager checks limits → Analyst validates thesis
  Risk breach alert → PM Agent reduces position → Execution Agent acts

This module replicates that workflow in code.

Architecture:
    Agent A → publishes Message → MessageBus → routes to Agent B
    Agent B → processes → publishes Response → back to Agent A

Message types:
    REQUEST   — Agent A asks Agent B to do something
    RESPONSE  — Agent B answers Agent A's request
    ALERT     — Risk Manager broadcasts a breach (no response needed)
    CONSENSUS — Coordinator asks all agents to vote on a decision
    VOTE      — Agent casts a vote in a consensus round
    BROADCAST — One-to-all announcement

All messages are persisted to SQLite so:
    - Nothing is lost if an agent crashes
    - Full audit trail of inter-agent communication
    - Replay capability for debugging and backtesting

Design:
    SQLite-backed (no Kafka/Redis dependency for Part 2)
    Upgrade path: swap _SqliteBackend for _RedisBackend in Part 9
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("hedge_fund.message_bus")

# ─────────────────────────────────────────────────────────────────────────────
# Message types and priorities
# ─────────────────────────────────────────────────────────────────────────────

class MessageType(str, Enum):
    REQUEST   = "REQUEST"     # Ask another agent to do something
    RESPONSE  = "RESPONSE"    # Answer to a REQUEST
    ALERT     = "ALERT"       # Risk/system alert — broadcast immediately
    CONSENSUS = "CONSENSUS"   # Request a vote from all agents
    VOTE      = "VOTE"        # An agent's vote in a consensus round
    BROADCAST = "BROADCAST"   # One-to-all, no response needed
    HEARTBEAT = "HEARTBEAT"   # Agent liveness signal


class Priority(int, Enum):
    CRITICAL = 1   # Risk breach — handle immediately
    HIGH     = 2   # Allocation decision in progress
    NORMAL   = 3   # Standard analysis request
    LOW      = 4   # Background research


class MessageStatus(str, Enum):
    PENDING    = "PENDING"
    PROCESSING = "PROCESSING"
    DELIVERED  = "DELIVERED"
    FAILED     = "FAILED"
    EXPIRED    = "EXPIRED"


# ─────────────────────────────────────────────────────────────────────────────
# Message data structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Message:
    """
    A single message between agents.

    Immutable once created — any modification creates a new message.
    Fully serialisable to/from dict for SQLite storage.
    """
    message_id:   str
    message_type: MessageType
    sender:       str                    # Agent name that sent this
    recipient:    str                    # Agent name to receive, or "ALL"
    subject:      str                    # Short description
    payload:      Dict[str, Any]         # The actual content
    priority:     Priority = Priority.NORMAL
    timestamp:    datetime = field(default_factory=datetime.now)
    ttl_seconds:  int = 300              # Time-to-live: expire after 5 min
    reply_to:     Optional[str] = None   # message_id this is responding to
    status:       MessageStatus = MessageStatus.PENDING
    metadata:     Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        sender:       str,
        recipient:    str,
        subject:      str,
        payload:      Dict[str, Any],
        message_type: MessageType = MessageType.REQUEST,
        priority:     Priority = Priority.NORMAL,
        reply_to:     Optional[str] = None,
        ttl_seconds:  int = 300,
    ) -> "Message":
        return cls(
            message_id   = f"MSG_{uuid.uuid4().hex[:12].upper()}",
            message_type = message_type,
            sender       = sender,
            recipient    = recipient,
            subject      = subject,
            payload      = payload,
            priority     = priority,
            reply_to     = reply_to,
            ttl_seconds  = ttl_seconds,
        )

    @property
    def is_expired(self) -> bool:
        cutoff = self.timestamp + timedelta(seconds=self.ttl_seconds)
        return datetime.now() > cutoff

    @property
    def age_seconds(self) -> float:
        return (datetime.now() - self.timestamp).total_seconds()

    def to_dict(self) -> dict:
        return {
            "message_id":   self.message_id,
            "message_type": self.message_type.value,
            "sender":       self.sender,
            "recipient":    self.recipient,
            "subject":      self.subject,
            "payload":      json.dumps(self.payload),
            "priority":     self.priority.value,
            "timestamp":    self.timestamp.isoformat(),
            "ttl_seconds":  self.ttl_seconds,
            "reply_to":     self.reply_to,
            "status":       self.status.value,
            "metadata":     json.dumps(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        return cls(
            message_id   = d["message_id"],
            message_type = MessageType(d["message_type"]),
            sender       = d["sender"],
            recipient    = d["recipient"],
            subject      = d["subject"],
            payload      = json.loads(d["payload"]) if isinstance(d["payload"], str) else d["payload"],
            priority     = Priority(d["priority"]),
            timestamp    = datetime.fromisoformat(d["timestamp"]),
            ttl_seconds  = d["ttl_seconds"],
            reply_to     = d.get("reply_to"),
            status       = MessageStatus(d["status"]),
            metadata     = json.loads(d["metadata"]) if isinstance(d.get("metadata"), str) else (d.get("metadata") or {}),
        )

    def __repr__(self):
        return (
            f"Message({self.message_id} | {self.message_type.value} | "
            f"{self.sender}→{self.recipient} | '{self.subject}' | "
            f"priority={self.priority.name})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SQLite backend
# ─────────────────────────────────────────────────────────────────────────────

class _SqliteBackend:
    """SQLite persistence for the message bus."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id   TEXT PRIMARY KEY,
                    message_type TEXT NOT NULL,
                    sender       TEXT NOT NULL,
                    recipient    TEXT NOT NULL,
                    subject      TEXT,
                    payload      TEXT,
                    priority     INTEGER DEFAULT 3,
                    timestamp    TEXT NOT NULL,
                    ttl_seconds  INTEGER DEFAULT 300,
                    reply_to     TEXT,
                    status       TEXT DEFAULT 'PENDING',
                    metadata     TEXT DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_recipient_status
                ON messages (recipient, status, priority, timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON messages (timestamp)
            """)
            conn.commit()

    def save(self, msg: Message) -> None:
        d = msg.to_dict()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO messages
                (message_id, message_type, sender, recipient, subject,
                 payload, priority, timestamp, ttl_seconds, reply_to,
                 status, metadata)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                d["message_id"], d["message_type"], d["sender"],
                d["recipient"], d["subject"], d["payload"],
                d["priority"], d["timestamp"], d["ttl_seconds"],
                d["reply_to"], d["status"], d["metadata"],
            ))
            conn.commit()

    def fetch_pending(
        self,
        recipient: str,
        limit: int = 20,
    ) -> List[Message]:
        """Fetch pending messages for a recipient, ordered by priority."""
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM messages
                WHERE (recipient = ? OR recipient = 'ALL')
                  AND status = 'PENDING'
                ORDER BY priority ASC, timestamp ASC
                LIMIT ?
            """, (recipient, limit)).fetchall()

        messages = []
        for row in rows:
            try:
                msg = Message.from_dict(dict(row))
                if not msg.is_expired:
                    messages.append(msg)
                else:
                    self.update_status(msg.message_id, MessageStatus.EXPIRED)
            except Exception as e:
                logger.warning(f"Failed to parse message {row['message_id']}: {e}")
        return messages

    def fetch_by_id(self, message_id: str) -> Optional[Message]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM messages WHERE message_id = ?",
                (message_id,)
            ).fetchone()
        return Message.from_dict(dict(row)) if row else None

    def update_status(self, message_id: str, status: MessageStatus) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE messages SET status = ? WHERE message_id = ?",
                (status.value, message_id)
            )
            conn.commit()

    def get_conversation(self, root_message_id: str) -> List[Message]:
        """Get all messages in a conversation thread."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM messages
                WHERE message_id = ? OR reply_to = ?
                ORDER BY timestamp ASC
            """, (root_message_id, root_message_id)).fetchall()
        return [Message.from_dict(dict(r)) for r in rows]

    def purge_expired(self) -> int:
        cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM messages WHERE timestamp < ? AND status IN ('DELIVERED','EXPIRED','FAILED')",
                (cutoff,)
            )
            conn.commit()
            return cursor.rowcount

    def stats(self) -> Dict[str, int]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) FROM messages GROUP BY status"
            ).fetchall()
        return {r[0]: r[1] for r in rows}


# ─────────────────────────────────────────────────────────────────────────────
# Main MessageBus class
# ─────────────────────────────────────────────────────────────────────────────

class MessageBus:
    """
    Central message broker for the multi-agent hedge fund system.

    Agents publish messages here; the bus routes them to recipients.
    Supports synchronous request-response and asynchronous fire-and-forget.

    Usage:
        bus = MessageBus()

        # Publish a message
        msg_id = bus.publish(Message.create(
            sender    = "PortfolioManager",
            recipient = "RiskManager",
            subject   = "pre_trade_check",
            payload   = {"ticker": "AAPL", "weight_delta": 0.08},
            priority  = Priority.HIGH,
        ))

        # Consume messages (called by each agent in its processing loop)
        messages = bus.consume("RiskManager")
        for msg in messages:
            bus.ack(msg.message_id)

        # Request-response (blocking)
        response = bus.request(
            sender    = "PortfolioManager",
            recipient = "RiskManager",
            subject   = "pre_trade_check",
            payload   = {"ticker": "AAPL"},
            timeout   = 30,
        )
    """

    def __init__(self, db_path: Optional[Path] = None):
        from pathlib import Path as P
        default_db = P(__file__).parents[3] / "db" / "message_bus.db"
        self._backend = _SqliteBackend(db_path or default_db)
        self._handlers: Dict[str, List[Callable]] = {}  # recipient -> [handler_fns]
        self._lock = threading.Lock()
        self._total_published = 0
        self._total_delivered = 0

        logger.info("MessageBus initialised")

    def publish(self, message: Message) -> str:
        """
        Publish a message to the bus.

        Returns the message_id for tracking.
        Non-blocking: returns immediately after persisting.
        """
        with self._lock:
            self._backend.save(message)
            self._total_published += 1

        logger.debug(f"Published: {message}")

        # Trigger registered handlers immediately (if any)
        self._dispatch_to_handlers(message)

        return message.message_id

    def consume(
        self,
        agent_name: str,
        limit: int = 10,
    ) -> List[Message]:
        """
        Fetch pending messages for an agent.

        Call this in the agent's processing loop.
        Messages are returned in priority order (CRITICAL first).
        """
        messages = self._backend.fetch_pending(agent_name, limit)
        for msg in messages:
            self._backend.update_status(msg.message_id, MessageStatus.PROCESSING)
        return messages

    def ack(self, message_id: str) -> None:
        """Mark a message as successfully delivered."""
        self._backend.update_status(message_id, MessageStatus.DELIVERED)
        self._total_delivered += 1

    def nack(self, message_id: str) -> None:
        """Mark a message as failed (will not be retried)."""
        self._backend.update_status(message_id, MessageStatus.FAILED)

    def reply(
        self,
        original_message: Message,
        sender: str,
        payload: Dict[str, Any],
        subject: Optional[str] = None,
    ) -> str:
        """
        Send a response to a message.

        Links reply_to the original message_id for conversation threading.
        """
        response = Message.create(
            sender       = sender,
            recipient    = original_message.sender,
            subject      = subject or f"RE: {original_message.subject}",
            payload      = payload,
            message_type = MessageType.RESPONSE,
            priority     = original_message.priority,
            reply_to     = original_message.message_id,
        )
        return self.publish(response)

    def request(
        self,
        sender:     str,
        recipient:  str,
        subject:    str,
        payload:    Dict[str, Any],
        timeout:    float = 30.0,
        priority:   Priority = Priority.NORMAL,
    ) -> Optional[Message]:
        """
        Synchronous request-response.

        Publishes a REQUEST and blocks until a RESPONSE arrives
        or timeout is reached.

        Returns the Response message or None if timed out.
        """
        request_msg = Message.create(
            sender       = sender,
            recipient    = recipient,
            subject      = subject,
            payload      = payload,
            message_type = MessageType.REQUEST,
            priority     = priority,
            ttl_seconds  = int(timeout * 2),
        )
        msg_id = self.publish(request_msg)

        # Poll for response
        start = time.time()
        poll_interval = 0.1  # 100ms

        while time.time() - start < timeout:
            responses = self._backend.fetch_pending(sender, limit=20)
            for resp in responses:
                if resp.reply_to == msg_id and resp.message_type == MessageType.RESPONSE:
                    self.ack(resp.message_id)
                    logger.debug(
                        f"request() got response in {time.time()-start:.2f}s: {resp}"
                    )
                    return resp
            time.sleep(poll_interval)

        logger.warning(
            f"request() timed out after {timeout}s: "
            f"{sender}→{recipient} '{subject}'"
        )
        return None

    def broadcast_alert(
        self,
        sender:   str,
        subject:  str,
        payload:  Dict[str, Any],
        priority: Priority = Priority.CRITICAL,
    ) -> str:
        """
        Broadcast an alert to ALL agents.

        Used by Risk Manager for breach alerts.
        All agents receive it regardless of their recipient field.
        """
        alert = Message.create(
            sender       = sender,
            recipient    = "ALL",
            subject      = subject,
            payload      = payload,
            message_type = MessageType.ALERT,
            priority     = priority,
            ttl_seconds  = 600,  # Alerts persist 10 min
        )
        msg_id = self.publish(alert)
        logger.warning(f"ALERT broadcast from {sender}: {subject}")
        return msg_id

    def register_handler(
        self,
        agent_name: str,
        handler:    Callable[[Message], None],
    ) -> None:
        """
        Register a callback for immediate message delivery.

        When a message arrives for agent_name, handler() is called
        in the publishing thread (synchronous).

        Use for high-priority alerts where polling latency is unacceptable.
        """
        with self._lock:
            if agent_name not in self._handlers:
                self._handlers[agent_name] = []
            self._handlers[agent_name].append(handler)

    def _dispatch_to_handlers(self, message: Message) -> None:
        recipients = [message.recipient]
        if message.recipient == "ALL":
            recipients = list(self._handlers.keys())

        for recipient in recipients:
            if recipient in self._handlers:
                for handler in self._handlers[recipient]:
                    try:
                        handler(message)
                    except Exception as e:
                        logger.error(f"Handler error for {recipient}: {e}")

    def conversation(self, message_id: str) -> List[Message]:
        """Get the full conversation thread for a message."""
        return self._backend.get_conversation(message_id)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_published": self._total_published,
            "total_delivered": self._total_delivered,
            "db_stats": self._backend.stats(),
        }

    def purge_old_messages(self) -> int:
        n = self._backend.purge_expired()
        logger.info(f"Purged {n} old messages")
        return n


# ─────────────────────────────────────────────────────────────────────────────
# Singleton bus instance
# ─────────────────────────────────────────────────────────────────────────────

_bus: Optional[MessageBus] = None


def get_bus() -> MessageBus:
    """Get or create the singleton MessageBus."""
    global _bus
    if _bus is None:
        _bus = MessageBus()
    return _bus


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    print("=" * 60)
    print("  MessageBus — Test")
    print("=" * 60)

    bus = MessageBus()

    # Test 1: Basic publish/consume
    print("\n1. Basic publish and consume...")
    msg = Message.create(
        sender    = "PortfolioManager",
        recipient = "RiskManager",
        subject   = "pre_trade_check",
        payload   = {"ticker": "AAPL", "weight_delta": 0.08},
        priority  = Priority.HIGH,
    )
    msg_id = bus.publish(msg)
    print(f"   Published: {msg_id}")

    consumed = bus.consume("RiskManager")
    print(f"   Consumed {len(consumed)} messages")
    assert len(consumed) == 1
    bus.ack(consumed[0].message_id)

    # Test 2: Alert broadcast
    print("\n2. Alert broadcast...")
    alert_id = bus.broadcast_alert(
        sender  = "RiskManager",
        subject = "VAR_LIMIT_BREACH",
        payload = {"current_var_pct": 0.025, "limit": 0.02, "breach_pct": 25.0},
    )
    print(f"   Alert broadcast: {alert_id}")

    # All agents should get it
    for agent in ["PortfolioManager", "ExecutionAgent", "ResearchAnalyst"]:
        msgs = bus.consume(agent)
        alerts = [m for m in msgs if m.message_type == MessageType.ALERT]
        print(f"   {agent} received {len(alerts)} alerts")

    # Test 3: Stats
    print("\n3. Bus stats:")
    stats = bus.get_stats()
    print(f"   Published: {stats['total_published']}")
    print(f"   Delivered: {stats['total_delivered']}")
    print(f"   DB stats: {stats['db_stats']}")

    print("\n✅ MessageBus tests passed")
