"""
AI Hedge Fund — Part 4: Execution Engine
==========================================
order_models.py — Complete Trade Lifecycle Data Structures

Every order in the system passes through these stages:
    PENDING   → Created, not yet sent to broker
    SUBMITTED → Sent to broker, awaiting acknowledgement
    ACCEPTED  → Broker confirmed receipt
    PARTIAL   → Partially filled
    FILLED    → Fully executed
    CANCELLED → Cancelled before fill
    REJECTED  → Broker rejected (bad params, insufficient margin, etc.)
    EXPIRED   → Time-in-force expired without fill

Why this matters:
    A production order management system needs to track every state
    transition, every partial fill, and every cost component.
    Sloppy order tracking = wrong P&L = wrong risk = catastrophic
    decisions downstream.

Key design decisions:
    - Immutable order ID from creation
    - Every state change produces an audit record
    - All costs tracked separately (commission, slippage, market impact)
    - Fill price vs decision price tracked for performance attribution
    - IB-compatible field naming for direct broker integration
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class OrderStatus(str, Enum):
    PENDING   = "PENDING"
    SUBMITTED = "SUBMITTED"
    ACCEPTED  = "ACCEPTED"
    PARTIAL   = "PARTIAL"
    FILLED    = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED  = "REJECTED"
    EXPIRED   = "EXPIRED"


class OrderSide(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET        = "MARKET"
    LIMIT         = "LIMIT"
    STOP          = "STOP"
    STOP_LIMIT    = "STOP_LIMIT"
    MOC           = "MOC"          # Market on Close
    LOC           = "LOC"          # Limit on Close
    ALGO_TWAP     = "ALGO_TWAP"
    ALGO_VWAP     = "ALGO_VWAP"
    ALGO_IS       = "ALGO_IS"      # Implementation Shortfall
    ALGO_SNIPER   = "ALGO_SNIPER"  # Opportunistic execution


class TimeInForce(str, Enum):
    DAY  = "DAY"       # Good for the day
    GTC  = "GTC"       # Good till cancelled
    IOC  = "IOC"       # Immediate or cancel
    FOK  = "FOK"       # Fill or kill
    GTD  = "GTD"       # Good till date


class ExecutionAlgo(str, Enum):
    TWAP  = "TWAP"   # Time-Weighted Average Price
    VWAP  = "VWAP"   # Volume-Weighted Average Price
    IS    = "IS"     # Implementation Shortfall (Almgren-Chriss)
    POV   = "POV"    # Percentage of Volume


# ─────────────────────────────────────────────────────────────────────────────
# Core order structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Order:
    """
    A single order in the system.

    Created by the Execution Agent from an AgentDecision.
    Tracks the full lifecycle from creation to final fill.
    """
    order_id:        str
    ticker:          str
    side:            OrderSide
    quantity:        float          # Total shares/units to execute
    order_type:      OrderType
    time_in_force:   TimeInForce = TimeInForce.DAY

    # Pricing
    limit_price:     Optional[float] = None    # For LIMIT orders
    stop_price:      Optional[float] = None    # For STOP orders
    decision_price:  Optional[float] = None    # Price when decision was made

    # Algo parameters
    algo:            Optional[ExecutionAlgo] = None
    algo_params:     Dict = field(default_factory=dict)

    # State tracking
    status:          OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price:  Optional[float] = None
    commission:      float = 0.0
    slippage_bps:    Optional[float] = None

    # Timestamps
    created_at:      datetime = field(default_factory=datetime.now)
    submitted_at:    Optional[datetime] = None
    filled_at:       Optional[datetime] = None
    cancelled_at:    Optional[datetime] = None

    # Broker fields
    broker_order_id: Optional[str] = None     # IB permId or similar
    account:         str = ""

    # Attribution
    agent_name:      str = ""
    decision_id:     str = ""
    strategy:        str = ""
    reasoning:       str = ""

    # State history
    status_history:  List[Dict] = field(default_factory=list)
    fills:           List["Fill"] = field(default_factory=list)
    reject_reason:   str = ""

    @classmethod
    def create(
        cls,
        ticker:        str,
        side:          OrderSide,
        quantity:      float,
        order_type:    OrderType = OrderType.MARKET,
        **kwargs
    ) -> "Order":
        return cls(
            order_id   = f"ORD_{uuid.uuid4().hex[:10].upper()}",
            ticker     = ticker,
            side       = side,
            quantity   = quantity,
            order_type = order_type,
            **kwargs,
        )

    @property
    def is_active(self) -> bool:
        return self.status in (
            OrderStatus.PENDING, OrderStatus.SUBMITTED,
            OrderStatus.ACCEPTED, OrderStatus.PARTIAL
        )

    @property
    def is_done(self) -> bool:
        return self.status in (
            OrderStatus.FILLED, OrderStatus.CANCELLED,
            OrderStatus.REJECTED, OrderStatus.EXPIRED
        )

    @property
    def remaining_quantity(self) -> float:
        return max(0.0, self.quantity - self.filled_quantity)

    @property
    def fill_pct(self) -> float:
        return self.filled_quantity / self.quantity if self.quantity > 0 else 0.0

    @property
    def gross_value(self) -> float:
        if self.avg_fill_price:
            return self.filled_quantity * self.avg_fill_price
        return 0.0

    @property
    def net_value(self) -> float:
        return self.gross_value + self.commission

    @property
    def implementation_shortfall_bps(self) -> Optional[float]:
        """
        Implementation shortfall vs decision price.
        Positive = we paid more than decision price (bad for buys)
        """
        if self.avg_fill_price is None or self.decision_price is None:
            return None
        if self.decision_price == 0:
            return None
        diff = self.avg_fill_price - self.decision_price
        if self.side == OrderSide.SELL:
            diff = -diff   # For sells, lower fill price = shortfall
        return (diff / self.decision_price) * 10000  # basis points

    def update_status(self, new_status: OrderStatus, note: str = "") -> None:
        self.status_history.append({
            "from":  self.status.value,
            "to":    new_status.value,
            "at":    datetime.now().isoformat(),
            "note":  note,
        })
        self.status = new_status

    def add_fill(self, fill: "Fill") -> None:
        self.fills.append(fill)
        self.filled_quantity += fill.quantity

        # Recalculate weighted avg fill price
        total_value = sum(f.quantity * f.price for f in self.fills)
        total_qty   = sum(f.quantity for f in self.fills)
        self.avg_fill_price = total_value / total_qty if total_qty > 0 else None

        self.commission += fill.commission

        if self.filled_quantity >= self.quantity * 0.9999:
            self.update_status(OrderStatus.FILLED, "Fully executed")
            self.filled_at = datetime.now()
        else:
            self.update_status(OrderStatus.PARTIAL, f"Filled {self.fill_pct:.0%}")

    def to_dict(self) -> dict:
        return {
            "order_id":       self.order_id,
            "ticker":         self.ticker,
            "side":           self.side.value,
            "quantity":       self.quantity,
            "order_type":     self.order_type.value,
            "status":         self.status.value,
            "filled_qty":     self.filled_quantity,
            "avg_fill":       self.avg_fill_price,
            "decision_price": self.decision_price,
            "commission":     self.commission,
            "is_bps":         self.implementation_shortfall_bps,
            "gross_value":    self.gross_value,
            "agent":          self.agent_name,
            "decision_id":    self.decision_id,
            "created_at":     self.created_at.isoformat(),
            "filled_at":      self.filled_at.isoformat() if self.filled_at else None,
        }

    def __repr__(self):
        return (
            f"Order({self.order_id} | {self.side.value} {self.quantity:.0f} "
            f"{self.ticker} @ {self.order_type.value} | "
            f"{self.status.value} {self.fill_pct:.0%})"
        )


@dataclass
class Fill:
    """A single execution fill — one Order can have multiple partial fills."""
    fill_id:    str = field(default_factory=lambda: f"FILL_{uuid.uuid4().hex[:8].upper()}")
    order_id:   str = ""
    ticker:     str = ""
    side:       OrderSide = OrderSide.BUY
    quantity:   float = 0.0
    price:      float = 0.0
    commission: float = 0.0
    exchange:   str = ""
    timestamp:  datetime = field(default_factory=datetime.now)
    venue:      str = ""       # ARCA, NYSE, NASDAQ, etc.


@dataclass
class ExecutionReport:
    """
    Complete execution report for a trade from decision to settlement.
    Used for performance attribution and TCA (Transaction Cost Analysis).
    """
    order:               Order
    pre_trade_estimate:  Optional["PreTradeEstimate"] = None
    post_trade_analysis: Optional["PostTradeAnalysis"] = None

    @property
    def ticker(self) -> str:
        return self.order.ticker

    @property
    def is_complete(self) -> bool:
        return self.order.is_done and self.post_trade_analysis is not None


@dataclass
class PreTradeEstimate:
    """Pre-trade cost estimate before execution."""
    ticker:             str
    quantity:           float
    side:               OrderSide
    decision_price:     float
    estimated_slippage_bps: float
    estimated_commission:   float
    estimated_market_impact_bps: float
    market_adv:         float    # Average Daily Volume
    participation_rate: float    # Our order / ADV
    algo_recommended:   ExecutionAlgo
    horizon_minutes:    int
    timestamp:          datetime = field(default_factory=datetime.now)

    @property
    def total_estimated_cost_bps(self) -> float:
        return self.estimated_slippage_bps + self.estimated_market_impact_bps

    @property
    def total_estimated_cost_usd(self) -> float:
        return (self.total_estimated_cost_bps / 10000) * self.quantity * self.decision_price + self.estimated_commission


@dataclass
class PostTradeAnalysis:
    """Post-trade Transaction Cost Analysis (TCA)."""
    order_id:             str
    ticker:               str
    arrival_price:        float   # Price when order was submitted
    avg_fill_price:       float
    twap_benchmark:       float   # TWAP over execution period
    vwap_benchmark:       float   # VWAP over execution period
    close_price:          float   # Day's closing price

    # Slippage components
    market_impact_bps:    float   # Permanent price impact
    timing_cost_bps:      float   # Cost of time taken to execute
    spread_cost_bps:      float   # Half-spread paid
    total_cost_bps:       float

    # vs benchmarks
    vs_arrival_bps:       float
    vs_twap_bps:          float
    vs_vwap_bps:          float
    vs_close_bps:         float

    execution_time_min:   float
    fill_rate_pct:        float
    timestamp:            datetime = field(default_factory=datetime.now)

    def summary(self) -> str:
        side_sign = 1 if self.vs_arrival_bps >= 0 else -1
        return (
            f"TCA: {self.ticker} | "
            f"Avg fill: ${self.avg_fill_price:.4f} | "
            f"vs Arrival: {self.vs_arrival_bps:+.1f}bps | "
            f"vs VWAP: {self.vs_vwap_bps:+.1f}bps | "
            f"Total cost: {self.total_cost_bps:.1f}bps | "
            f"Time: {self.execution_time_min:.1f}min"
        )
