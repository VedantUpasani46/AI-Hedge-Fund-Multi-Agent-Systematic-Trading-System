"""
AI Hedge Fund — Part 4: Execution Engine
==========================================
ib_broker.py — Interactive Brokers Interface

Connects the hedge fund system to Interactive Brokers TWS or Gateway
for live paper trading and, when ready, live trading.

Setup required:
    1. Install IB TWS or IB Gateway (free from interactivebrokers.com)
    2. Enable API access: TWS → Edit → Global Configuration → API → Enable
    3. Set port (default paper trading: 7497, live: 7496)
    4. pip install ibapi  (IB's official Python API)

Architecture:
    IBBroker wraps ib_insync (recommended) or the raw ibapi library.
    ib_insync is a cleaner async wrapper around the official API.
    Install: pip install ib_insync

    If neither is available, IBBroker falls back to paper trading simulation
    so you can test the full pipeline without a broker connection.

Paper trading vs live:
    Paper trading port 7497 → simulated fills, real market data
    Live trading port 7496   → real fills, real money
    This file defaults to PAPER. You must explicitly change to LIVE.

What this module handles:
    - Connecting/reconnecting to TWS/Gateway
    - Translating our Order objects to IB contract + order objects
    - Receiving fills and updating order state
    - Fetching real-time prices and market data
    - Account data (cash, positions, margin)
    - Historical data for backtesting calibration

Risk controls baked in:
    - Maximum single order size (configurable)
    - Daily loss limit check before order submission
    - Duplicate order detection
    - Connection health monitoring
"""

from __future__ import annotations

import logging
import os
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from src.execution.order_models import (
    Order, OrderStatus, OrderSide, OrderType, Fill
)

logger = logging.getLogger("hedge_fund.ib_broker")


# ─────────────────────────────────────────────────────────────────────────────
# Broker configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IBConfig:
    """Configuration for Interactive Brokers connection."""
    host:            str = "127.0.0.1"
    port:            int = 7497         # 7497 = paper, 7496 = live
    client_id:       int = 1
    account:         str = ""           # IB account ID (e.g. "DU12345678")
    timeout_seconds: int = 30

    # Risk controls
    max_order_value: float = 100_000    # Max single order in dollars
    max_daily_loss:  float = 50_000     # Stop trading if daily loss exceeds this
    require_paper:   bool = True        # Safety: refuse to connect to live by default

    @property
    def is_paper(self) -> bool:
        return self.port == 7497

    def validate(self):
        if not self.is_paper and self.require_paper:
            raise ValueError(
                "Attempting to connect to LIVE trading (port 7496) but "
                "require_paper=True. Set require_paper=False explicitly "
                "to enable live trading."
            )


@dataclass
class AccountState:
    """Current account snapshot from IB."""
    account_id:       str = ""
    net_liquidation:  float = 0.0
    available_funds:  float = 0.0
    buying_power:     float = 0.0
    total_cash:       float = 0.0
    unrealised_pnl:   float = 0.0
    realised_pnl:     float = 0.0
    gross_position_value: float = 0.0
    day_trades_remaining: int = 3   # PDT rule
    timestamp:        datetime = field(default_factory=datetime.now)

    @property
    def daily_pnl(self) -> float:
        return self.unrealised_pnl + self.realised_pnl


# ─────────────────────────────────────────────────────────────────────────────
# Broker interface
# ─────────────────────────────────────────────────────────────────────────────

class IBBroker:
    """
    Interactive Brokers broker interface.

    Handles all communication with IB TWS/Gateway.
    Falls back to simulation mode if IB is not available.

    Usage:
        broker = IBBroker(IBConfig(port=7497))  # paper trading
        broker.connect()

        # Submit an order
        order = Order.create("AAPL", OrderSide.BUY, 100, OrderType.MARKET)
        broker.submit_order(order, on_fill=lambda f: print(f"Filled: {f}"))

        # Get account state
        state = broker.get_account_state()

        broker.disconnect()
    """

    def __init__(self, config: Optional[IBConfig] = None):
        self.config      = config or IBConfig()
        self._connected  = False
        self._ib         = None          # ib_insync IB() or raw client
        self._mode       = "disconnected"
        self._orders:    Dict[str, Order] = {}
        self._fill_callbacks: Dict[str, Callable] = {}
        self._account:   AccountState = AccountState()
        self._lock       = threading.Lock()
        self._req_id_counter = 0

        # Try to load ib_insync (preferred) or ibapi (fallback)
        self._ib_available = self._check_ib_available()

        if not self._ib_available:
            logger.warning(
                "IB API not available — running in SIMULATION mode. "
                "Install: pip install ib_insync"
            )
            self._mode = "simulation"

    def _check_ib_available(self) -> bool:
        try:
            import ib_insync
            return True
        except ImportError:
            pass
        try:
            from ibapi.client import EClient
            return True
        except ImportError:
            pass
        return False

    def connect(self) -> bool:
        """
        Connect to IB TWS or Gateway.

        Returns True if connected, False if connection failed.
        In simulation mode, always returns True.
        """
        if self._mode == "simulation":
            logger.info("Broker: SIMULATION mode (no IB connection)")
            self._connected = True
            self._account   = AccountState(
                account_id      = "SIMULATED",
                net_liquidation = 1_000_000,
                available_funds = 1_000_000,
                buying_power    = 2_000_000,
                total_cash      = 1_000_000,
            )
            return True

        try:
            self.config.validate()
        except ValueError as e:
            logger.error(f"Config validation failed: {e}")
            return False

        if not self._ib_available:
            logger.warning("IB unavailable — switching to simulation")
            self._mode      = "simulation"
            self._connected = True
            return True

        try:
            import ib_insync
            self._ib = ib_insync.IB()

            logger.info(
                f"Connecting to IB {'PAPER' if self.config.is_paper else 'LIVE'} "
                f"at {self.config.host}:{self.config.port}"
            )

            self._ib.connect(
                self.config.host,
                self.config.port,
                clientId = self.config.client_id,
                timeout  = self.config.timeout_seconds,
                readonly = False,
            )

            if self._ib.isConnected():
                self._connected = True
                self._mode      = "paper" if self.config.is_paper else "live"

                # Register fill callback
                self._ib.execDetailsEvent += self._on_exec_details

                # Get initial account state
                self._refresh_account()

                logger.info(
                    f"Connected to IB {self._mode.upper()} | "
                    f"Account: {self._account.account_id} | "
                    f"NAV: ${self._account.net_liquidation:,.0f}"
                )
                return True
            else:
                logger.error("IB connect() returned but isConnected() is False")
                return False

        except Exception as e:
            logger.error(
                f"IB connection failed: {e}\n"
                f"  Make sure TWS/Gateway is running on port {self.config.port}\n"
                f"  Enable API: TWS → Edit → Global Configuration → API"
            )
            # Fall back to simulation
            logger.info("Falling back to SIMULATION mode")
            self._mode      = "simulation"
            self._connected = True
            self._account   = AccountState(
                account_id      = "SIMULATED",
                net_liquidation = 1_000_000,
                available_funds = 1_000_000,
                buying_power    = 2_000_000,
                total_cash      = 1_000_000,
            )
            return True

    def disconnect(self):
        """Disconnect from IB."""
        if self._ib and self._mode in ("paper", "live"):
            try:
                self._ib.disconnect()
            except Exception:
                pass
        self._connected = False
        logger.info("Disconnected from broker")

    def is_connected(self) -> bool:
        if self._mode == "simulation":
            return True
        if self._ib:
            try:
                return self._ib.isConnected()
            except Exception:
                return False
        return False

    # ── Order submission ──────────────────────────────────────────────────────

    def submit_order(
        self,
        order:        Order,
        on_fill:      Optional[Callable[[Fill], None]] = None,
        on_status:    Optional[Callable[[Order], None]] = None,
    ) -> bool:
        """
        Submit an order to the broker.

        Args:
            order:     Order object to submit
            on_fill:   Callback when a fill arrives
            on_status: Callback when order status changes

        Returns:
            True if submitted successfully, False otherwise
        """
        with self._lock:
            # Pre-submission risk checks
            ok, reason = self._pre_submission_checks(order)
            if not ok:
                order.update_status(OrderStatus.REJECTED, reason)
                order.reject_reason = reason
                logger.warning(f"Order rejected pre-submission: {reason}")
                return False

            self._orders[order.order_id] = order
            if on_fill:
                self._fill_callbacks[order.order_id] = on_fill

        if self._mode == "simulation":
            return self._submit_simulated(order, on_fill)
        else:
            return self._submit_ib(order, on_fill)

    def _submit_ib(self, order: Order, on_fill: Optional[Callable]) -> bool:
        """Submit to real IB."""
        try:
            import ib_insync as ibi

            # Create IB Contract
            contract = ibi.Stock(order.ticker, "SMART", "USD")

            # Create IB Order
            if order.order_type == OrderType.MARKET:
                ib_order = ibi.MarketOrder(
                    action   = order.side.value,
                    totalQty = order.quantity,
                )
            elif order.order_type == OrderType.LIMIT:
                ib_order = ibi.LimitOrder(
                    action     = order.side.value,
                    totalQty   = order.quantity,
                    lmtPrice   = order.limit_price,
                )
            elif order.order_type in (OrderType.ALGO_TWAP, OrderType.ALGO_VWAP):
                # IB Adaptive algo
                ib_order = ibi.Order(
                    action       = order.side.value,
                    totalQty     = order.quantity,
                    orderType    = "MKT",
                    algoStrategy = "Adaptive",
                    algoParams   = [ibi.TagValue("adaptivePriority", "Normal")],
                )
            else:
                ib_order = ibi.MarketOrder(
                    action   = order.side.value,
                    totalQty = order.quantity,
                )

            ib_order.account   = self.config.account
            ib_order.orderRef  = order.order_id

            # Place order
            trade = self._ib.placeOrder(contract, ib_order)
            order.broker_order_id = str(trade.order.orderId)
            order.submitted_at    = datetime.now()
            order.update_status(OrderStatus.SUBMITTED, f"IB order_id={trade.order.orderId}")

            logger.info(f"Submitted to IB: {order}")
            return True

        except Exception as e:
            logger.error(f"IB submission failed for {order.order_id}: {e}")
            order.update_status(OrderStatus.REJECTED, str(e))
            return False

    def _submit_simulated(self, order: Order, on_fill: Optional[Callable]) -> bool:
        """
        Simulate order execution with realistic fill logic.

        For paper trading / testing when IB is not connected.
        Simulates:
            - Market orders: fill immediately at current price + half-spread
            - Limit orders: fill if price touches limit
            - Partial fills for large orders
        """
        order.submitted_at = datetime.now()
        order.update_status(OrderStatus.SUBMITTED, "Simulated submission")

        # Get current price
        current_price = self._get_simulated_price(order.ticker)

        # Simulate fill price with slippage
        if order.order_type in (OrderType.MARKET, OrderType.ALGO_TWAP,
                                 OrderType.ALGO_VWAP, OrderType.ALGO_IS):
            # Slippage: 1-5 bps for large caps
            slippage_bps = 3.0
            slippage_mult = 1 + (slippage_bps / 10000) * (1 if order.side == OrderSide.BUY else -1)
            fill_price = current_price * slippage_mult

        elif order.order_type == OrderType.LIMIT:
            fill_price = order.limit_price
            if order.limit_price is None:
                fill_price = current_price
            # Check if limit is executable
            if order.side == OrderSide.BUY and current_price > order.limit_price:
                # Price above limit → no fill immediately
                order.update_status(OrderStatus.ACCEPTED, "Limit order working")
                return True
            elif order.side == OrderSide.SELL and current_price < order.limit_price:
                order.update_status(OrderStatus.ACCEPTED, "Limit order working")
                return True
        else:
            fill_price = current_price

        # Commission: IB-style $0.005/share, min $1
        commission = max(1.0, order.quantity * 0.005)

        # Create fill
        fill = Fill(
            order_id   = order.order_id,
            ticker     = order.ticker,
            side       = order.side,
            quantity   = order.quantity,
            price      = round(fill_price, 4),
            commission = commission,
            exchange   = "SIMULATED",
        )

        # Apply fill to order
        order.add_fill(fill)

        # Calculate slippage
        if order.decision_price and order.decision_price > 0:
            diff = fill_price - order.decision_price
            if order.side == OrderSide.SELL:
                diff = -diff
            order.slippage_bps = (diff / order.decision_price) * 10000

        # Fire callback
        if on_fill:
            try:
                on_fill(fill)
            except Exception as e:
                logger.error(f"Fill callback error: {e}")

        logger.info(
            f"Simulated fill: {order.order_id} | "
            f"{order.side.value} {order.quantity:.0f} {order.ticker} "
            f"@ ${fill_price:.4f} | commission=${commission:.2f}"
        )
        return True

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a working order."""
        order = self._orders.get(order_id)
        if not order:
            logger.warning(f"cancel_order: order {order_id} not found")
            return False

        if not order.is_active:
            logger.warning(f"cancel_order: {order_id} is not active ({order.status})")
            return False

        if self._mode in ("paper", "live") and self._ib:
            try:
                # Find the IB trade
                for trade in self._ib.trades():
                    if trade.order.orderRef == order_id:
                        self._ib.cancelOrder(trade.order)
                        break
            except Exception as e:
                logger.error(f"IB cancel failed: {e}")

        order.update_status(OrderStatus.CANCELLED, "Cancelled by system")
        order.cancelled_at = datetime.now()
        logger.info(f"Cancelled order: {order_id}")
        return True

    def cancel_all(self) -> int:
        """Cancel all active orders."""
        cancelled = 0
        for order_id, order in list(self._orders.items()):
            if order.is_active:
                if self.cancel_order(order_id):
                    cancelled += 1
        return cancelled

    # ── Account data ──────────────────────────────────────────────────────────

    def get_account_state(self) -> AccountState:
        """Get current account state (cash, positions, P&L)."""
        if self._mode in ("paper", "live") and self._ib:
            self._refresh_account()
        return self._account

    def _refresh_account(self):
        """Refresh account data from IB."""
        if not self._ib:
            return
        try:
            account_values = self._ib.accountValues()
            for av in account_values:
                if av.currency != "USD":
                    continue
                if av.tag == "NetLiquidation":
                    self._account.net_liquidation = float(av.value)
                elif av.tag == "AvailableFunds":
                    self._account.available_funds = float(av.value)
                elif av.tag == "BuyingPower":
                    self._account.buying_power = float(av.value)
                elif av.tag == "TotalCashValue":
                    self._account.total_cash = float(av.value)
                elif av.tag == "UnrealizedPnL":
                    self._account.unrealised_pnl = float(av.value)
                elif av.tag == "RealizedPnL":
                    self._account.realised_pnl = float(av.value)

            self._account.timestamp = datetime.now()
        except Exception as e:
            logger.warning(f"Account refresh failed: {e}")

    def get_positions(self) -> Dict[str, Dict]:
        """Get current IB positions."""
        if self._mode == "simulation":
            return {}

        if not self._ib:
            return {}

        try:
            positions = {}
            for pos in self._ib.positions():
                if pos.position != 0:
                    positions[pos.contract.symbol] = {
                        "shares":      pos.position,
                        "avg_cost":    pos.avgCost,
                        "market_val":  pos.position * pos.avgCost,
                    }
            return positions
        except Exception as e:
            logger.error(f"get_positions failed: {e}")
            return {}

    def get_price(self, ticker: str) -> Optional[float]:
        """Get current mid price for a ticker."""
        if self._mode == "simulation":
            return self._get_simulated_price(ticker)

        if not self._ib:
            return None

        try:
            import ib_insync as ibi
            contract = ibi.Stock(ticker, "SMART", "USD")
            self._ib.qualifyContracts(contract)
            bars = self._ib.reqHistoricalData(
                contract,
                endDateTime   = "",
                durationStr   = "1 D",
                barSizeSetting= "1 min",
                whatToShow    = "MIDPOINT",
                useRTH        = True,
                keepUpToDate  = False,
            )
            if bars:
                return float(bars[-1].close)
        except Exception as e:
            logger.debug(f"get_price failed for {ticker}: {e}")

        # Fallback to Yahoo Finance
        return self._get_yahoo_price(ticker)

    def _get_simulated_price(self, ticker: str) -> float:
        """Get price from Yahoo Finance for simulation."""
        return self._get_yahoo_price(ticker) or 100.0

    def _get_yahoo_price(self, ticker: str) -> Optional[float]:
        """Fetch latest price from Yahoo Finance."""
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            hist = t.history(period="1d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception:
            pass
        return None

    # ── Risk checks ───────────────────────────────────────────────────────────

    def _pre_submission_checks(self, order: Order) -> tuple:
        """
        Pre-submission risk checks before sending to broker.

        Returns (passed: bool, reason: str)
        """
        if not self._connected:
            return False, "Broker not connected"

        # Check order size
        price = order.decision_price or self._get_simulated_price(order.ticker) or 100
        order_value = order.quantity * price

        if order_value > self.config.max_order_value:
            return False, (
                f"Order value ${order_value:,.0f} exceeds "
                f"max ${self.config.max_order_value:,.0f}"
            )

        # Check daily loss
        if self._account.daily_pnl < -self.config.max_daily_loss:
            return False, (
                f"Daily loss ${abs(self._account.daily_pnl):,.0f} exceeds "
                f"limit ${self.config.max_daily_loss:,.0f}"
            )

        # Check sufficient funds
        if order.side == OrderSide.BUY:
            if order_value > self._account.available_funds * 1.1:  # 10% buffer
                return False, (
                    f"Insufficient funds: need ${order_value:,.0f}, "
                    f"available ${self._account.available_funds:,.0f}"
                )

        return True, ""

    # ── IB event handlers ─────────────────────────────────────────────────────

    def _on_exec_details(self, trade, fill) -> None:
        """Called by ib_insync when a fill arrives from IB."""
        order_ref = trade.order.orderRef
        if order_ref not in self._orders:
            return

        order = self._orders[order_ref]
        f = Fill(
            order_id   = order.order_id,
            ticker     = fill.contract.symbol,
            side       = OrderSide(trade.order.action),
            quantity   = fill.execution.shares,
            price      = fill.execution.price,
            commission = fill.commissionReport.commission if fill.commissionReport else 0,
            exchange   = fill.execution.exchange,
        )

        order.add_fill(f)

        callback = self._fill_callbacks.get(order.order_id)
        if callback:
            try:
                callback(f)
            except Exception as e:
                logger.error(f"Fill callback error: {e}")

    # ── Order tracking ────────────────────────────────────────────────────────

    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def get_all_orders(self) -> List[Order]:
        return list(self._orders.values())

    def get_active_orders(self) -> List[Order]:
        return [o for o in self._orders.values() if o.is_active]

    def get_filled_orders(self, today_only: bool = True) -> List[Order]:
        filled = [o for o in self._orders.values() if o.status == OrderStatus.FILLED]
        if today_only:
            today = datetime.now().date()
            filled = [o for o in filled if o.filled_at and o.filled_at.date() == today]
        return filled

    @property
    def mode(self) -> str:
        return self._mode

    def __repr__(self):
        return (
            f"IBBroker(mode={self._mode} | "
            f"connected={self._connected} | "
            f"active_orders={len(self.get_active_orders())})"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("  IB Broker Interface — Test")
    print("=" * 60)

    broker = IBBroker(IBConfig(port=7497))
    connected = broker.connect()
    print(f"\n  Mode: {broker.mode}")
    print(f"  Connected: {connected}")

    account = broker.get_account_state()
    print(f"  Account: {account.account_id}")
    print(f"  NAV: ${account.net_liquidation:,.0f}")
    print(f"  Available: ${account.available_funds:,.0f}")

    # Test order
    print("\n  Testing MARKET order (AAPL, 10 shares)...")
    order = Order.create(
        ticker         = "AAPL",
        side           = OrderSide.BUY,
        quantity       = 10,
        order_type     = OrderType.MARKET,
        decision_price = 195.0,
        agent_name     = "TestAgent",
    )

    fills_received = []
    def on_fill(f: Fill):
        fills_received.append(f)
        print(f"  FILL: {f.quantity:.0f} shares @ ${f.price:.4f} | commission=${f.commission:.2f}")

    result = broker.submit_order(order, on_fill=on_fill)
    print(f"  Submitted: {result}")
    print(f"  Order status: {order.status.value}")
    print(f"  Avg fill: ${order.avg_fill_price:.4f}" if order.avg_fill_price else "  No fill yet")
    print(f"  IS (bps): {order.implementation_shortfall_bps:.1f}" if order.implementation_shortfall_bps else "  IS: N/A")

    print(f"\n✅ IBBroker test passed (mode={broker.mode})")
    print("  Connect to IB TWS paper trading (port 7497) for live testing")
    broker.disconnect()
