"""
AI Hedge Fund — Part 1: Foundation
====================================
data_models.py — Core financial data structures

These are the financial primitives every other module uses.
Getting data models right is the most important architectural decision —
everything else (agents, risk engine, execution, reporting) depends on them.

Models defined here:
  OHLCVBar       — Single price bar (Open/High/Low/Close/Volume)
  MarketSnapshot — Complete market state at a point in time
  Position       — A live portfolio holding
  Trade          — An executed trade record
  Portfolio      — Full portfolio state
  Signal         — An alpha signal from any model
  RiskMetrics    — Portfolio-level risk measurements
  AgentDecision  — An LLM agent's allocation decision

Design principles:
  - Immutable where possible (frozen dataclasses)
  - Rich in computed properties (don't store derivable values)
  - Serializable to/from dict and JSON (for database and API)
  - Typed everywhere (Literal, Optional — no untyped Any)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from enum import Enum, auto
from typing import Dict, List, Literal, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class Direction(str, Enum):
    LONG  = "LONG"
    SHORT = "SHORT"
    FLAT  = "FLAT"


class SignalStrength(str, Enum):
    STRONG_BUY    = "STRONG_BUY"
    BUY           = "BUY"
    WEAK_BUY      = "WEAK_BUY"
    NEUTRAL       = "NEUTRAL"
    WEAK_SELL     = "WEAK_SELL"
    SELL          = "SELL"
    STRONG_SELL   = "STRONG_SELL"


class Regime(str, Enum):
    BULL         = "BULL"
    BEAR         = "BEAR"
    CRISIS       = "CRISIS"
    SIDEWAYS     = "SIDEWAYS"
    HIGH_VOL     = "HIGH_VOL"
    UNKNOWN      = "UNKNOWN"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT  = "LIMIT"
    STOP   = "STOP"
    TWAP   = "TWAP"
    VWAP   = "VWAP"


class AssetClass(str, Enum):
    EQUITY        = "EQUITY"
    FIXED_INCOME  = "FIXED_INCOME"
    COMMODITY     = "COMMODITY"
    FX            = "FX"
    CRYPTO        = "CRYPTO"
    DERIVATIVE    = "DERIVATIVE"
    ETF           = "ETF"


class Conviction(str, Enum):
    HIGH   = "HIGH"
    MEDIUM = "MEDIUM"
    LOW    = "LOW"


# ─────────────────────────────────────────────────────────────────────────────
# Price Data
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OHLCVBar:
    """
    A single OHLCV price bar — the atomic unit of market data.

    Immutable by design: price history should never be modified after ingestion.
    Validated on construction: negative prices or volumes are programming errors.
    """
    ticker:     str
    timestamp:  datetime
    open:       float
    high:       float
    low:        float
    close:      float
    volume:     float
    adj_close:  Optional[float] = None   # Dividend/split adjusted
    vwap:       Optional[float] = None   # Volume-weighted average price

    def __post_init__(self):
        if self.close <= 0:
            raise ValueError(f"Close price must be positive: {self.close}")
        if self.high < self.low:
            raise ValueError(f"High {self.high} < Low {self.low} for {self.ticker}")
        if self.volume < 0:
            raise ValueError(f"Volume cannot be negative: {self.volume}")

    @property
    def mid(self) -> float:
        """High-low midpoint."""
        return (self.high + self.low) / 2.0

    @property
    def range(self) -> float:
        """Intraday price range."""
        return self.high - self.low

    @property
    def body(self) -> float:
        """Absolute candle body size."""
        return abs(self.close - self.open)

    @property
    def effective_close(self) -> float:
        """Use adjusted close if available, else raw close."""
        return self.adj_close if self.adj_close is not None else self.close

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "adj_close": self.adj_close,
            "vwap": self.vwap,
        }


@dataclass
class MarketSnapshot:
    """
    Complete market state for a set of securities at a point in time.

    This is what the Portfolio Manager Agent sees when making decisions.
    Combines price data, computed features, macro context, and regime.
    """
    timestamp:     datetime
    prices:        Dict[str, float]          # ticker -> current price
    returns_1d:    Dict[str, float]          # 1-day return
    returns_5d:    Dict[str, float]          # 5-day return
    returns_21d:   Dict[str, float]          # 21-day return
    volumes:       Dict[str, float]          # today's volume
    vols_21d:      Dict[str, float]          # 21-day realized vol (annualized)
    regime:        Regime = Regime.UNKNOWN
    vix_level:     Optional[float] = None
    spy_return_1d: Optional[float] = None    # Market benchmark
    notes:         str = ""

    @property
    def universe(self) -> List[str]:
        return list(self.prices.keys())

    def top_movers(self, n: int = 5) -> List[Tuple[str, float]]:
        """Return top N gainers by 1-day return."""
        sorted_returns = sorted(self.returns_1d.items(), key=lambda x: x[1], reverse=True)
        return sorted_returns[:n]

    def bottom_movers(self, n: int = 5) -> List[Tuple[str, float]]:
        """Return top N losers by 1-day return."""
        sorted_returns = sorted(self.returns_1d.items(), key=lambda x: x[1])
        return sorted_returns[:n]

    def high_vol_names(self, threshold: float = 0.35) -> List[str]:
        """Securities with annualized vol above threshold (35% default)."""
        return [t for t, v in self.vols_21d.items() if v > threshold]

    def market_summary(self) -> str:
        vix_str = f"VIX: {self.vix_level:.1f}" if self.vix_level else "VIX: N/A"
        spy_str = f"SPY: {self.spy_return_1d:+.2%}" if self.spy_return_1d else "SPY: N/A"
        return (
            f"Market Snapshot [{self.timestamp:%Y-%m-%d}] | "
            f"Regime: {self.regime.value} | {vix_str} | {spy_str} | "
            f"Universe: {len(self.prices)} securities"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Position:
    """
    A live position in the portfolio.

    Tracks cost basis, current market value, unrealised P&L,
    and position-level risk metrics.
    """
    ticker:          str
    direction:       Direction
    shares:          float
    avg_cost:        float          # Average cost per share
    current_price:   float
    asset_class:     AssetClass = AssetClass.EQUITY
    entry_date:      Optional[date] = None
    sector:          Optional[str] = None
    industry:        Optional[str] = None

    # Position-level risk (populated by risk engine)
    position_var_1d: Optional[float] = None   # Dollar VaR
    position_vol:    Optional[float] = None   # Annualized vol
    beta:            Optional[float] = None   # Market beta

    @property
    def market_value(self) -> float:
        """Current market value of position (always positive)."""
        return abs(self.shares) * self.current_price

    @property
    def cost_basis(self) -> float:
        """Total cost basis (what we paid)."""
        return abs(self.shares) * self.avg_cost

    @property
    def unrealised_pnl(self) -> float:
        """Unrealised P&L in dollars."""
        pnl = (self.current_price - self.avg_cost) * abs(self.shares)
        return pnl if self.direction == Direction.LONG else -pnl

    @property
    def unrealised_pnl_pct(self) -> float:
        """Unrealised P&L as fraction of cost basis."""
        if self.cost_basis == 0:
            return 0.0
        return self.unrealised_pnl / self.cost_basis

    @property
    def holding_days(self) -> Optional[int]:
        if self.entry_date is None:
            return None
        return (date.today() - self.entry_date).days

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "direction": self.direction.value,
            "shares": self.shares,
            "avg_cost": self.avg_cost,
            "current_price": self.current_price,
            "market_value": self.market_value,
            "cost_basis": self.cost_basis,
            "unrealised_pnl": self.unrealised_pnl,
            "unrealised_pnl_pct": self.unrealised_pnl_pct,
            "asset_class": self.asset_class.value,
            "sector": self.sector,
            "holding_days": self.holding_days,
        }

    def __repr__(self):
        return (
            f"Position({self.ticker} "
            f"{self.direction.value} {self.shares:.0f} shares @ "
            f"${self.avg_cost:.2f} | MV=${self.market_value:,.0f} | "
            f"PnL={self.unrealised_pnl_pct:+.1%})"
        )


@dataclass
class Portfolio:
    """
    Complete portfolio state — the source of truth for the hedge fund.

    This is what the Risk Agent monitors, the PM Agent reads,
    and the Execution Agent writes to.
    """
    portfolio_id:    str
    cash:            float
    positions:       Dict[str, Position] = field(default_factory=dict)
    initial_capital: float = 1_000_000
    timestamp:       datetime = field(default_factory=datetime.now)

    # Performance tracking
    peak_nav:        float = 0.0
    inception_date:  Optional[date] = None

    @property
    def total_market_value(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    @property
    def net_asset_value(self) -> float:
        return self.cash + self.total_market_value

    @property
    def total_unrealised_pnl(self) -> float:
        return sum(p.unrealised_pnl for p in self.positions.values())

    @property
    def total_pnl(self) -> float:
        return self.net_asset_value - self.initial_capital

    @property
    def total_return_pct(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return self.total_pnl / self.initial_capital

    @property
    def cash_pct(self) -> float:
        nav = self.net_asset_value
        return self.cash / nav if nav > 0 else 1.0

    @property
    def invested_pct(self) -> float:
        return 1.0 - self.cash_pct

    @property
    def num_positions(self) -> int:
        return len(self.positions)

    @property
    def drawdown(self) -> float:
        """Current drawdown from peak NAV."""
        peak = max(self.peak_nav, self.net_asset_value)
        if peak == 0:
            return 0.0
        return (self.net_asset_value - peak) / peak

    def position_weight(self, ticker: str) -> float:
        """Weight of a position in the portfolio (0-1)."""
        if ticker not in self.positions:
            return 0.0
        nav = self.net_asset_value
        return self.positions[ticker].market_value / nav if nav > 0 else 0.0

    def sector_weights(self) -> Dict[str, float]:
        """Sector concentration map."""
        weights: Dict[str, float] = {}
        nav = self.net_asset_value
        for pos in self.positions.values():
            sector = pos.sector or "Unknown"
            weights[sector] = weights.get(sector, 0.0) + pos.market_value / nav
        return weights

    def largest_positions(self, n: int = 5) -> List[Tuple[str, float]]:
        nav = self.net_asset_value
        weights = [(t, p.market_value / nav) for t, p in self.positions.items()]
        return sorted(weights, key=lambda x: x[1], reverse=True)[:n]

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update all position prices from a market snapshot."""
        for ticker, pos in self.positions.items():
            if ticker in prices:
                pos.current_price = prices[ticker]
        self.timestamp = datetime.now()
        self.peak_nav = max(self.peak_nav, self.net_asset_value)

    def summary(self) -> str:
        lines = [
            f"Portfolio: {self.portfolio_id}",
            f"NAV       : ${self.net_asset_value:>14,.2f}",
            f"Cash      : ${self.cash:>14,.2f}  ({self.cash_pct:.1%})",
            f"Invested  : ${self.total_market_value:>14,.2f}  ({self.invested_pct:.1%})",
            f"Total P&L : ${self.total_pnl:>+14,.2f}  ({self.total_return_pct:+.2%})",
            f"Drawdown  : {self.drawdown:.2%}",
            f"Positions : {self.num_positions}",
            "─" * 48,
        ]
        for ticker, weight in self.largest_positions(n=10):
            pos = self.positions[ticker]
            lines.append(
                f"  {ticker:<6} {weight:>6.1%}  "
                f"${pos.market_value:>10,.0f}  "
                f"{pos.unrealised_pnl_pct:>+7.2%}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "portfolio_id": self.portfolio_id,
            "timestamp": self.timestamp.isoformat(),
            "net_asset_value": self.net_asset_value,
            "cash": self.cash,
            "total_market_value": self.total_market_value,
            "total_pnl": self.total_pnl,
            "total_return_pct": self.total_return_pct,
            "drawdown": self.drawdown,
            "num_positions": self.num_positions,
            "positions": {t: p.to_dict() for t, p in self.positions.items()},
        }


# ─────────────────────────────────────────────────────────────────────────────
# Trades
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    """An executed trade record — immutable audit trail."""
    trade_id:       str
    ticker:         str
    direction:      Direction
    order_type:     OrderType
    shares:         float
    price:          float
    commission:     float
    timestamp:      datetime
    agent_name:     str              # Which agent initiated the trade
    reasoning:      str              # LLM reasoning that led to this trade
    signal_score:   Optional[float] = None
    slippage_bps:   Optional[float] = None  # Actual slippage in basis points

    @property
    def gross_value(self) -> float:
        return self.shares * self.price

    @property
    def net_value(self) -> float:
        return self.gross_value + self.commission

    @property
    def signed_value(self) -> float:
        """Positive = cash out (buy), Negative = cash in (sell)."""
        if self.direction == Direction.LONG:
            return self.net_value
        return -self.net_value

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "ticker": self.ticker,
            "direction": self.direction.value,
            "order_type": self.order_type.value,
            "shares": self.shares,
            "price": self.price,
            "gross_value": self.gross_value,
            "commission": self.commission,
            "timestamp": self.timestamp.isoformat(),
            "agent_name": self.agent_name,
            "reasoning": self.reasoning,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Signals
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Signal:
    """
    An alpha signal from any model (XGBoost, GARCH, NLP, factor model, etc.)

    All signals are normalised to a common format so the Portfolio Manager
    Agent can reason across heterogeneous signal sources.
    """
    ticker:       str
    source:       str            # "xgboost", "nlp_sentiment", "garch", "heston", etc.
    signal_type:  str            # "alpha", "sentiment", "vol_forecast", "regime", etc.
    value:        float          # Raw signal value (interpretation depends on signal_type)
    strength:     SignalStrength
    confidence:   float          # 0.0 to 1.0
    timestamp:    datetime
    horizon_days: int = 5        # Signal horizon (5 = 5-day forward return)
    metadata:     dict = field(default_factory=dict)
    notes:        str = ""

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")

    @property
    def is_bullish(self) -> bool:
        return self.strength in (
            SignalStrength.STRONG_BUY, SignalStrength.BUY, SignalStrength.WEAK_BUY
        )

    @property
    def is_bearish(self) -> bool:
        return self.strength in (
            SignalStrength.STRONG_SELL, SignalStrength.SELL, SignalStrength.WEAK_SELL
        )

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "source": self.source,
            "signal_type": self.signal_type,
            "value": self.value,
            "strength": self.strength.value,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "horizon_days": self.horizon_days,
            "notes": self.notes,
        }

    def __repr__(self):
        return (
            f"Signal({self.ticker} | {self.source} | "
            f"{self.strength.value} | val={self.value:.4f} | "
            f"conf={self.confidence:.0%})"
        )


@dataclass
class SignalBundle:
    """
    Collection of signals for a single ticker from multiple sources.

    The PM Agent receives a SignalBundle and synthesises across sources.
    """
    ticker:    str
    signals:   List[Signal]
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def sources(self) -> List[str]:
        return list({s.source for s in self.signals})

    def by_source(self, source: str) -> Optional[Signal]:
        for s in self.signals:
            if s.source == source:
                return s
        return None

    def weighted_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Compute a weighted aggregate signal score.

        Weights map source names to importance (default: equal weighting).
        Normalises SignalStrength to [-1, 1] scale.
        """
        strength_map = {
            SignalStrength.STRONG_BUY:  1.0,
            SignalStrength.BUY:         0.6,
            SignalStrength.WEAK_BUY:    0.3,
            SignalStrength.NEUTRAL:     0.0,
            SignalStrength.WEAK_SELL:  -0.3,
            SignalStrength.SELL:       -0.6,
            SignalStrength.STRONG_SELL:-1.0,
        }
        total_weight = 0.0
        weighted_sum = 0.0
        for sig in self.signals:
            w = (weights or {}).get(sig.source, 1.0) * sig.confidence
            weighted_sum += strength_map[sig.strength] * w
            total_weight += w

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def summary(self) -> str:
        lines = [f"SignalBundle for {self.ticker} ({len(self.signals)} signals):"]
        for s in self.signals:
            lines.append(
                f"  [{s.source:<20}] {s.strength.value:<12} "
                f"val={s.value:>+8.4f}  conf={s.confidence:.0%}"
            )
        score = self.weighted_score()
        lines.append(f"  Aggregate score: {score:+.3f}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Risk Metrics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RiskMetrics:
    """
    Portfolio-level risk measurements.

    Computed by the Risk Engine and consumed by the PM Agent
    to make risk-adjusted allocation decisions.
    """
    timestamp:             datetime

    # Volatility
    portfolio_vol_annual:  float           # Annualised portfolio volatility
    portfolio_vol_daily:   float           # Daily portfolio volatility

    # VaR / Expected Shortfall
    var_95_1d:             float           # 1-day 95% VaR (dollar loss)
    var_99_1d:             float           # 1-day 99% VaR
    cvar_95_1d:            float           # Expected Shortfall at 95%

    # Performance
    sharpe_ratio:          Optional[float] = None
    sortino_ratio:         Optional[float] = None
    calmar_ratio:          Optional[float] = None
    max_drawdown:          Optional[float] = None
    current_drawdown:      Optional[float] = None

    # Exposures
    net_market_exposure:   float = 0.0     # Net long - short (fraction of NAV)
    gross_exposure:        float = 0.0     # Sum of |positions| (fraction of NAV)
    beta_to_spy:           Optional[float] = None

    # Factor exposures (from your factor model)
    factor_exposures:      Dict[str, float] = field(default_factory=dict)

    @property
    def var_95_pct(self) -> float:
        """VaR as fraction of NAV (requires knowing NAV externally)."""
        return self.var_95_1d  # Caller should divide by NAV

    def risk_budget_used(self, nav: float, max_var_pct: float = 0.02) -> float:
        """Fraction of risk budget consumed (0-1, where 1 = limit)."""
        if nav == 0 or max_var_pct == 0:
            return 0.0
        return (self.var_95_1d / nav) / max_var_pct

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "portfolio_vol_annual": self.portfolio_vol_annual,
            "portfolio_vol_daily": self.portfolio_vol_daily,
            "var_95_1d": self.var_95_1d,
            "var_99_1d": self.var_99_1d,
            "cvar_95_1d": self.cvar_95_1d,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "current_drawdown": self.current_drawdown,
            "beta_to_spy": self.beta_to_spy,
        }

    def summary(self) -> str:
        lines = [
            "=== Risk Metrics ===",
            f"  Portfolio Vol (annual) : {self.portfolio_vol_annual:.2%}",
            f"  VaR 95% 1-day          : ${self.var_95_1d:>10,.0f}",
            f"  VaR 99% 1-day          : ${self.var_99_1d:>10,.0f}",
            f"  CVaR 95% 1-day         : ${self.cvar_95_1d:>10,.0f}",
        ]
        if self.sharpe_ratio is not None:
            lines.append(f"  Sharpe Ratio           : {self.sharpe_ratio:>10.2f}")
        if self.max_drawdown is not None:
            lines.append(f"  Max Drawdown           : {self.max_drawdown:>10.2%}")
        if self.beta_to_spy is not None:
            lines.append(f"  Beta to SPY            : {self.beta_to_spy:>10.2f}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Agent Decision
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentDecision:
    """
    A structured allocation decision from an LLM agent.

    This is the output of the Portfolio Manager Agent after reasoning
    through signals, risk metrics, and portfolio state.

    Used for:
      - Audit trail (every decision is recorded)
      - Execution engine (converts decisions to trades)
      - Performance attribution (did the agent's reasoning prove correct?)
    """
    decision_id:    str
    agent_name:     str
    ticker:         str
    recommendation: Literal["BUY", "SELL", "HOLD", "PASS"]
    conviction:     Conviction
    target_weight:  float           # Target portfolio weight (0.0 to 1.0)
    current_weight: float           # Current portfolio weight
    weight_delta:   float           # How much to buy (+) or sell (-)

    # LLM reasoning (the full text output)
    reasoning:      str
    key_factors:    List[str]       # Bullet-point factors
    risks:          List[str]       # Key risks identified

    # Signals that informed the decision
    signals_used:   List[Signal]

    # Risk assessment
    estimated_var:  Optional[float] = None
    estimated_vol:  Optional[float] = None

    timestamp:      datetime = field(default_factory=datetime.now)
    llm_model:      str = ""
    llm_cost_usd:   float = 0.0     # API cost for this decision
    latency_ms:     float = 0.0     # Time taken

    @property
    def is_actionable(self) -> bool:
        """Is there a trade to execute?"""
        return (
            self.recommendation in ("BUY", "SELL")
            and abs(self.weight_delta) > 0.005  # At least 0.5% change
        )

    @property
    def dollar_value(self, nav: float = 1_000_000) -> float:
        return abs(self.weight_delta) * nav

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "agent_name": self.agent_name,
            "ticker": self.ticker,
            "recommendation": self.recommendation,
            "conviction": self.conviction.value,
            "target_weight": self.target_weight,
            "current_weight": self.current_weight,
            "weight_delta": self.weight_delta,
            "reasoning": self.reasoning,
            "key_factors": self.key_factors,
            "risks": self.risks,
            "timestamp": self.timestamp.isoformat(),
            "llm_model": self.llm_model,
            "llm_cost_usd": self.llm_cost_usd,
            "latency_ms": self.latency_ms,
        }

    def __repr__(self):
        return (
            f"AgentDecision({self.ticker} {self.recommendation} "
            f"target={self.target_weight:.1%} delta={self.weight_delta:+.1%} "
            f"conviction={self.conviction.value})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Macro Context
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MacroContext:
    """
    Macro-economic context for the PM Agent.

    Built from FRED data, market indicators, and news sentiment.
    The PM Agent uses this to adjust allocations for the current regime.
    """
    timestamp:       datetime

    # Interest rates
    fed_funds_rate:  Optional[float] = None   # Current FFR
    us_10y_yield:    Optional[float] = None   # 10Y Treasury yield
    yield_curve_10y2y: Optional[float] = None  # 10Y-2Y spread (recession signal)

    # Market indicators
    vix:             Optional[float] = None   # CBOE Volatility Index
    put_call_ratio:  Optional[float] = None   # Options market sentiment
    credit_spread:   Optional[float] = None   # HY-IG spread (stress indicator)

    # Economic indicators
    unemployment:    Optional[float] = None
    cpi_yoy:         Optional[float] = None   # Inflation
    gdp_growth_qoq:  Optional[float] = None

    # Derived regime assessment
    regime:          Regime = Regime.UNKNOWN
    recession_prob:  float = 0.0              # 0-1 probability

    def describe(self) -> str:
        lines = [f"Macro Context [{self.timestamp:%Y-%m-%d}]"]
        if self.fed_funds_rate is not None:
            lines.append(f"  Fed Funds Rate : {self.fed_funds_rate:.2f}%")
        if self.us_10y_yield is not None:
            lines.append(f"  10Y Yield      : {self.us_10y_yield:.2f}%")
        if self.yield_curve_10y2y is not None:
            inverted = "⚠️ INVERTED" if self.yield_curve_10y2y < 0 else ""
            lines.append(f"  10Y-2Y Spread  : {self.yield_curve_10y2y:.2f}% {inverted}")
        if self.vix is not None:
            vol_regime = "HIGH" if self.vix > 30 else "ELEVATED" if self.vix > 20 else "LOW"
            lines.append(f"  VIX            : {self.vix:.1f} ({vol_regime})")
        if self.cpi_yoy is not None:
            lines.append(f"  CPI YoY        : {self.cpi_yoy:.1f}%")
        lines.append(f"  Regime         : {self.regime.value}")
        lines.append(f"  Recession Prob : {self.recession_prob:.0%}")
        return "\n".join(lines)


if __name__ == "__main__":
    # Smoke test all models
    from datetime import datetime, date

    print("Testing data models...")

    bar = OHLCVBar("AAPL", datetime.now(), 195.0, 197.0, 194.0, 196.5, 55_000_000)
    print(f"OHLCVBar: {bar.ticker} close={bar.close} mid={bar.mid:.2f}")

    pos = Position("AAPL", Direction.LONG, 100, 180.0, 196.5, entry_date=date.today())
    print(f"Position: {pos}")

    port = Portfolio("PORT_001", cash=500_000, initial_capital=1_000_000)
    port.positions["AAPL"] = pos
    print(f"Portfolio NAV: ${port.net_asset_value:,.0f}")
    print(f"AAPL weight: {port.position_weight('AAPL'):.2%}")

    sig = Signal("AAPL", "xgboost", "alpha", 0.082, SignalStrength.BUY, 0.75, datetime.now())
    print(f"Signal: {sig}")

    print("\n✅ All data models initialised correctly")
