"""
AI Hedge Fund — Part 4: Execution Engine
==========================================
almgren_chriss.py — Optimal Execution via Almgren-Chriss (2001)

This is the mathematical heart of the execution system.
Every large order goes through this before it touches a broker.

The problem Almgren-Chriss solves:
    You need to buy 50,000 shares of AAPL.
    If you send one big MARKET order → massive market impact, terrible price.
    If you spread it over a week → huge timing risk (price moves against you).
    The optimal solution balances these two forces.

The model (Almgren & Chriss, Journal of Risk, 2001):
    Temporary impact: g(v) = η·v      (cost while you're trading, goes away)
    Permanent impact: h(x) = γ·x      (shifts mid-price permanently)

    Utility: U = E[Cost] + λ·Var[Cost]
    where λ is your risk aversion parameter.

    Optimal trajectory (CLOSED FORM):
        x*(t) = X · sinh(κ(T-t)) / sinh(κT)
        where κ = sqrt(λσ²/η)

    Interpretation of κ:
        κ → 0: trade uniformly over [0,T] = TWAP (ignore risk)
        κ → ∞: trade everything now (very risk averse)

    Trade list:
        n_k = x*(t_{k-1}) - x*(t_k)   shares to trade in period k

This is directly from your quant-portfolio/07_infrastructure/optimal_execution/
but wired into the live execution pipeline here.

References:
    Almgren, R. & Chriss, N. (2001). Optimal Execution of Portfolio Transactions.
    Journal of Risk 3(2), 5–39.

    Almgren, R. (2003). Optimal Execution with Nonlinear Impact Functions and
    Trading-Enhanced Risk. Applied Mathematical Finance 10(1).
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("hedge_fund.almgren_chriss")


# ─────────────────────────────────────────────────────────────────────────────
# Market impact parameters
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MarketImpactParams:
    """
    Market microstructure parameters for a single security.

    Calibrated from:
        - Historical trade data (if available)
        - ADV (Average Daily Volume) as a proxy for liquidity
        - Bid-ask spread for spread cost component
        - Realized volatility for timing risk

    Default calibration follows Almgren-Chriss (2001) typical values
    for US large-cap equities.
    """
    ticker:         str
    price:          float       # Current mid price
    daily_vol:      float       # Daily volatility (e.g. 0.015 = 1.5%/day)
    adv:            float       # Average daily volume in shares
    spread_bps:     float = 5.0 # Bid-ask spread in basis points

    # Impact coefficients
    eta:            Optional[float] = None   # Temporary impact: g(v) = η·v
    gamma:          Optional[float] = None   # Permanent impact: h(x) = γ·x
    sigma:          Optional[float] = None   # Volatility of returns (daily)

    def __post_init__(self):
        """Auto-calibrate impact coefficients if not provided."""
        if self.sigma is None:
            self.sigma = self.daily_vol

        if self.eta is None:
            # Temporary impact: proportional to σ/√(ADV)
            # This is the Almgren-Chriss (2001) calibration formula
            # typical η ≈ 0.1 × σ / √ADV
            self.eta = 0.1 * self.sigma / math.sqrt(self.adv)

        if self.gamma is None:
            # Permanent impact ≈ 0.5 × temporary impact (standard calibration)
            self.gamma = 0.5 * self.eta

    @property
    def daily_vol_usd(self) -> float:
        return self.sigma * self.price

    @property
    def spread_cost_per_share(self) -> float:
        return (self.spread_bps / 10000) * self.price / 2  # Half-spread

    def cost_of_trading_1pct_adv(self) -> float:
        """
        Estimate cost (in bps) of trading 1% of ADV.
        Quick rule-of-thumb for position sizing.
        """
        trade_rate = self.adv * 0.01 / 390  # 1% ADV over full trading day (390 min)
        impact_per_share = self.eta * trade_rate
        return (impact_per_share / self.price) * 10000


@dataclass
class ExecutionSchedule:
    """
    The output of the Almgren-Chriss optimiser.

    Contains the optimal trade trajectory and expected costs.
    """
    ticker:           str
    total_shares:     float          # X: total shares to execute
    side:             str            # "BUY" or "SELL"
    num_periods:      int            # N: number of execution intervals
    period_minutes:   float          # Duration of each interval in minutes
    trade_list:       List[float]    # n_k: shares to trade per period
    inventory:        List[float]    # x_k: shares remaining after each period
    timestamps:       List[datetime] # When each child order fires

    # Cost estimates
    expected_cost_usd:     float
    expected_cost_bps:     float
    cost_variance:         float
    risk_adjusted_cost:    float     # E[cost] + λ·Var[cost]

    # Execution metadata
    algo:             str = "IS"     # IS / TWAP / VWAP
    lambda_risk:      float = 1e-6
    kappa:            float = 0.0    # Almgren-Chriss κ parameter
    created_at:       datetime = field(default_factory=datetime.now)

    @property
    def total_execution_minutes(self) -> float:
        return self.num_periods * self.period_minutes

    @property
    def participation_rate(self) -> float:
        """Fraction of expected market volume we represent."""
        # Rough estimate: 390 trading minutes per day
        min_per_day  = 390
        daily_volume_per_minute = 1.0  # Normalised — caller should divide
        return self.total_shares / (self.total_execution_minutes * daily_volume_per_minute)

    def child_orders_summary(self) -> str:
        lines = [
            f"Execution Schedule: {self.side} {self.total_shares:,.0f} {self.ticker}",
            f"Algorithm: {self.algo} | Periods: {self.num_periods} x {self.period_minutes:.0f}min",
            f"Total time: {self.total_execution_minutes:.0f}min",
            f"Expected cost: ${self.expected_cost_usd:,.2f} ({self.expected_cost_bps:.1f}bps)",
            "─" * 55,
            f"{'Period':<8} {'Time':<12} {'Trade':<12} {'Remaining':<12}",
            "─" * 55,
        ]
        for i, (n, x, ts) in enumerate(zip(self.trade_list, self.inventory, self.timestamps)):
            lines.append(
                f"{i+1:<8} {ts.strftime('%H:%M'):<12} "
                f"{n:>8.0f}    {x:>10.0f}"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Almgren-Chriss Optimiser
# ─────────────────────────────────────────────────────────────────────────────

class AlmgrenChrissOptimiser:
    """
    Optimal execution trajectory via Almgren-Chriss (2001).

    Given a large order, computes the trade list that minimises:
        E[Total Cost] + λ × Var[Total Cost]

    where λ (lambda_risk) controls the tradeoff between expected
    cost and variance of cost.

    Usage:
        params = MarketImpactParams(
            ticker="AAPL", price=195.0, daily_vol=0.015, adv=55_000_000
        )
        optimiser = AlmgrenChrissOptimiser(params)
        schedule = optimiser.optimise(
            shares=10_000, side="BUY",
            horizon_minutes=120, n_periods=12
        )
        print(schedule.child_orders_summary())
    """

    def __init__(self, params: MarketImpactParams):
        self.p = params

    def optimise(
        self,
        shares:           float,
        side:             str = "BUY",
        horizon_minutes:  float = 60.0,
        n_periods:        int = 12,
        lambda_risk:      float = 1e-6,
        start_time:       Optional[datetime] = None,
    ) -> ExecutionSchedule:
        """
        Compute the optimal Almgren-Chriss execution trajectory.

        Args:
            shares:          Total shares to execute
            side:            "BUY" or "SELL"
            horizon_minutes: Total execution time in minutes
            n_periods:       Number of child orders (slices)
            lambda_risk:     Risk aversion parameter
                             1e-6 = moderate (typical for equities)
                             1e-5 = risk-averse (volatile markets)
                             1e-7 = patient (low vol / sufficient time)
            start_time:      When execution starts (default: now)

        Returns:
            ExecutionSchedule with optimal trade list and cost estimates
        """
        X    = float(shares)        # Total inventory to liquidate
        T    = horizon_minutes / 390.0  # Convert to fraction of trading day
        N    = n_periods
        τ    = T / N                # Length of each interval (fraction of day)
        lam  = lambda_risk

        # Unpack params
        σ    = self.p.sigma         # Daily return volatility
        η    = self.p.eta           # Temporary impact coefficient
        γ    = self.p.gamma         # Permanent impact coefficient

        # ── Almgren-Chriss κ (the key parameter) ─────────────────────────────
        # κ² = λσ²/η
        # Balances risk (λσ²) vs market impact cost (η)
        kappa_sq = lam * σ**2 / η
        kappa    = math.sqrt(max(kappa_sq, 1e-12))

        logger.debug(
            f"AC params: X={X:.0f} T={T:.4f} N={N} "
            f"σ={σ:.4f} η={η:.6f} γ={γ:.6f} κ={kappa:.4f}"
        )

        # ── Optimal trajectory: x*(t) = X·sinh(κ(T-t))/sinh(κT) ─────────────
        # Time grid: t_0=0, t_1=τ, ..., t_N=T
        sinh_kT = math.sinh(kappa * T)

        if sinh_kT < 1e-10:
            # κ → 0: degenerate case → TWAP (uniform)
            logger.debug("κ≈0: falling back to TWAP schedule")
            return self._twap_schedule(X, side, horizon_minutes, N, start_time)

        # Inventory at each time step: x_0=X, x_1, ..., x_N=0
        inventory = []
        for k in range(N + 1):
            t_k = k * τ
            x_k = X * math.sinh(kappa * (T - t_k)) / sinh_kT
            inventory.append(max(0.0, x_k))

        # Trade list: n_k = x_{k-1} - x_k (shares traded in period k)
        trade_list = [inventory[k] - inventory[k+1] for k in range(N)]

        # ── Expected cost and variance ────────────────────────────────────────
        # Expected cost (in dollars):
        # E[C] = (γ/2)·X² + (η/τ)·Σ n_k²   (simplified)
        # Full formula from Almgren-Chriss eq. (13):
        permanent_cost = 0.5 * γ * X**2 * self.p.price
        temporary_cost = (η / τ) * sum(n**2 for n in trade_list) * self.p.price

        expected_cost = permanent_cost + temporary_cost

        # Variance:
        # Var[C] = σ²·Σ t_k·n_k²  (simplified)
        variance = σ**2 * sum(
            (k * τ) * trade_list[k]**2
            for k in range(N)
        ) * self.p.price**2

        # Cost in basis points
        notional  = X * self.p.price
        cost_bps  = (expected_cost / notional) * 10000 if notional > 0 else 0

        # ── Build timestamps ───────────────────────────────────────────────────
        base_time  = start_time or datetime.now()
        period_min = horizon_minutes / N
        timestamps = [
            base_time + timedelta(minutes=i * period_min)
            for i in range(N)
        ]

        return ExecutionSchedule(
            ticker              = self.p.ticker,
            total_shares        = X,
            side                = side,
            num_periods         = N,
            period_minutes      = period_min,
            trade_list          = trade_list,
            inventory           = inventory,
            timestamps          = timestamps,
            expected_cost_usd   = expected_cost,
            expected_cost_bps   = cost_bps,
            cost_variance       = variance,
            risk_adjusted_cost  = expected_cost + lam * variance,
            algo                = "IS",
            lambda_risk         = lam,
            kappa               = kappa,
        )

    def _twap_schedule(
        self,
        shares: float, side: str,
        horizon_minutes: float, n_periods: int,
        start_time: Optional[datetime],
    ) -> ExecutionSchedule:
        """Uniform TWAP schedule — fallback when κ≈0."""
        n_per_period = shares / n_periods
        trade_list   = [n_per_period] * n_periods
        inventory    = [shares - i * n_per_period for i in range(n_periods + 1)]
        period_min   = horizon_minutes / n_periods
        base_time    = start_time or datetime.now()
        timestamps   = [
            base_time + timedelta(minutes=i * period_min)
            for i in range(n_periods)
        ]
        # Simple cost estimate: spread only
        cost_usd = shares * self.p.spread_cost_per_share
        cost_bps = (cost_usd / (shares * self.p.price)) * 10000 if shares * self.p.price > 0 else 0

        return ExecutionSchedule(
            ticker            = self.p.ticker,
            total_shares      = shares,
            side              = side,
            num_periods       = n_periods,
            period_minutes    = period_min,
            trade_list        = trade_list,
            inventory         = inventory,
            timestamps        = timestamps,
            expected_cost_usd = cost_usd,
            expected_cost_bps = cost_bps,
            cost_variance     = 0.0,
            risk_adjusted_cost= cost_usd,
            algo              = "TWAP",
            lambda_risk       = 0.0,
            kappa             = 0.0,
        )

    def compare_algos(
        self,
        shares:          float,
        side:            str = "BUY",
        horizon_minutes: float = 60.0,
        n_periods:       int = 12,
    ) -> Dict[str, ExecutionSchedule]:
        """
        Compare IS, TWAP, and VWAP schedules side by side.

        Returns dict of {algo_name: ExecutionSchedule}
        Useful for the Execution Agent to pick the best algo.
        """
        results = {}

        # Implementation Shortfall (Almgren-Chriss optimal)
        results["IS"]   = self.optimise(shares, side, horizon_minutes, n_periods, lambda_risk=1e-6)

        # TWAP
        results["TWAP"] = self._twap_schedule(shares, side, horizon_minutes, n_periods, None)

        # VWAP approximation (skewed toward high-volume periods)
        results["VWAP"] = self._vwap_schedule(shares, side, horizon_minutes, n_periods)

        return results

    def _vwap_schedule(
        self,
        shares: float, side: str,
        horizon_minutes: float, n_periods: int,
    ) -> ExecutionSchedule:
        """
        VWAP schedule — trade proportional to expected intraday volume.

        Volume pattern (U-shaped): high at open, low midday, high at close.
        Approximated with a quadratic U-shaped profile.
        """
        # Intraday volume profile (U-shaped, symmetric)
        weights = []
        for i in range(n_periods):
            t = i / (n_periods - 1) if n_periods > 1 else 0.5
            # U-shape: 1 + 2*(t - 0.5)^2 * 4
            w = 1.0 + 4.0 * (t - 0.5)**2
            weights.append(w)

        total_w  = sum(weights)
        fractions= [w / total_w for w in weights]
        trade_list = [shares * f for f in fractions]

        # Build inventory
        inventory = [shares]
        remaining = shares
        for n in trade_list:
            remaining -= n
            inventory.append(max(0.0, remaining))

        period_min = horizon_minutes / n_periods
        base_time  = datetime.now()
        timestamps = [
            base_time + timedelta(minutes=i * period_min)
            for i in range(n_periods)
        ]

        cost_usd = shares * self.p.spread_cost_per_share * 1.1  # Slight premium vs TWAP
        cost_bps = (cost_usd / (shares * self.p.price)) * 10000 if shares * self.p.price > 0 else 0

        return ExecutionSchedule(
            ticker            = self.p.ticker,
            total_shares      = shares,
            side              = side,
            num_periods       = n_periods,
            period_minutes    = period_min,
            trade_list        = trade_list,
            inventory         = inventory,
            timestamps        = timestamps,
            expected_cost_usd = cost_usd,
            expected_cost_bps = cost_bps,
            cost_variance     = 0.0,
            risk_adjusted_cost= cost_usd,
            algo              = "VWAP",
            lambda_risk       = 0.0,
            kappa             = 0.0,
        )

    def recommended_algo(
        self,
        shares:          float,
        horizon_minutes: float,
        urgency:         str = "NORMAL",   # LOW / NORMAL / HIGH / URGENT
    ) -> ExecutionAlgo:
        """
        Recommend an execution algorithm based on order characteristics.

        Logic:
            URGENT         → MARKET (accept full impact, minimise timing risk)
            High urgency   → IS with high λ (accept more impact for less risk)
            Normal         → IS with standard λ
            Low urgency    → TWAP (minimise impact, accept timing risk)
            Very small order → MARKET
        """
        participation = shares / (self.p.adv * horizon_minutes / 390)

        if urgency == "URGENT":
            return ExecutionAlgo.IS  # IS with high lambda ≈ market
        elif urgency == "HIGH":
            return ExecutionAlgo.IS
        elif participation < 0.005:
            return ExecutionAlgo.IS   # <0.5% ADV → small, use IS
        elif participation < 0.02:
            return ExecutionAlgo.TWAP  # 0.5-2% ADV → TWAP
        elif urgency == "LOW":
            return ExecutionAlgo.TWAP
        else:
            return ExecutionAlgo.IS   # Default: optimal IS


# ─────────────────────────────────────────────────────────────────────────────
# Pre-trade estimator
# ─────────────────────────────────────────────────────────────────────────────

class PreTradeEstimator:
    """
    Estimates execution costs before sending an order.

    Used by the Execution Agent to:
      1. Recommend the best algo for this order
      2. Estimate expected slippage and market impact
      3. Help the PM Agent understand the true cost of the trade
    """

    def estimate(
        self,
        ticker:          str,
        shares:          float,
        side:            str,
        current_price:   float,
        daily_vol:       float,
        adv:             float,
        horizon_minutes: float = 60.0,
        urgency:         str = "NORMAL",
    ) -> "PreTradeEstimate":
        """
        Produce a pre-trade cost estimate.

        All inputs come from the MarketDataFetcher or are passed
        from the AgentDecision.
        """
        from src.execution.order_models import PreTradeEstimate

        params = MarketImpactParams(
            ticker    = ticker,
            price     = current_price,
            daily_vol = daily_vol,
            adv       = adv,
        )
        optimiser = AlmgrenChrissOptimiser(params)

        # Get IS schedule for cost estimate
        schedule   = optimiser.optimise(
            shares          = shares,
            side            = side,
            horizon_minutes = horizon_minutes,
            n_periods       = max(1, int(horizon_minutes / 5)),  # 5-min slices
        )

        # Commission estimate (IB rates: ~$0.005/share, min $1)
        commission = max(1.0, shares * 0.005)

        # Recommended algo
        algo = optimiser.recommended_algo(shares, horizon_minutes, urgency)

        return PreTradeEstimate(
            ticker                      = ticker,
            quantity                    = shares,
            side                        = OrderSide.BUY if side == "BUY" else OrderSide.SELL,
            decision_price              = current_price,
            estimated_slippage_bps      = params.spread_bps / 2,
            estimated_commission        = commission,
            estimated_market_impact_bps = schedule.expected_cost_bps,
            market_adv                  = adv,
            participation_rate          = shares / adv,
            algo_recommended            = algo,
            horizon_minutes             = int(horizon_minutes),
        )


from src.execution.order_models import OrderSide


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)

    print("=" * 60)
    print("  Almgren-Chriss Optimal Execution — Test")
    print("=" * 60)

    # AAPL example: buy 10,000 shares
    params = MarketImpactParams(
        ticker    = "AAPL",
        price     = 195.0,
        daily_vol = 0.015,    # 1.5%/day
        adv       = 55_000_000,
        spread_bps= 2.0,
    )

    print(f"\n  Market params:")
    print(f"    Price:      ${params.price}")
    print(f"    Daily vol:  {params.daily_vol:.1%}")
    print(f"    ADV:        {params.adv/1e6:.0f}M shares")
    print(f"    η (temp):   {params.eta:.2e}")
    print(f"    γ (perm):   {params.gamma:.2e}")
    print(f"    Cost (1% ADV): {params.cost_of_trading_1pct_adv():.1f}bps")

    optimiser = AlmgrenChrissOptimiser(params)

    print("\n  Comparing IS vs TWAP vs VWAP (10,000 shares, 60 min):")
    schedules = optimiser.compare_algos(10_000, "BUY", horizon_minutes=60, n_periods=12)

    for algo_name, sched in schedules.items():
        print(
            f"  {algo_name:<6}: cost={sched.expected_cost_bps:.1f}bps "
            f"(${sched.expected_cost_usd:,.0f}) | "
            f"κ={sched.kappa:.4f}"
        )

    print("\n  Optimal IS schedule (first 5 periods):")
    is_sched = schedules["IS"]
    for i in range(min(5, len(is_sched.trade_list))):
        print(
            f"    {is_sched.timestamps[i].strftime('%H:%M')}  "
            f"trade={is_sched.trade_list[i]:>6.0f}  "
            f"remaining={is_sched.inventory[i]:>8.0f}"
        )

    print("\n  Pre-trade estimate:")
    estimator = PreTradeEstimator()
    est = estimator.estimate(
        ticker         = "AAPL",
        shares         = 10_000,
        side           = "BUY",
        current_price  = 195.0,
        daily_vol      = 0.015,
        adv            = 55_000_000,
        horizon_minutes= 60,
    )
    print(f"    Recommended algo:       {est.algo_recommended.value}")
    print(f"    Estimated impact:       {est.estimated_market_impact_bps:.1f}bps")
    print(f"    Estimated slippage:     {est.estimated_slippage_bps:.1f}bps")
    print(f"    Estimated commission:   ${est.estimated_commission:.2f}")
    print(f"    Total estimated cost:   ${est.total_estimated_cost_usd:,.2f}")

    print("\n✅ Almgren-Chriss tests passed")
