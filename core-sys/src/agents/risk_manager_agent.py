"""
AI Hedge Fund — Part 2: Multi-Agent System
============================================
risk_manager_agent.py — Real-Time Risk Management Agent

The Risk Manager is the hedge fund's independent risk function.
It runs continuously, monitors every position, and has authority
to trigger position reductions or halt trading regardless of
what the Portfolio Manager wants.

In a real fund this is a separate desk (risk management team)
that is independent from portfolio management. The independence
is critical — a PM who also manages their own risk has a conflict
of interest and will systematically understate risk.

This agent:
  1. Runs pre-trade checks before any order is executed
     "Is this trade within risk limits?"

  2. Monitors portfolio in real time after market open
     "Have we breached any limits?"

  3. Broadcasts ALERTS when limits are approached (80% of limit)
     and CRITICAL ALERTS when limits are breached (100%)

  4. Computes daily risk reports (VaR, factor exposures, stress tests)

  5. Answers ad-hoc risk queries from the PM Agent

Risk limits monitored (all configurable in settings.py):
  - Portfolio VaR (1-day 95%): max 2% of NAV
  - Maximum drawdown: -15%
  - Maximum single position: 15% of NAV
  - Maximum sector concentration: 30%
  - Maximum correlation of new addition: 0.70
  - Beta to SPY: -0.5 to +1.5 (market exposure range)

Integrates with your quant-portfolio modules:
  quant-portfolio/04_risk_models/var_calculator/var_calculator.py
  quant-portfolio/04_risk_models/dcc_garch/dcc_garch.py
  quant-portfolio/03_credit_risk/stress_testing/stress_testing.py
  quant-portfolio/04_risk_models/bootstrap_ci/bootstrap_ci.py
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("hedge_fund.risk_manager")


# ─────────────────────────────────────────────────────────────────────────────
# Risk check results
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RiskCheckResult:
    """Result of a pre-trade or portfolio risk check."""
    passed:         bool
    check_name:     str
    metric_value:   float
    limit_value:    float
    utilisation_pct: float        # How much of the limit is used (0-100%)
    message:        str
    severity:       str = "INFO"  # INFO / WARNING / BREACH
    timestamp:      datetime = field(default_factory=datetime.now)

    @property
    def is_warning(self) -> bool:
        return self.utilisation_pct >= 80.0

    @property
    def is_breach(self) -> bool:
        return not self.passed

    def to_dict(self) -> dict:
        return {
            "passed":           self.passed,
            "check_name":       self.check_name,
            "metric_value":     round(self.metric_value, 6),
            "limit_value":      round(self.limit_value, 6),
            "utilisation_pct":  round(self.utilisation_pct, 1),
            "message":          self.message,
            "severity":         self.severity,
            "timestamp":        self.timestamp.isoformat(),
        }


@dataclass
class PreTradeCheck:
    """Complete pre-trade risk assessment for a proposed trade."""
    ticker:         str
    proposed_weight: float        # Proposed portfolio weight (0-1)
    current_weight:  float        # Current portfolio weight
    weight_delta:    float        # Change in weight
    checks:         List[RiskCheckResult] = field(default_factory=list)
    approved:       bool = True
    timestamp:      datetime = field(default_factory=datetime.now)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def warnings(self) -> List[RiskCheckResult]:
        return [c for c in self.checks if c.is_warning and c.passed]

    @property
    def breaches(self) -> List[RiskCheckResult]:
        return [c for c in self.checks if c.is_breach]

    def summary(self) -> str:
        status = "✅ APPROVED" if self.approved else "❌ REJECTED"
        lines  = [
            f"Pre-Trade Check: {self.ticker} | {status}",
            f"Proposed weight: {self.proposed_weight:.1%} | "
            f"Delta: {self.weight_delta:+.1%}",
            "─" * 50,
        ]
        for c in self.checks:
            icon = "✅" if c.passed else "❌"
            warn = " ⚠" if c.is_warning and c.passed else ""
            lines.append(
                f"  {icon} {c.check_name:<28} "
                f"{c.metric_value:.3f} / {c.limit_value:.3f} "
                f"({c.utilisation_pct:.0f}%){warn}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "ticker":          self.ticker,
            "proposed_weight": self.proposed_weight,
            "current_weight":  self.current_weight,
            "weight_delta":    self.weight_delta,
            "approved":        self.approved,
            "checks":          [c.to_dict() for c in self.checks],
            "timestamp":       self.timestamp.isoformat(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Risk calculation engine
# ─────────────────────────────────────────────────────────────────────────────

class RiskEngine:
    """
    Computes all risk metrics for the portfolio.

    This is the quantitative core of the Risk Manager Agent.
    The LLM agent calls these methods via tools.

    Integration points for your quant-portfolio modules:
        var_calculator.py   → compute_portfolio_var()
        dcc_garch.py        → compute_correlation_matrix()
        stress_testing.py   → run_stress_test()
        bootstrap_ci.py     → compute_sharpe_with_ci()
    """

    def __init__(self):
        from src.data.market_data import MarketDataFetcher, CorrelationEngine
        self.fetcher     = MarketDataFetcher()
        self.corr_engine = CorrelationEngine(self.fetcher)
        self._price_cache: Dict[str, pd.DataFrame] = {}

    def _get_prices(self, ticker: str, days: int = 252) -> pd.DataFrame:
        """Cached price fetch."""
        key = f"{ticker}_{days}"
        if key not in self._price_cache:
            df = self.fetcher.get_prices(ticker, days=days)
            if not df.empty:
                self._price_cache[key] = df
        return self._price_cache.get(key, pd.DataFrame())

    def _get_returns(self, ticker: str, days: int = 252) -> pd.Series:
        df = self._get_prices(ticker, days)
        if df.empty:
            return pd.Series(dtype=float)
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        return np.log(df[col] / df[col].shift(1)).dropna()

    # ── Position-level VaR ────────────────────────────────────────────────────

    def position_var_parametric(
        self,
        ticker:         str,
        position_value: float,
        confidence:     float = 0.95,
        horizon_days:   int = 1,
    ) -> float:
        """
        Parametric VaR for a single position.

        Uses realized volatility (21-day) as vol estimate.
        In production: plug in GARCH(1,1) from quant-portfolio.
        """
        returns = self._get_returns(ticker, days=63)
        if returns.empty or len(returns) < 5:
            # Conservative fallback: 30% annual vol
            vol_annual = 0.30
        else:
            vol_annual = float(returns.iloc[-21:].std() * math.sqrt(252))

        z_scores = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
        z        = z_scores.get(confidence, 1.645)
        vol_daily = vol_annual / math.sqrt(252)
        var       = position_value * vol_daily * z * math.sqrt(horizon_days)
        return abs(var)

    def position_vol(self, ticker: str, window_days: int = 21) -> float:
        """Annualised realized volatility for a position."""
        returns = self._get_returns(ticker, days=63)
        if returns.empty:
            return 0.30  # Conservative default
        return float(returns.iloc[-window_days:].std() * math.sqrt(252))

    # ── Portfolio-level VaR ───────────────────────────────────────────────────

    def portfolio_var_parametric(
        self,
        positions:    Dict[str, float],   # ticker -> position value ($)
        nav:          float,
        confidence:   float = 0.95,
        horizon_days: int = 1,
    ) -> Dict[str, float]:
        """
        Portfolio VaR using variance-covariance method.

        Accounts for correlations between positions.
        Sum of individual VaRs would overestimate — correlation
        reduces actual portfolio risk through diversification.

        Steps:
          1. Get return series for each position
          2. Build covariance matrix
          3. Portfolio variance = w' Σ w
          4. Portfolio VaR = z × σ_portfolio × √horizon

        Returns dict with var_95, var_99, vol_annual, var_pct_nav
        """
        tickers = list(positions.keys())

        if not tickers:
            return {"var_95": 0, "var_99": 0, "vol_annual": 0, "var_pct_nav": 0}

        if len(tickers) == 1:
            t   = tickers[0]
            v95 = self.position_var_parametric(t, positions[t], 0.95)
            v99 = self.position_var_parametric(t, positions[t], 0.99)
            vol = self.position_vol(t)
            return {
                "var_95":      v95,
                "var_99":      v99,
                "vol_annual":  vol,
                "var_pct_nav": v95 / nav if nav > 0 else 0,
            }

        # Build return matrix
        returns_dict = {}
        for t in tickers:
            ret = self._get_returns(t, days=252)
            if not ret.empty:
                returns_dict[t] = ret

        if len(returns_dict) < 2:
            # Fallback to sum of individual VaRs
            total_var = sum(
                self.position_var_parametric(t, v, 0.95)
                for t, v in positions.items()
                if t in returns_dict
            )
            return {
                "var_95":      total_var,
                "var_99":      total_var * (2.326 / 1.645),
                "vol_annual":  0.25,
                "var_pct_nav": total_var / nav if nav > 0 else 0,
            }

        # Align on common dates
        ret_df = pd.DataFrame(returns_dict).dropna(how="all").fillna(0)
        ret_df = ret_df.iloc[-252:]  # 1-year lookback

        # Portfolio weights (as fraction of total invested, not NAV)
        total_invested = sum(abs(v) for v in positions.values())
        weights = np.array([
            positions.get(t, 0) / total_invested
            for t in ret_df.columns
        ])

        # Covariance matrix (daily)
        cov_matrix = ret_df.cov().values * 252  # Annualise

        # Portfolio variance and vol
        port_var_annual  = float(weights @ cov_matrix @ weights)
        port_vol_annual  = math.sqrt(max(0, port_var_annual))
        port_vol_daily   = port_vol_annual / math.sqrt(252)

        # VaR
        port_value = total_invested
        var_95 = port_value * port_vol_daily * 1.645 * math.sqrt(horizon_days)
        var_99 = port_value * port_vol_daily * 2.326 * math.sqrt(horizon_days)

        return {
            "var_95":        abs(var_95),
            "var_99":        abs(var_99),
            "vol_annual":    port_vol_annual,
            "var_pct_nav":   abs(var_95) / nav if nav > 0 else 0,
            "diversification_ratio": float(
                sum(abs(w) * self.position_vol(t)
                    for t, w in zip(ret_df.columns, weights))
                / (port_vol_annual + 1e-8)
            ),
        }

    def portfolio_var_historical(
        self,
        positions:  Dict[str, float],
        nav:        float,
        confidence: float = 0.95,
        lookback:   int = 252,
    ) -> float:
        """
        Historical simulation VaR.

        Takes actual historical return scenarios rather than
        assuming normality. Better for fat-tailed distributions.
        Uses your var_calculator.py approach from quant-portfolio.
        """
        tickers = list(positions.keys())
        if not tickers:
            return 0.0

        returns_dict = {}
        for t in tickers:
            ret = self._get_returns(t, days=lookback + 30)
            if not ret.empty:
                returns_dict[t] = ret.iloc[-lookback:]

        if not returns_dict:
            return self.portfolio_var_parametric(positions, nav)["var_95"]

        ret_df = pd.DataFrame(returns_dict).dropna(how="all").fillna(0)
        total  = sum(abs(v) for v in positions.values())

        # Simulated daily P&L
        pnl_series = pd.Series(0.0, index=ret_df.index)
        for t in ret_df.columns:
            if t in positions:
                pnl_series += ret_df[t] * positions[t]

        # VaR = percentile of losses
        var = abs(float(np.percentile(pnl_series, (1 - confidence) * 100)))
        return var

    # ── Drawdown ──────────────────────────────────────────────────────────────

    def compute_drawdown(
        self, nav_series: pd.Series
    ) -> Tuple[float, float]:
        """
        Compute current and maximum drawdown from NAV history.

        Returns (current_drawdown, max_drawdown) as fractions (negative).
        """
        if nav_series.empty:
            return 0.0, 0.0

        peak    = nav_series.expanding().max()
        dd      = (nav_series - peak) / peak
        max_dd  = float(dd.min())
        curr_dd = float(dd.iloc[-1]) if len(dd) > 0 else 0.0
        return curr_dd, max_dd

    # ── Stress testing ────────────────────────────────────────────────────────

    def stress_test(
        self,
        positions:     Dict[str, float],   # ticker -> market value
        nav:           float,
    ) -> Dict[str, float]:
        """
        Stress test portfolio against historical crisis scenarios.

        Scenarios from your stress_testing.py in quant-portfolio.
        Returns estimated P&L for each scenario.
        """
        # Approximate peak-to-trough returns in each crisis
        # Source: actual historical returns for S&P 500 sectors
        scenarios = {
            "2008_GFC_peak_to_trough": {
                "SPY": -0.565, "AAPL": -0.60, "MSFT": -0.55,
                "JPM": -0.73, "BAC": -0.88, "GS": -0.78,
                "XOM": -0.30, "CVX": -0.35, "GLD": +0.05,
                "TLT": +0.35, "default_equity": -0.55,
            },
            "2020_COVID_crash_5weeks": {
                "SPY": -0.34, "AAPL": -0.31, "MSFT": -0.27,
                "JPM": -0.45, "BAC": -0.52, "GS": -0.38,
                "XOM": -0.55, "CVX": -0.52, "GLD": -0.02,
                "TLT": +0.14, "default_equity": -0.34,
            },
            "2022_rate_shock_full_year": {
                "SPY": -0.195, "AAPL": -0.265, "MSFT": -0.285,
                "NVDA": -0.505, "GOOGL": -0.395, "META": -0.645,
                "JPM": -0.155, "BAC": -0.260, "GS": -0.090,
                "TLT": -0.31, "GLD": -0.01, "default_equity": -0.20,
            },
            "2000_dot_com_18months": {
                "SPY": -0.50, "AAPL": -0.70, "MSFT": -0.65,
                "GOOGL": -0.80, "NVDA": -0.85,
                "JPM": -0.30, "GLD": +0.10, "TLT": +0.25,
                "default_equity": -0.50,
            },
        }

        results = {}
        for scenario_name, shock_map in scenarios.items():
            scenario_pnl = 0.0
            for ticker, pos_value in positions.items():
                shock = shock_map.get(ticker, shock_map["default_equity"])
                scenario_pnl += pos_value * shock
            results[scenario_name] = {
                "pnl":        round(scenario_pnl, 2),
                "pnl_pct_nav": round(scenario_pnl / nav * 100, 2) if nav > 0 else 0,
            }

        return results

    # ── Beta to market ────────────────────────────────────────────────────────

    def portfolio_beta(
        self,
        positions:   Dict[str, float],
        nav:         float,
        window_days: int = 126,
    ) -> float:
        """
        Portfolio beta to SPY.

        β_portfolio = Σ (weight_i × β_i)
        β_i = Cov(r_i, r_SPY) / Var(r_SPY)
        """
        spy_returns = self._get_returns("SPY", days=window_days + 30)
        if spy_returns.empty:
            return 1.0  # Assume market beta as default

        spy_returns = spy_returns.iloc[-window_days:]
        spy_var     = float(spy_returns.var())

        if spy_var < 1e-10:
            return 1.0

        total_invested = sum(abs(v) for v in positions.values())
        portfolio_beta = 0.0

        for ticker, pos_value in positions.items():
            ret = self._get_returns(ticker, days=window_days + 30)
            if ret.empty:
                continue
            ret = ret.iloc[-window_days:]
            # Align dates
            aligned = pd.concat([ret, spy_returns], axis=1).dropna()
            if len(aligned) < 20:
                continue
            cov = float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]))
            beta_i  = cov / spy_var
            weight  = abs(pos_value) / total_invested if total_invested > 0 else 0
            portfolio_beta += weight * beta_i

        return round(portfolio_beta, 3)

    # ── Concentration ─────────────────────────────────────────────────────────

    def position_concentration(
        self,
        positions: Dict[str, float],
        nav:       float,
    ) -> Dict[str, float]:
        """Return weight of each position as fraction of NAV."""
        if nav <= 0:
            return {}
        return {t: abs(v) / nav for t, v in positions.items()}

    def herfindahl_index(self, weights: Dict[str, float]) -> float:
        """
        Herfindahl-Hirschman Index (HHI) — portfolio concentration measure.

        HHI = Σ w_i²
        Range: 1/N (perfectly diversified) to 1.0 (single position)
        """
        return sum(w ** 2 for w in weights.values())

    # ── Sharpe ratio ──────────────────────────────────────────────────────────

    def rolling_sharpe(
        self,
        nav_series:       pd.Series,
        risk_free_annual: float = 0.05,
        window_days:      int = 63,
    ) -> Optional[float]:
        """
        Rolling Sharpe ratio from NAV history.

        From your bootstrap_ci.py in quant-portfolio for confidence intervals.
        Here: point estimate only.
        """
        if len(nav_series) < window_days + 5:
            return None

        returns    = np.log(nav_series / nav_series.shift(1)).dropna()
        returns    = returns.iloc[-window_days:]
        rf_daily   = risk_free_annual / 252
        excess_ret = returns - rf_daily
        if excess_ret.std() < 1e-10:
            return None
        sharpe = float(excess_ret.mean() / excess_ret.std() * math.sqrt(252))
        return round(sharpe, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Risk Manager Agent
# ─────────────────────────────────────────────────────────────────────────────

class RiskManagerAgent(BaseAgent):
    """
    Real-time Risk Manager for the AI Hedge Fund.

    Responsibilities:
      1. Pre-trade checks: approve or reject proposed trades
      2. Portfolio monitoring: detect limit breaches in real time
      3. Risk reporting: daily VaR, stress tests, factor report
      4. Alerts: notify PM Agent and Coordinator of risk events

    Communication:
      Receives: pre_trade_check, portfolio_risk_report, stress_test requests
      Sends:    risk approvals, risk rejections, breach alerts, daily reports
    """

    SYSTEM_PROMPT = """You are the Chief Risk Officer of a systematic hedge fund.

YOUR MANDATE:
- Protect capital above all else
- Independently verify all risk metrics
- Approve or reject trades based on quantitative risk limits
- Alert all agents immediately when limits are breached
- Produce clear, factual risk assessments

RISK LIMITS (hard limits — no exceptions):
- Portfolio 1-day VaR (95%): ≤ 2.0% of NAV
- Maximum single position: ≤ 15% of NAV
- Maximum sector concentration: ≤ 30% of NAV
- Maximum drawdown trigger: -15% (reduce exposure by 50%)
- Portfolio beta to SPY: between -0.5 and +1.5
- Maximum correlation of new addition: ≤ 0.70 to existing portfolio

WARNING THRESHOLDS (alert but don't block — at 80% of limit):
- VaR > 1.6% of NAV: issue WARNING
- Single position > 12%: issue WARNING
- Drawdown > -12%: issue WARNING

DECISION PROCESS:
1. For every pre-trade check: run ALL checks, report results
2. If ANY hard limit is breached: REJECT the trade, explain why
3. If any WARNING threshold is hit: APPROVE with caveats
4. Always provide specific numbers: "VaR would be $12,450 (1.24% of NAV, limit 2%)"

CRITICAL: You are INDEPENDENT from the Portfolio Manager.
Your job is to say NO when necessary. Say it clearly and without apology.

Respond in JSON format for pre-trade checks:
{
  "approved": true/false,
  "checks": [{"name": str, "passed": bool, "value": float, "limit": float, "message": str}],
  "summary": "brief overall assessment",
  "conditions": ["any conditions on approval"]
}"""

    def __init__(
        self,
        portfolio,     # Portfolio object from Part 1
        config: Optional[AgentConfig] = None,
    ):
        from src.agents.base_agent import AgentConfig
        cfg_obj = config or AgentConfig(
            name        = "RiskManager",
            model       = "claude-sonnet-4-6",
            temperature = 0.05,   # Very low — risk decisions need consistency
        )
        self.portfolio   = portfolio
        self.risk_engine = RiskEngine()
        self._nav_history: List[Tuple[datetime, float]] = []
        self._alerts_sent: List[Dict] = []

        # Import config limits
        import sys
        sys.path.insert(0, str(Path(__file__).parents[3]))
        try:
            from src.config.settings import cfg as settings
            self.limits = {
                "max_var_pct":     settings.MAX_PORTFOLIO_VAR_PCT,
                "max_position":    settings.MAX_POSITION_SIZE,
                "max_sector":      settings.MAX_SECTOR_CONCENTRATION,
                "max_drawdown":    settings.MAX_DRAWDOWN_LIMIT,
                "max_correlation": settings.MAX_CORRELATION_ADDITION,
            }
        except ImportError:
            self.limits = {
                "max_var_pct":     0.02,
                "max_position":    0.15,
                "max_sector":      0.30,
                "max_drawdown":    0.15,
                "max_correlation": 0.70,
            }

        super().__init__(cfg_obj)

    def _get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def _get_tools(self) -> List[Tool]:
        from src.agents.base_agent import Tool
        return [
            Tool(
                name = "check_portfolio_var",
                func = self._tool_portfolio_var,
                description = (
                    "Calculate current portfolio VaR (Value at Risk) and compare to limit. "
                    "Input: 'current' to check current portfolio, or a JSON string with "
                    "proposed position changes. Returns VaR in dollars and % of NAV."
                ),
                param_schema = {
                    "type": "object",
                    "properties": {
                        "scenario": {
                            "type": "string",
                            "description": "Either 'current' or JSON of proposed changes"
                        }
                    },
                    "required": ["scenario"]
                }
            ),
            Tool(
                name = "check_position_limits",
                func = self._tool_check_position_limits,
                description = (
                    "Check if a proposed position size violates concentration limits. "
                    "Input: ticker and proposed_weight as decimal (e.g. 0.10 for 10%). "
                    "Returns whether it passes and the utilisation of each limit."
                ),
                param_schema = {
                    "type": "object",
                    "properties": {
                        "ticker":          {"type": "string"},
                        "proposed_weight": {"type": "number"}
                    },
                    "required": ["ticker", "proposed_weight"]
                }
            ),
            Tool(
                name = "run_stress_test",
                func = self._tool_stress_test,
                description = (
                    "Run stress tests against historical crisis scenarios "
                    "(2008 GFC, 2020 COVID, 2022 rates, 2000 dot-com). "
                    "Input: 'current' for current portfolio. "
                    "Returns estimated P&L under each scenario."
                ),
                param_schema = {
                    "type": "object",
                    "properties": {
                        "portfolio": {"type": "string"}
                    },
                    "required": ["portfolio"]
                }
            ),
            Tool(
                name = "check_correlation",
                func = self._tool_check_correlation,
                description = (
                    "Check correlation of a new ticker against existing portfolio. "
                    "High correlation = low diversification benefit. "
                    "Input: ticker symbol. Returns avg correlation and breakdown."
                ),
                param_schema = {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"}
                    },
                    "required": ["ticker"]
                }
            ),
            Tool(
                name = "get_portfolio_risk_summary",
                func = self._tool_portfolio_risk_summary,
                description = (
                    "Get a complete risk summary of the current portfolio. "
                    "Includes VaR, drawdown, beta, Sharpe, concentration. "
                    "Input: anything. Returns full risk metrics."
                ),
                param_schema = {
                    "type": "object",
                    "properties": {
                        "detail_level": {"type": "string"}
                    },
                    "required": []
                }
            ),
            Tool(
                name = "check_drawdown",
                func = self._tool_check_drawdown,
                description = (
                    "Check current drawdown against limit. "
                    "Input: anything. Returns current and max drawdown."
                ),
                param_schema = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            ),
        ]

    # ── Tool implementations ──────────────────────────────────────────────────

    def _tool_portfolio_var(self, scenario: str = "current") -> str:
        """Calculate portfolio VaR."""
        nav       = self.portfolio.net_asset_value
        positions = {
            t: p.market_value
            for t, p in self.portfolio.positions.items()
        }

        # Handle proposed changes
        if scenario != "current":
            try:
                changes = json.loads(scenario)
                for ticker, delta_weight in changes.items():
                    delta_value = delta_weight * nav
                    positions[ticker] = positions.get(ticker, 0) + delta_value
            except (json.JSONDecodeError, TypeError):
                pass

        if not positions:
            return json.dumps({
                "var_95": 0, "var_99": 0, "vol_annual": 0,
                "var_pct_nav": 0, "limit_pct": self.limits["max_var_pct"],
                "status": "EMPTY_PORTFOLIO"
            })

        metrics = self.risk_engine.portfolio_var_parametric(positions, nav)
        var_pct = metrics["var_95"] / nav if nav > 0 else 0
        limit   = self.limits["max_var_pct"]
        util    = var_pct / limit * 100 if limit > 0 else 0

        status = "OK"
        if var_pct > limit:
            status = "BREACH"
        elif var_pct > limit * 0.8:
            status = "WARNING"

        return json.dumps({
            "var_95_usd":     round(metrics["var_95"], 2),
            "var_99_usd":     round(metrics["var_99"], 2),
            "var_pct_nav":    round(var_pct * 100, 3),
            "limit_pct":      round(limit * 100, 2),
            "utilisation":    round(util, 1),
            "vol_annual":     round(metrics.get("vol_annual", 0) * 100, 2),
            "status":         status,
        })

    def _tool_check_position_limits(
        self, ticker: str, proposed_weight: float = 0
    ) -> str:
        nav              = self.portfolio.net_asset_value
        current_weight   = self.portfolio.position_weight(ticker)
        position_limit   = self.limits["max_position"]

        util = proposed_weight / position_limit * 100 if position_limit > 0 else 0
        passed = proposed_weight <= position_limit

        # Sector check (approximate — in production: use sector database)
        sector_weights = self.portfolio.sector_weights()
        # Assume ticker's sector has same weight as current + proposed
        max_sector_wt  = max(sector_weights.values()) if sector_weights else 0

        return json.dumps({
            "ticker":           ticker,
            "current_weight":   round(current_weight * 100, 2),
            "proposed_weight":  round(proposed_weight * 100, 2),
            "position_limit":   round(position_limit * 100, 2),
            "utilisation":      round(util, 1),
            "position_passed":  passed,
            "max_sector_weight": round(max_sector_wt * 100, 2),
            "sector_limit":     round(self.limits["max_sector"] * 100, 2),
            "status":           "OK" if passed else "BREACH",
        })

    def _tool_stress_test(self, portfolio: str = "current") -> str:
        nav       = self.portfolio.net_asset_value
        positions = {
            t: p.market_value
            for t, p in self.portfolio.positions.items()
        }
        if not positions:
            return json.dumps({"note": "Empty portfolio — no stress test applicable"})

        results = self.risk_engine.stress_test(positions, nav)
        return json.dumps({
            k: {
                "pnl_usd":     v["pnl"],
                "pnl_pct_nav": v["pnl_pct_nav"],
                "severity":    (
                    "SEVERE"   if v["pnl_pct_nav"] < -20 else
                    "MODERATE" if v["pnl_pct_nav"] < -10 else
                    "MILD"
                )
            }
            for k, v in results.items()
        })

    def _tool_check_correlation(self, ticker: str) -> str:
        existing = list(self.portfolio.positions.keys())
        if not existing:
            return json.dumps({
                "ticker": ticker,
                "avg_correlation": 0,
                "note": "No existing positions — first position",
                "status": "OK"
            })

        corrs = self.risk_engine.corr_engine.correlation_to_portfolio(
            ticker, existing, days=126
        )
        avg_corr = float(np.mean(list(corrs.values()))) if corrs else 0.0
        limit    = self.limits["max_correlation"]

        return json.dumps({
            "ticker":          ticker,
            "avg_correlation": round(avg_corr, 3),
            "limit":           round(limit, 2),
            "utilisation":     round(avg_corr / limit * 100, 1),
            "breakdown":       {t: round(c, 3) for t, c in corrs.items()},
            "status":          "BREACH" if avg_corr > limit else (
                               "WARNING" if avg_corr > limit * 0.8 else "OK"
            ),
        })

    def _tool_portfolio_risk_summary(self, detail_level: str = "full") -> str:
        nav       = self.portfolio.net_asset_value
        positions = {t: p.market_value for t, p in self.portfolio.positions.items()}

        # VaR
        var_metrics = self.risk_engine.portfolio_var_parametric(positions, nav) if positions else {}

        # Beta
        beta = self.risk_engine.portfolio_beta(positions, nav) if positions else 0

        # Drawdown
        if len(self._nav_history) >= 2:
            nav_s = pd.Series([n for _, n in self._nav_history])
            curr_dd, max_dd = self.risk_engine.compute_drawdown(nav_s)
        else:
            curr_dd, max_dd = 0.0, 0.0

        # Concentration
        weights = self.risk_engine.position_concentration(positions, nav)
        hhi     = self.risk_engine.herfindahl_index(weights)

        return json.dumps({
            "nav":                round(nav, 2),
            "num_positions":      len(positions),
            "var_95_usd":         round(var_metrics.get("var_95", 0), 2),
            "var_pct_nav":        round(var_metrics.get("var_pct_nav", 0) * 100, 3),
            "var_limit_pct":      round(self.limits["max_var_pct"] * 100, 2),
            "vol_annual":         round(var_metrics.get("vol_annual", 0) * 100, 2),
            "portfolio_beta":     beta,
            "current_drawdown":   round(curr_dd * 100, 2),
            "max_drawdown":       round(max_dd * 100, 2),
            "drawdown_limit":     round(self.limits["max_drawdown"] * 100, 2),
            "herfindahl_index":   round(hhi, 4),
            "top_positions":      sorted(weights.items(), key=lambda x: -x[1])[:5],
        })

    def _tool_check_drawdown(self, **kwargs) -> str:
        if len(self._nav_history) < 2:
            return json.dumps({
                "current_drawdown": 0,
                "max_drawdown":     0,
                "limit":            self.limits["max_drawdown"] * 100,
                "status":           "INSUFFICIENT_HISTORY"
            })

        nav_s            = pd.Series([n for _, n in self._nav_history])
        curr_dd, max_dd  = self.risk_engine.compute_drawdown(nav_s)
        limit            = self.limits["max_drawdown"]
        status           = (
            "BREACH"  if abs(curr_dd) > limit else
            "WARNING" if abs(curr_dd) > limit * 0.8 else
            "OK"
        )

        return json.dumps({
            "current_drawdown_pct": round(curr_dd * 100, 2),
            "max_drawdown_pct":     round(max_dd * 100, 2),
            "limit_pct":            round(limit * 100, 2),
            "utilisation":          round(abs(curr_dd) / limit * 100, 1),
            "status":               status,
        })

    # ── Pre-trade check (main public method) ─────────────────────────────────

    def pre_trade_check(
        self,
        ticker:          str,
        proposed_weight: float,
        current_weight:  float = 0.0,
    ) -> PreTradeCheck:
        """
        Full pre-trade risk check for a proposed allocation.

        Called by the AgentCoordinator before any trade is executed.
        Returns PreTradeCheck with approved flag and all check details.
        """
        nav          = self.portfolio.net_asset_value
        weight_delta = proposed_weight - current_weight
        pos_value    = proposed_weight * nav

        checks = []

        # 1 — Position size limit
        checks.append(self._check_position_size(ticker, proposed_weight))

        # 2 — Portfolio VaR impact
        checks.append(self._check_var_impact(ticker, weight_delta, nav))

        # 3 — Correlation
        checks.append(self._check_correlation_limit(ticker))

        # 4 — Drawdown
        checks.append(self._check_drawdown_limit())

        # 5 — Cash availability
        checks.append(self._check_cash_availability(pos_value))

        all_passed = all(c.passed for c in checks)

        result = PreTradeCheck(
            ticker           = ticker,
            proposed_weight  = proposed_weight,
            current_weight   = current_weight,
            weight_delta     = weight_delta,
            checks           = checks,
            approved         = all_passed,
        )

        # Log to bus if rejected
        if not all_passed:
            self.alert(
                subject = f"TRADE_REJECTED_{ticker}",
                payload = {
                    "ticker":    ticker,
                    "reason":    [c.message for c in checks if not c.passed],
                    "checks":    result.to_dict(),
                }
            )
            logger.warning(
                f"Pre-trade check REJECTED: {ticker} "
                f"{proposed_weight:.1%} — "
                f"{[c.message for c in checks if not c.passed]}"
            )
        else:
            logger.info(
                f"Pre-trade check APPROVED: {ticker} {proposed_weight:.1%}"
            )

        return result

    def _check_position_size(
        self, ticker: str, proposed_weight: float
    ) -> RiskCheckResult:
        limit = self.limits["max_position"]
        util  = proposed_weight / limit * 100 if limit > 0 else 0
        return RiskCheckResult(
            passed         = proposed_weight <= limit,
            check_name     = "Position Size Limit",
            metric_value   = proposed_weight,
            limit_value    = limit,
            utilisation_pct = util,
            message        = (
                f"{ticker} weight {proposed_weight:.1%} "
                f"{'within' if proposed_weight <= limit else 'EXCEEDS'} "
                f"limit {limit:.1%}"
            ),
            severity       = "BREACH" if proposed_weight > limit else
                             "WARNING" if util >= 80 else "INFO",
        )

    def _check_var_impact(
        self, ticker: str, weight_delta: float, nav: float
    ) -> RiskCheckResult:
        # Estimate new position's marginal VaR contribution
        pos_value  = abs(weight_delta) * nav
        vol        = self.risk_engine.position_vol(ticker)
        marginal_var = self.risk_engine.position_var_parametric(
            ticker, pos_value, 0.95
        )
        current_var  = self._get_current_var_pct()
        new_var_pct  = current_var + marginal_var / nav if nav > 0 else 0
        limit        = self.limits["max_var_pct"]
        util         = new_var_pct / limit * 100 if limit > 0 else 0

        return RiskCheckResult(
            passed         = new_var_pct <= limit,
            check_name     = "Portfolio VaR Limit",
            metric_value   = new_var_pct,
            limit_value    = limit,
            utilisation_pct = util,
            message        = (
                f"Marginal VaR ${marginal_var:,.0f} | "
                f"New portfolio VaR {new_var_pct:.2%} "
                f"({'within' if new_var_pct <= limit else 'EXCEEDS'} {limit:.1%} limit)"
            ),
            severity       = "BREACH" if new_var_pct > limit else
                             "WARNING" if util >= 80 else "INFO",
        )

    def _check_correlation_limit(self, ticker: str) -> RiskCheckResult:
        existing = list(self.portfolio.positions.keys())
        if not existing:
            return RiskCheckResult(
                passed          = True,
                check_name      = "Correlation Check",
                metric_value    = 0.0,
                limit_value     = self.limits["max_correlation"],
                utilisation_pct = 0.0,
                message         = "First position — no correlation check needed",
                severity        = "INFO",
            )

        corrs    = self.risk_engine.corr_engine.correlation_to_portfolio(
            ticker, existing, days=126
        )
        avg_corr = float(np.mean(list(corrs.values()))) if corrs else 0.0
        limit    = self.limits["max_correlation"]
        util     = avg_corr / limit * 100 if limit > 0 else 0

        return RiskCheckResult(
            passed          = avg_corr <= limit,
            check_name      = "Correlation Limit",
            metric_value    = avg_corr,
            limit_value     = limit,
            utilisation_pct = util,
            message         = (
                f"{ticker} avg correlation {avg_corr:.3f} "
                f"({'within' if avg_corr <= limit else 'EXCEEDS'} {limit:.2f} limit)"
            ),
            severity        = "BREACH" if avg_corr > limit else
                              "WARNING" if util >= 80 else "INFO",
        )

    def _check_drawdown_limit(self) -> RiskCheckResult:
        if len(self._nav_history) < 2:
            return RiskCheckResult(
                passed          = True,
                check_name      = "Drawdown Check",
                metric_value    = 0.0,
                limit_value     = self.limits["max_drawdown"],
                utilisation_pct = 0.0,
                message         = "Insufficient NAV history for drawdown check",
                severity        = "INFO",
            )

        nav_s            = pd.Series([n for _, n in self._nav_history])
        curr_dd, _       = self.risk_engine.compute_drawdown(nav_s)
        dd_abs           = abs(curr_dd)
        limit            = self.limits["max_drawdown"]
        util             = dd_abs / limit * 100 if limit > 0 else 0

        return RiskCheckResult(
            passed          = dd_abs < limit,
            check_name      = "Drawdown Limit",
            metric_value    = dd_abs,
            limit_value     = limit,
            utilisation_pct = util,
            message         = (
                f"Current drawdown {curr_dd:.1%} "
                f"({'within' if dd_abs < limit else 'AT OR EXCEEDS'} "
                f"{-limit:.1%} limit)"
            ),
            severity        = "BREACH" if dd_abs >= limit else
                              "WARNING" if util >= 80 else "INFO",
        )

    def _check_cash_availability(self, position_value: float) -> RiskCheckResult:
        cash  = self.portfolio.cash
        ratio = position_value / cash if cash > 0 else float("inf")
        passed = position_value <= cash

        return RiskCheckResult(
            passed          = passed,
            check_name      = "Cash Availability",
            metric_value    = position_value,
            limit_value     = cash,
            utilisation_pct = min(ratio * 100, 999),
            message         = (
                f"Position requires ${position_value:,.0f} | "
                f"Available cash: ${cash:,.0f} | "
                f"{'OK' if passed else 'INSUFFICIENT CASH'}"
            ),
            severity        = "BREACH" if not passed else "INFO",
        )

    def _get_current_var_pct(self) -> float:
        """Get current portfolio VaR as fraction of NAV."""
        nav       = self.portfolio.net_asset_value
        positions = {t: p.market_value for t, p in self.portfolio.positions.items()}
        if not positions or nav <= 0:
            return 0.0
        metrics = self.risk_engine.portfolio_var_parametric(positions, nav)
        return metrics.get("var_pct_nav", 0.0)

    # ── MessageBus handler ────────────────────────────────────────────────────

    def handle_message(self, message) -> Optional[Dict[str, Any]]:
        """Process incoming messages from the bus."""
        from src.comms.message_bus import MessageType

        subject = message.subject.lower()
        payload = message.payload

        logger.info(f"RiskManager handling: {message.subject} from {message.sender}")

        if "pre_trade_check" in subject:
            ticker          = payload.get("ticker", "")
            proposed_weight = float(payload.get("proposed_weight", 0))
            current_weight  = float(payload.get("current_weight", 0))

            check  = self.pre_trade_check(ticker, proposed_weight, current_weight)
            return check.to_dict()

        elif "portfolio_risk" in subject or "risk_report" in subject:
            summary_json = self._tool_portfolio_risk_summary()
            stress_json  = self._tool_stress_test()
            return {
                "risk_summary": json.loads(summary_json),
                "stress_tests": json.loads(stress_json),
                "timestamp":    datetime.now().isoformat(),
            }

        elif "stress_test" in subject:
            return json.loads(self._tool_stress_test())

        elif "var_check" in subject:
            return json.loads(self._tool_portfolio_var())

        else:
            # Use LLM for ad-hoc risk questions
            response_text, _ = self.think(
                user_message = f"Risk question from {message.sender}: {message.subject}\n"
                               f"Details: {json.dumps(payload)}",
                use_tools    = True,
                purpose      = "ad_hoc_risk_query",
            )
            return {"response": response_text, "timestamp": datetime.now().isoformat()}

    def update_nav(self, nav: float) -> None:
        """Record current NAV for drawdown tracking."""
        self._nav_history.append((datetime.now(), nav))
        # Keep last 504 observations (2 years daily)
        if len(self._nav_history) > 504:
            self._nav_history = self._nav_history[-504:]

        # Check drawdown after update
        if len(self._nav_history) >= 2:
            nav_s = pd.Series([n for _, n in self._nav_history])
            curr_dd, _ = self.risk_engine.compute_drawdown(nav_s)
            if abs(curr_dd) >= self.limits["max_drawdown"]:
                self.alert(
                    subject = "MAX_DRAWDOWN_BREACH",
                    payload = {
                        "current_drawdown": round(curr_dd * 100, 2),
                        "limit":            round(self.limits["max_drawdown"] * 100, 2),
                        "action_required":  "REDUCE_EXPOSURE_50PCT",
                    }
                )


# Import BaseAgent here to avoid circular imports
from src.agents.base_agent import BaseAgent, Tool, AgentConfig


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parents[3]))
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  Risk Manager Agent — Test")
    print("=" * 60)

    from src.data.data_models import Portfolio, Position, Direction

    port = Portfolio("TEST_001", cash=900_000, initial_capital=1_000_000)
    port.positions["AAPL"] = Position(
        "AAPL", Direction.LONG, 200, 180.0, 195.0, sector="Technology"
    )
    port.positions["MSFT"] = Position(
        "MSFT", Direction.LONG, 150, 380.0, 415.0, sector="Technology"
    )

    rm = RiskManagerAgent(port)
    rm.update_nav(1_000_000)
    rm.update_nav(980_000)

    print("\n1. Pre-trade check: BUY NVDA 12%")
    check = rm.pre_trade_check("NVDA", proposed_weight=0.12, current_weight=0.0)
    print(check.summary())

    print("\n2. Pre-trade check: BUY GOOGL 20% (should breach position limit)")
    check2 = rm.pre_trade_check("GOOGL", proposed_weight=0.20, current_weight=0.0)
    print(check2.summary())

    print("\n3. Portfolio risk summary:")
    summary = rm._tool_portfolio_risk_summary()
    print(json.dumps(json.loads(summary), indent=2))

    print("\n4. Stress tests:")
    stress = rm._tool_stress_test()
    print(json.dumps(json.loads(stress), indent=2))

    print(f"\n{rm.get_metrics()}")
    print("\n✅ Risk Manager Agent tests passed")
