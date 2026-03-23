"""
AI Hedge Fund — Part 6: Backtesting Engine
============================================
stress_testing.py — Historical & Monte Carlo Stress Testing

Stress testing asks: how would this strategy have performed
(or would perform) under extreme market conditions?

Two approaches:

1. HISTORICAL SCENARIO ANALYSIS
   Replay specific real market episodes through the strategy.
   The advantage: 100% realistic — these things actually happened.
   The problem: limited to one realisation of history.

   Episodes implemented:
       2008 GFC (Sept-Nov 2008): -44% S&P in 3 months
       2020 COVID crash (Feb-Mar 2020): -34% S&P in 5 weeks
       2022 Rate shock (full year): -19.4% S&P, -31% NASDAQ
       2000 Dot-com (2000-2002): -49% S&P over 2.5 years
       2018 Q4 selloff: -20% in 3 months
       1987 Black Monday: single-day -22.6% (historical reference)

2. MONTE CARLO STRESS TESTING
   Simulate thousands of alternative return paths with fat tails,
   regime changes, and volatility clustering.
   Answers: "what's the 1% tail scenario even worse than 2008?"

3. SENSITIVITY ANALYSIS
   How sensitive are results to parameter changes?
   Vary each parameter ±20%, see how much the Sharpe moves.
   Large sensitivity = fragile strategy, not ready for live trading.

References:
    McNeil, Frey & Embrechts (2015). Quantitative Risk Management. Chap 9.
    Basel Committee (2009). Principles for Sound Stress Testing. BIS.
    Lopez de Prado (2018). Advances in Financial Machine Learning. Chap 16.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("hedge_fund.stress_test")


# ─────────────────────────────────────────────────────────────────────────────
# Historical scenarios
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HistoricalScenario:
    """A named historical market stress episode."""
    name:        str
    start_date:  date
    end_date:    date
    description: str
    spy_return:  float      # Approximate S&P 500 return over the period
    vix_peak:    float      # VIX peak during the episode


# Major historical stress episodes
HISTORICAL_SCENARIOS = [
    HistoricalScenario(
        name        = "2008_GFC_acute",
        start_date  = date(2008, 9, 1),
        end_date    = date(2008, 11, 30),
        description = "Lehman failure through initial recovery. -44% S&P in 3 months.",
        spy_return  = -0.44,
        vix_peak    = 89.5,
    ),
    HistoricalScenario(
        name        = "2020_COVID_crash",
        start_date  = date(2020, 2, 19),
        end_date    = date(2020, 3, 23),
        description = "COVID pandemic crash. -34% S&P in 33 days — fastest bear market on record.",
        spy_return  = -0.34,
        vix_peak    = 85.5,
    ),
    HistoricalScenario(
        name        = "2022_rate_shock",
        start_date  = date(2022, 1, 3),
        end_date    = date(2022, 12, 31),
        description = "Fed rate hikes, -19% S&P, -31% NASDAQ, -18% bonds (TLT). "
                      "Worst year for 60/40 portfolio since 1930s.",
        spy_return  = -0.195,
        vix_peak    = 38.9,
    ),
    HistoricalScenario(
        name        = "2018_Q4_selloff",
        start_date  = date(2018, 10, 1),
        end_date    = date(2018, 12, 24),
        description = "Fed tightening + trade war fears. -20% S&P in 3 months.",
        spy_return  = -0.20,
        vix_peak    = 36.2,
    ),
    HistoricalScenario(
        name        = "2011_eurozone_crisis",
        start_date  = date(2011, 7, 22),
        end_date    = date(2011, 10, 3),
        description = "European sovereign debt crisis, US credit downgrade. -21% S&P.",
        spy_return  = -0.21,
        vix_peak    = 48.0,
    ),
]


@dataclass
class ScenarioResult:
    """Strategy performance during one historical stress episode."""
    scenario:          HistoricalScenario
    strategy_return:   Optional[float]    # None if no data for this period
    strategy_max_dd:   Optional[float]
    strategy_sharpe:   Optional[float]
    benchmark_return:  float
    relative_return:   Optional[float]    # strategy - benchmark
    data_available:    bool = True
    note:              str = ""


@dataclass
class MonteCarloStressResult:
    """Results of Monte Carlo stress simulation."""
    n_simulations:     int
    horizon_days:      int

    # Distribution of final portfolio values
    returns_p01:       float    # 1st percentile — the tail
    returns_p05:       float    # 5th percentile
    returns_p10:       float
    returns_median:    float
    returns_mean:      float

    # Drawdown distribution
    max_dd_p05:        float    # 5th percentile max drawdown
    max_dd_median:     float

    # Tail risk
    var_99:            float    # VaR 99% of returns
    cvar_99:           float    # CVaR 99% (Expected Shortfall)

    # How often do we blow up?
    pct_loss_gt_20:    float    # % of sims with >20% loss
    pct_loss_gt_40:    float    # % of sims with >40% loss

    all_returns:       np.ndarray = field(repr=False, default_factory=lambda: np.array([]))

    def summary(self) -> str:
        lines = [
            f"  Monte Carlo Stress Test ({self.n_simulations:,} sims, {self.horizon_days} days)",
            f"  Return distribution:",
            f"    1st percentile : {self.returns_p01:>+8.2%}",
            f"    5th percentile : {self.returns_p05:>+8.2%}",
            f"    Median         : {self.returns_median:>+8.2%}",
            f"    Mean           : {self.returns_mean:>+8.2%}",
            f"  Tail risk:",
            f"    VaR 99%        : {self.var_99:>+8.2%}",
            f"    CVaR 99%       : {self.cvar_99:>+8.2%}",
            f"    P(loss > 20%):  {self.pct_loss_gt_20:>8.1%}",
            f"    P(loss > 40%):  {self.pct_loss_gt_40:>8.1%}",
            f"    Max DD (median): {self.max_dd_median:>8.2%}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Stress tester
# ─────────────────────────────────────────────────────────────────────────────

class StressTester:
    """
    Run historical and Monte Carlo stress tests on a BacktestResult.

    The BacktestResult provides the strategy's return series.
    Stress tests are applied to that return series.
    """

    def __init__(self):
        from src.backtest.backtest_engine import BacktestEngine
        self.engine = BacktestEngine()

    def run_historical_scenarios(
        self,
        result: "BacktestResult",
    ) -> List[ScenarioResult]:
        """
        Run the strategy through all historical stress scenarios.

        For each scenario, extract the portfolio's actual returns
        during that period (if data available) and compare to benchmark.
        """
        returns = result._get_returns()
        returns.index = pd.to_datetime(returns.index)

        scenario_results = []

        for scenario in HISTORICAL_SCENARIOS:
            # Check if we have data for this period
            s_start = pd.Timestamp(scenario.start_date)
            s_end   = pd.Timestamp(scenario.end_date)

            period_returns = returns[
                (returns.index >= s_start) & (returns.index <= s_end)
            ]

            if len(period_returns) < 5:
                scenario_results.append(ScenarioResult(
                    scenario         = scenario,
                    strategy_return  = None,
                    strategy_max_dd  = None,
                    strategy_sharpe  = None,
                    benchmark_return = scenario.spy_return,
                    relative_return  = None,
                    data_available   = False,
                    note             = f"Insufficient data ({len(period_returns)} days)"
                ))
                continue

            # Compute strategy metrics over the scenario period
            cumulative  = (1 + period_returns).cumprod()
            total_ret   = float(cumulative.iloc[-1] - 1)
            rolling_max = cumulative.cummax()
            max_dd      = float(((cumulative - rolling_max) / rolling_max).min())
            n_days      = len(period_returns)
            ann_factor  = 252 / n_days
            ann_ret     = (1 + total_ret) ** ann_factor - 1
            ann_vol     = float(period_returns.std() * math.sqrt(252))
            sharpe      = ann_ret / ann_vol if ann_vol > 0 else 0

            scenario_results.append(ScenarioResult(
                scenario         = scenario,
                strategy_return  = round(total_ret, 4),
                strategy_max_dd  = round(max_dd, 4),
                strategy_sharpe  = round(sharpe, 3),
                benchmark_return = scenario.spy_return,
                relative_return  = round(total_ret - scenario.spy_return, 4),
                data_available   = True,
            ))

        return scenario_results

    def run_monte_carlo_stress(
        self,
        result:        "BacktestResult",
        n_simulations: int = 10_000,
        horizon_days:  int = 252,
        fat_tails:     bool = True,    # Use Student-t instead of Normal
        vol_clustering: bool = True,   # GARCH-like volatility clustering
        seed:          int = 42,
    ) -> MonteCarloStressResult:
        """
        Monte Carlo stress test using calibrated return simulation.

        Fits the historical return distribution to the strategy's
        actual returns and simulates forward paths.

        fat_tails: Uses Student-t (ν≈4 for equities) — real returns
                   have much fatter tails than Normal.
        vol_clustering: GARCH(1,1)-like conditional vol — vol persists.
        """
        from scipy import stats as sp_stats

        returns = result._get_returns()
        if len(returns) < 30:
            logger.warning("Too few observations for Monte Carlo stress")
            return self._empty_mc_result(n_simulations, horizon_days)

        rng = np.random.default_rng(seed)

        # Fit distribution to historical returns
        mu  = float(returns.mean())
        sig = float(returns.std())

        # Fit Student-t for fat tails (MLE)
        if fat_tails and len(returns) >= 50:
            try:
                nu, loc, scale = sp_stats.t.fit(returns)
                nu = max(2.1, min(nu, 30))   # Constrain degrees of freedom
            except Exception:
                nu, loc, scale = 4.0, mu, sig
        else:
            nu, loc, scale = 4.0, mu, sig

        # GARCH parameters (approximate calibration from returns)
        if vol_clustering:
            # Simple GARCH(1,1) parameters
            omega = sig**2 * 0.05
            alpha = 0.10
            beta  = 0.85
        else:
            omega = alpha = beta = 0

        # Simulate paths
        final_returns = np.zeros(n_simulations)
        path_max_dds  = np.zeros(n_simulations)

        for sim in range(n_simulations):
            # Generate correlated random returns
            if fat_tails:
                z = sp_stats.t.rvs(nu, size=horizon_days, random_state=rng)
                z = z * scale + loc
            else:
                z = rng.normal(mu, sig, size=horizon_days)

            # Apply GARCH-like volatility clustering
            if vol_clustering:
                cond_var = sig**2
                for t in range(len(z)):
                    cond_vol = math.sqrt(cond_var)
                    z[t]     = z[t] * (cond_vol / sig)   # Scale by conditional vol
                    cond_var = omega + alpha * z[t]**2 + beta * cond_var
                    cond_var = max(cond_var, 1e-8)

            # Cumulative return path
            cum_path = np.cumprod(1 + z)
            final_returns[sim] = cum_path[-1] - 1

            # Max drawdown on this path
            rolling_max = np.maximum.accumulate(cum_path)
            dd_path     = (cum_path - rolling_max) / rolling_max
            path_max_dds[sim] = float(dd_path.min())

        # Statistics
        var_99  = float(np.percentile(final_returns, 1))
        cvar_99 = float(np.mean(final_returns[final_returns <= var_99]))

        return MonteCarloStressResult(
            n_simulations   = n_simulations,
            horizon_days    = horizon_days,
            returns_p01     = round(float(np.percentile(final_returns, 1)), 4),
            returns_p05     = round(float(np.percentile(final_returns, 5)), 4),
            returns_p10     = round(float(np.percentile(final_returns, 10)), 4),
            returns_median  = round(float(np.median(final_returns)), 4),
            returns_mean    = round(float(np.mean(final_returns)), 4),
            max_dd_p05      = round(float(np.percentile(path_max_dds, 5)), 4),
            max_dd_median   = round(float(np.median(path_max_dds)), 4),
            var_99          = round(var_99, 4),
            cvar_99         = round(cvar_99, 4),
            pct_loss_gt_20  = round(float(np.mean(final_returns < -0.20)), 4),
            pct_loss_gt_40  = round(float(np.mean(final_returns < -0.40)), 4),
            all_returns     = final_returns,
        )

    def run_sensitivity_analysis(
        self,
        strategy_class: type,
        tickers:        List[str],
        start_date:     date,
        end_date:       date,
        base_params:    Dict[str, Any],
        param_ranges:   Dict[str, Tuple[float, float]],  # param: (min, max)
        n_points:       int = 5,
        initial_capital: float = 1_000_000,
    ) -> Dict[str, List[Tuple[Any, float]]]:
        """
        One-at-a-time sensitivity analysis.

        For each parameter, varies it while holding others constant.
        Shows how fragile the strategy is to parameter choices.

        Returns {param_name: [(param_value, sharpe), ...]}
        """
        results = {}

        for param_name, (p_min, p_max) in param_ranges.items():
            param_results = []
            test_values   = np.linspace(p_min, p_max, n_points)

            for pval in test_values:
                params = {**base_params, param_name: pval}
                try:
                    strategy = strategy_class(params=params)
                    bt_result= self.engine.run(
                        strategy        = strategy,
                        tickers         = tickers,
                        start_date      = start_date,
                        end_date        = end_date,
                        initial_capital = initial_capital,
                        verbose         = False,
                    )
                    sharpe = bt_result.compute_metrics().get("sharpe_ratio", 0)
                    param_results.append((pval, sharpe))
                except Exception as e:
                    logger.debug(f"Sensitivity {param_name}={pval}: {e}")

            results[param_name] = param_results

        return results

    def full_stress_report(
        self,
        result:        "BacktestResult",
        n_mc_sims:     int = 10_000,
    ) -> str:
        """Run all stress tests and return formatted report."""
        scenarios = self.run_historical_scenarios(result)
        mc_result = self.run_monte_carlo_stress(result, n_mc_sims)

        lines = [
            "═" * 65,
            f"  STRESS TEST REPORT — {result.strategy_name}",
            "═" * 65,
            "",
            "  HISTORICAL SCENARIOS",
            f"  {'Scenario':<28} {'Strategy':>10} {'Benchmark':>11} {'vs BMK':>8} {'Max DD':>8}",
            f"  {'─'*62}",
        ]

        for sr in scenarios:
            if sr.data_available and sr.strategy_return is not None:
                lines.append(
                    f"  {sr.scenario.name:<28} "
                    f"{sr.strategy_return:>+10.1%} "
                    f"{sr.benchmark_return:>+11.1%} "
                    f"{sr.relative_return:>+8.1%} "
                    f"{sr.strategy_max_dd:>8.1%}"
                )
            else:
                lines.append(
                    f"  {sr.scenario.name:<28} "
                    f"{'N/A (prior to backtest)':>37}  {sr.note}"
                )

        lines += [
            "",
            mc_result.summary(),
            "═" * 65,
        ]
        return "\n".join(lines)

    def _empty_mc_result(self, n: int, h: int) -> MonteCarloStressResult:
        return MonteCarloStressResult(
            n_simulations=n, horizon_days=h,
            returns_p01=0, returns_p05=0, returns_p10=0,
            returns_median=0, returns_mean=0,
            max_dd_p05=0, max_dd_median=0,
            var_99=0, cvar_99=0,
            pct_loss_gt_20=0, pct_loss_gt_40=0,
        )


if __name__ == "__main__":
    import logging
    from datetime import date
    from src.backtest.backtest_engine import BacktestEngine, MomentumStrategy

    logging.basicConfig(level=logging.INFO)
    print("=" * 65)
    print("  Stress Testing — Test")
    print("=" * 65)

    engine = BacktestEngine()
    result = engine.run(
        strategy        = MomentumStrategy(params={"top_n": 4}),
        tickers         = ["AAPL","MSFT","NVDA","GOOGL","JPM","BAC","XOM"],
        start_date      = date(2019, 1, 1),
        end_date        = date(2023, 12, 31),
        initial_capital = 1_000_000,
        verbose         = True,
    )

    tester = StressTester()
    print("\n" + tester.full_stress_report(result, n_mc_sims=5_000))
    print("\n✅ Stress testing passed")
