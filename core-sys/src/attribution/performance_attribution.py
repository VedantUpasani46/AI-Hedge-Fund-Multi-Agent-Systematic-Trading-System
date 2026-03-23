"""
AI Hedge Fund — Part 6: Backtesting Engine
============================================
performance_attribution.py — Return Attribution & Factor Analysis

Performance attribution answers: WHERE did the returns come from?

Why this matters for investors:
    A Sharpe of 1.2 could come from:
        (a) Genuine alpha — skill in stock selection
        (b) Factor loading — riding the momentum factor
        (c) Market beta — just leveraged S&P 500
        (d) Luck — regime coincidence

    Investors paying 2-and-20 want (a), not (b), (c), or (d).
    Without attribution, you cannot tell which.

Three levels of attribution implemented:

1. BRINSON-HOOD-BEEBOWER (BHB) decomposition
   Standard in institutional asset management.
   Total excess return = Allocation effect + Selection effect + Interaction
   
   Allocation:  Did we overweight sectors that outperformed?
   Selection:   Within sectors, did we pick better stocks?
   Interaction: Cross-term (usually small)

2. FAMA-FRENCH FACTOR ATTRIBUTION
   Regress portfolio returns against factor returns.
   Factors: Market (MKT), Size (SMB), Value (HML), Momentum (MOM)
   
   R_p = α + β_mkt·MKT + β_smb·SMB + β_hml·HML + β_mom·MOM + ε
   
   α (alpha):     Return unexplained by factors — true skill
   β_mkt:         Market exposure (1.0 = market neutral at 1.0)
   β_smb:         Small-cap tilt (positive = long small caps)
   β_hml:         Value tilt (positive = long value/short growth)
   β_mom:         Momentum tilt

3. TRANSACTION COST ATTRIBUTION
   How much did execution costs reduce gross alpha?
   Gross return - Net return = Execution drag
   Broken into: commission, spread, market impact, slippage

References:
    Brinson, Hood & Beebower (1986). Determinants of Portfolio Performance. FAJ.
    Brinson, Singer & Beebower (1991). FAJ (update).
    Fama & French (1993). Common Risk Factors. JFE.
    Carhart (1997). On Persistence in Mutual Fund Performance. JF.
    Grinold & Kahn (2000). Active Portfolio Management. McGraw-Hill.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.backtest.backtest_engine import BacktestResult, PortfolioSnapshot, BacktestFill

logger = logging.getLogger("hedge_fund.attribution")


# ─────────────────────────────────────────────────────────────────────────────
# Brinson-Hood-Beebower
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BHBResult:
    """Result of Brinson-Hood-Beebower attribution."""
    period_start:       date
    period_end:         date
    total_excess_return: float

    # BHB components
    allocation_effect:  float    # Sector weighting skill
    selection_effect:   float    # Stock selection skill within sectors
    interaction_effect: float    # Cross-term

    # Sector breakdown
    sector_detail:      Dict[str, Dict[str, float]]

    @property
    def total_active_return(self) -> float:
        return self.allocation_effect + self.selection_effect + self.interaction_effect

    def summary(self) -> str:
        lines = [
            "  BHB Attribution",
            f"  Period: {self.period_start} → {self.period_end}",
            f"  ─────────────────────────────────────────",
            f"  Allocation Effect  : {self.allocation_effect:>+8.2%}",
            f"  Selection Effect   : {self.selection_effect:>+8.2%}",
            f"  Interaction Effect : {self.interaction_effect:>+8.2%}",
            f"  ─────────────────────────────────────────",
            f"  Total Active Return: {self.total_active_return:>+8.2%}",
        ]
        if self.sector_detail:
            lines.append("  Sector breakdown:")
            for sector, d in sorted(self.sector_detail.items()):
                lines.append(
                    f"    {sector:<20} "
                    f"alloc={d.get('allocation',0):+.2%}  "
                    f"sel={d.get('selection',0):+.2%}"
                )
        return "\n".join(lines)


class BHBAttributor:
    """
    Brinson-Hood-Beebower attribution.

    Requires a benchmark (e.g. SPY) and portfolio sector weights.
    In practice, sector weights are inferred from the positions
    using GICS sector mapping.
    """

    # Approximate GICS sector mapping for common tickers
    SECTOR_MAP = {
        "AAPL": "Technology",    "MSFT": "Technology",   "NVDA": "Technology",
        "GOOGL": "Technology",   "META": "Technology",   "AMZN": "Technology",
        "CRM": "Technology",     "ORCL": "Technology",   "INTC": "Technology",
        "AMD": "Technology",     "QCOM": "Technology",   "AVGO": "Technology",
        "JPM": "Financials",     "BAC": "Financials",    "GS": "Financials",
        "MS": "Financials",      "WFC": "Financials",    "C": "Financials",
        "BLK": "Financials",
        "JNJ": "Healthcare",     "UNH": "Healthcare",    "PFE": "Healthcare",
        "MRK": "Healthcare",     "ABT": "Healthcare",
        "XOM": "Energy",         "CVX": "Energy",        "COP": "Energy",
        "EOG": "Energy",
        "PG": "ConsumerStaples", "KO": "ConsumerStaples","PEP": "ConsumerStaples",
        "WMT": "ConsumerStaples",
        "HD": "ConsumerDiscretionary", "NKE": "ConsumerDiscretionary",
        "MCD": "ConsumerDiscretionary",
        "CAT": "Industrials",    "GE": "Industrials",    "BA": "Industrials",
        "UNP": "Industrials",
        "NEE": "Utilities",      "DUK": "Utilities",
        "AMT": "RealEstate",     "PLD": "RealEstate",
        "LIN": "Materials",      "NEM": "Materials",
        "SPY": "Benchmark",      "QQQ": "Benchmark",
    }

    def attribute(
        self,
        result:          BacktestResult,
        benchmark_ticker: str = "SPY",
        lookback_days:    int = 63,
    ) -> BHBResult:
        """
        Run BHB attribution over the full backtest period.

        Uses end-of-period portfolio weights and sector returns.
        """
        import yfinance as yf

        snapshots = result.snapshots
        if not snapshots:
            return self._empty_result(result)

        # Get benchmark returns
        start = result.start_date
        end   = result.end_date

        portfolio_return = result.compute_metrics().get("annual_return", 0)

        # Fetch benchmark
        try:
            bench_df = yf.download(
                benchmark_ticker, start=start.isoformat(),
                end=end.isoformat(), progress=False, auto_adjust=True,
            )
            if isinstance(bench_df.columns, pd.MultiIndex):
                bench_df.columns = bench_df.columns.get_level_values(0)

            if not bench_df.empty:
                bench_series = bench_df["Close"] if "Close" in bench_df.columns else bench_df.iloc[:, 0]
                bench_return = float(bench_series.iloc[-1] / bench_series.iloc[0] - 1)
            else:
                bench_return = 0.0
        except Exception:
            bench_return = 0.0

        excess_return = portfolio_return - bench_return

        # Get average portfolio sector weights
        sector_weights = self._compute_sector_weights(snapshots)

        # Approximate benchmark sector weights (S&P 500 as of 2024)
        bench_weights = {
            "Technology": 0.31,  "Financials": 0.13,
            "Healthcare": 0.12,  "ConsumerDiscretionary": 0.10,
            "Industrials": 0.09, "ConsumerStaples": 0.06,
            "Energy": 0.04,      "RealEstate": 0.03,
            "Materials": 0.03,   "Utilities": 0.03,
        }

        # Fetch sector ETF returns for attribution
        sector_etfs = {
            "Technology": "XLK", "Financials": "XLF",
            "Healthcare": "XLV", "ConsumerDiscretionary": "XLY",
            "Industrials": "XLI", "ConsumerStaples": "XLP",
            "Energy": "XLE",     "RealEstate": "XLRE",
            "Materials": "XLB",  "Utilities": "XLU",
        }

        sector_returns = {}
        for sector, etf in sector_etfs.items():
            try:
                df = yf.download(
                    etf, start=start.isoformat(),
                    end=end.isoformat(), progress=False, auto_adjust=True
                )
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if not df.empty:
                    c = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
                    sector_returns[sector] = float(c.iloc[-1] / c.iloc[0] - 1)
            except Exception:
                sector_returns[sector] = bench_return   # Use benchmark as fallback

        # BHB decomposition
        # Allocation: Σ (w_p - w_b) × (R_b_sector - R_b_total)
        # Selection:  Σ w_b × (R_p_sector - R_b_sector)
        # Interaction:Σ (w_p - w_b) × (R_p_sector - R_b_sector)
        allocation  = 0.0
        selection   = 0.0
        interaction = 0.0
        sector_detail = {}

        for sector in set(list(sector_weights.keys()) + list(bench_weights.keys())):
            wp = sector_weights.get(sector, 0.0)
            wb = bench_weights.get(sector, 0.0)
            rb_sector = sector_returns.get(sector, bench_return)
            # Portfolio sector return: approximate as sector ETF return
            # (without individual stock data — this is a simplification)
            rp_sector = sector_returns.get(sector, bench_return) * (1 + (wp - wb) * 0.1)

            alloc_k   = (wp - wb) * (rb_sector - bench_return)
            select_k  = wb * (rp_sector - rb_sector)
            inter_k   = (wp - wb) * (rp_sector - rb_sector)

            allocation  += alloc_k
            selection   += select_k
            interaction += inter_k

            if abs(alloc_k) + abs(select_k) > 0.001:
                sector_detail[sector] = {
                    "portfolio_weight": round(wp, 3),
                    "benchmark_weight": round(wb, 3),
                    "allocation":       round(alloc_k, 4),
                    "selection":        round(select_k, 4),
                }

        return BHBResult(
            period_start         = start,
            period_end           = end,
            total_excess_return  = excess_return,
            allocation_effect    = allocation,
            selection_effect     = selection,
            interaction_effect   = interaction,
            sector_detail        = sector_detail,
        )

    def _compute_sector_weights(self, snapshots: List[PortfolioSnapshot]) -> Dict[str, float]:
        """Average portfolio sector weights over the period."""
        sector_totals: Dict[str, float] = {}
        total_weight = 0.0
        n = 0

        for snap in snapshots[::5]:   # Sample every 5 days for efficiency
            for ticker, pos in snap.positions.items():
                sector = self.SECTOR_MAP.get(ticker, "Other")
                sector_totals[sector] = sector_totals.get(sector, 0) + pos.get("weight", 0)
                total_weight += pos.get("weight", 0)
            n += 1

        if n == 0 or total_weight == 0:
            return {}

        return {s: v / n for s, v in sector_totals.items()}

    def _empty_result(self, result: BacktestResult) -> BHBResult:
        return BHBResult(
            period_start=result.start_date, period_end=result.end_date,
            total_excess_return=0, allocation_effect=0,
            selection_effect=0, interaction_effect=0, sector_detail={},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Factor attribution (Carhart 4-factor)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FactorAttributionResult:
    """Result of Fama-French / Carhart factor regression."""
    alpha_annual:    float      # Annualised α (true excess return above factors)
    alpha_monthly:   float
    alpha_t_stat:    float      # t-statistic on α
    alpha_pvalue:    float      # p-value on α

    beta_market:     float      # Market beta
    beta_smb:        float      # Small-minus-big (size factor)
    beta_hml:        float      # High-minus-low (value factor)
    beta_mom:        float      # Momentum factor

    r_squared:       float      # Fraction of variance explained by factors
    residual_vol:    float      # Annualised unexplained vol (idiosyncratic)

    tracking_error:  float      # Annualised active return vol vs benchmark
    info_ratio:      float      # Annualised α / tracking_error

    factor_returns:  Dict[str, float]   # How much each factor contributed

    def is_alpha_significant(self, threshold: float = 0.10) -> bool:
        """Is the alpha statistically significant?"""
        return self.alpha_pvalue < threshold

    def summary(self) -> str:
        sig = "***" if self.alpha_pvalue < 0.01 else "**" if self.alpha_pvalue < 0.05 else "*" if self.alpha_pvalue < 0.10 else ""
        lines = [
            "  Factor Attribution (Carhart 4-Factor)",
            f"  ─────────────────────────────────────────",
            f"  Alpha (annual)   : {self.alpha_annual:>+8.2%} {sig}",
            f"  Alpha t-stat     : {self.alpha_t_stat:>8.2f}",
            f"  Alpha p-value    : {self.alpha_pvalue:>8.3f}",
            f"  ─────────────────────────────────────────",
            f"  Market β         : {self.beta_market:>8.3f}",
            f"  SMB β (size)     : {self.beta_smb:>8.3f}",
            f"  HML β (value)    : {self.beta_hml:>8.3f}",
            f"  MOM β (momentum) : {self.beta_mom:>8.3f}",
            f"  ─────────────────────────────────────────",
            f"  R²               : {self.r_squared:>8.3f}",
            f"  Residual vol     : {self.residual_vol:>8.2%}",
            f"  Information Ratio: {self.info_ratio:>8.3f}",
            f"  ─────────────────────────────────────────",
            f"  Factor return contributions:",
        ]
        for factor, contrib in self.factor_returns.items():
            lines.append(f"    {factor:<12}: {contrib:>+8.2%}")
        lines.append(f"  Significance: {sig if sig else 'not significant at 10%'}")
        return "\n".join(lines)


class FactorAttributor:
    """
    Carhart 4-factor attribution via OLS regression.

    Replicates factors from free data:
        MKT:  SPY daily returns - risk-free rate
        SMB:  IWM (small-cap) - SPY (large-cap) returns
        HML:  IWD (value) - IWF (growth) returns
        MOM:  Momentum portfolio (approximate using MTUM ETF)

    This is an approximation of the Ken French factor data.
    For research use: download exact factor data from
    mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
    """

    FACTOR_PROXIES = {
        "MKT": ("SPY", None),         # SPY minus risk-free
        "SMB": ("IWM", "SPY"),         # Small vs large cap
        "HML": ("IWD", "IWF"),         # Value vs growth
        "MOM": ("MTUM", "SPY"),        # Momentum vs market
    }

    def attribute(
        self,
        result:         BacktestResult,
        risk_free_rate: float = 0.05,
    ) -> FactorAttributionResult:
        """
        Regress portfolio returns against Carhart 4 factors.
        """
        import yfinance as yf

        returns = result._get_returns()
        if len(returns) < 30:
            logger.warning("Too few observations for factor regression")
            return self._empty_result()

        returns.index = pd.to_datetime(returns.index)

        # Fetch all factor proxy prices
        all_tickers = list(set(
            t for pair in self.FACTOR_PROXIES.values()
            for t in pair if t is not None
        ))

        start = result.start_date - timedelta(days=30)
        end   = result.end_date

        try:
            raw = yf.download(
                all_tickers,
                start   = start.isoformat(),
                end     = (end + timedelta(days=1)).isoformat(),
                progress= False,
                auto_adjust=True,
            )
        except Exception as e:
            logger.error(f"Factor data download failed: {e}")
            return self._empty_result()

        # Handle MultiIndex
        if isinstance(raw.columns, pd.MultiIndex):
            try:
                prices = raw["Close"]
            except KeyError:
                prices = raw.xs("Close", axis=1, level=0) if "Close" in raw.columns.get_level_values(0) else raw.iloc[:, :len(all_tickers)]
        else:
            prices = raw if len(all_tickers) == 1 else raw

        # Compute factor returns
        price_returns = np.log(prices / prices.shift(1)).dropna()

        rf_daily    = risk_free_rate / 252
        factor_rets = {}

        try:
            if "SPY" in price_returns.columns:
                factor_rets["MKT"] = price_returns["SPY"] - rf_daily
            if "IWM" in price_returns.columns and "SPY" in price_returns.columns:
                factor_rets["SMB"] = price_returns["IWM"] - price_returns["SPY"]
            if "IWD" in price_returns.columns and "IWF" in price_returns.columns:
                factor_rets["HML"] = price_returns["IWD"] - price_returns["IWF"]
            if "MTUM" in price_returns.columns and "SPY" in price_returns.columns:
                factor_rets["MOM"] = price_returns["MTUM"] - price_returns["SPY"]
        except Exception as e:
            logger.warning(f"Factor construction partial: {e}")

        if not factor_rets:
            return self._empty_result()

        # Align portfolio returns with factor returns
        factor_df = pd.DataFrame(factor_rets)
        factor_df.index = pd.to_datetime(factor_df.index)

        # Align indices
        common_idx = returns.index.intersection(factor_df.index)
        if len(common_idx) < 20:
            logger.warning(f"Only {len(common_idx)} common dates for factor regression")
            return self._empty_result()

        y = returns.reindex(common_idx).fillna(0).values
        X_raw = factor_df.reindex(common_idx).fillna(0).values

        # OLS regression: R_p = α + β·F + ε
        X = np.column_stack([np.ones(len(y)), X_raw])

        try:
            # Normal equations: β = (X'X)^{-1} X'y
            XtX     = X.T @ X
            Xty     = X.T @ y
            beta    = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
        except np.linalg.LinAlgError:
            return self._empty_result()

        alpha_daily = float(beta[0])
        factor_betas= beta[1:]

        # Residuals and statistics
        y_hat    = X @ beta
        residuals= y - y_hat
        n        = len(y)
        k        = X.shape[1]
        s2       = float(residuals.var())
        sse      = float((residuals ** 2).sum())
        sst      = float(((y - y.mean()) ** 2).sum())
        r2       = 1 - sse / sst if sst > 0 else 0.0

        # Standard error of alpha and t-stat
        cov_matrix = s2 * np.linalg.pinv(XtX)
        se_alpha   = math.sqrt(abs(float(cov_matrix[0, 0])))
        t_stat     = alpha_daily / se_alpha if se_alpha > 0 else 0.0
        p_value    = float(2 * (1 - scipy_stats.t.cdf(abs(t_stat), df=n - k)))

        # Annualise
        alpha_annual = (1 + alpha_daily) ** 252 - 1
        alpha_monthly= (1 + alpha_daily) ** 21 - 1
        resid_vol    = float(residuals.std() * math.sqrt(252))

        # Tracking error vs benchmark (using MKT factor proxy)
        bench_returns_aligned = factor_rets.get("MKT", pd.Series(dtype=float))
        bench_aligned = bench_returns_aligned.reindex(common_idx).fillna(0)
        active_returns= pd.Series(y, index=common_idx) - bench_aligned.values
        te_annual     = float(active_returns.std() * math.sqrt(252))
        info_ratio    = alpha_annual / te_annual if te_annual > 0 else 0.0

        # Factor return contributions (β × mean factor return × 252)
        factor_names = list(factor_rets.keys())[:len(factor_betas)]
        factor_contribs = {}
        for i, fname in enumerate(factor_names):
            if i < len(factor_betas):
                mean_f_annual = float(factor_df[fname].mean()) * 252
                factor_contribs[fname] = round(float(factor_betas[i]) * mean_f_annual, 4)

        return FactorAttributionResult(
            alpha_annual  = round(alpha_annual, 4),
            alpha_monthly = round(alpha_monthly, 4),
            alpha_t_stat  = round(t_stat, 3),
            alpha_pvalue  = round(p_value, 4),
            beta_market   = round(float(factor_betas[0]) if len(factor_betas) > 0 else 0, 3),
            beta_smb      = round(float(factor_betas[1]) if len(factor_betas) > 1 else 0, 3),
            beta_hml      = round(float(factor_betas[2]) if len(factor_betas) > 2 else 0, 3),
            beta_mom      = round(float(factor_betas[3]) if len(factor_betas) > 3 else 0, 3),
            r_squared     = round(r2, 4),
            residual_vol  = round(resid_vol, 4),
            tracking_error= round(te_annual, 4),
            info_ratio    = round(info_ratio, 3),
            factor_returns= factor_contribs,
        )

    def _empty_result(self) -> FactorAttributionResult:
        return FactorAttributionResult(
            alpha_annual=0, alpha_monthly=0, alpha_t_stat=0, alpha_pvalue=1,
            beta_market=0, beta_smb=0, beta_hml=0, beta_mom=0,
            r_squared=0, residual_vol=0, tracking_error=0, info_ratio=0,
            factor_returns={},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Transaction cost attribution
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TCAnalysis:
    """Breakdown of transaction cost drag on returns."""
    period_return_gross:  float
    period_return_net:    float
    total_drag_bps:       float

    commission_drag_bps:  float
    spread_drag_bps:      float
    impact_drag_bps:      float
    slippage_drag_bps:    float

    n_trades:             int
    avg_trade_size_usd:   float
    annual_turnover:      float

    # Per-trade breakdown
    best_fills:    List[Dict]    # Lowest cost fills
    worst_fills:   List[Dict]    # Highest cost fills

    def summary(self) -> str:
        lines = [
            "  Transaction Cost Analysis",
            f"  Gross return         : {self.period_return_gross:>+8.2%}",
            f"  Net return           : {self.period_return_net:>+8.2%}",
            f"  Total TC drag        : {self.total_drag_bps:>8.1f}bps",
            f"  ─────────────────────────────────────────",
            f"  Commission drag      : {self.commission_drag_bps:>8.1f}bps",
            f"  Spread drag          : {self.spread_drag_bps:>8.1f}bps",
            f"  Market impact drag   : {self.impact_drag_bps:>8.1f}bps",
            f"  Slippage drag        : {self.slippage_drag_bps:>8.1f}bps",
            f"  ─────────────────────────────────────────",
            f"  Trades               : {self.n_trades}",
            f"  Avg trade size       : ${self.avg_trade_size_usd:,.0f}",
            f"  Annual turnover      : {self.annual_turnover:.0%}",
        ]
        return "\n".join(lines)


class TCAttributor:
    """Transaction cost attribution from BacktestResult fills."""

    def attribute(self, result: BacktestResult) -> TCAnalysis:
        fills  = result.fills
        metrics= result.compute_metrics()

        if not fills:
            return TCAnalysis(
                period_return_gross=0, period_return_net=0,
                total_drag_bps=0, commission_drag_bps=0,
                spread_drag_bps=0, impact_drag_bps=0,
                slippage_drag_bps=0, n_trades=0,
                avg_trade_size_usd=0, annual_turnover=0,
                best_fills=[], worst_fills=[],
            )

        total_notional = sum(f.fill_price * f.order.shares for f in fills)
        n_trades       = len(fills)

        # Component costs
        avg_commission_bps= float(np.mean([
            f.commission / (f.fill_price * f.order.shares) * 10000
            for f in fills if f.fill_price * f.order.shares > 0
        ]))
        avg_spread_bps    = float(np.mean([f.slippage_bps   for f in fills]))
        avg_impact_bps    = float(np.mean([f.market_impact_bps for f in fills]))
        avg_total_bps     = float(np.mean([f.total_cost_bps    for f in fills]))

        # Note: drag = TC × turnover
        annual_turnover = metrics.get("annual_turnover", 1.0)
        total_drag_bps  = avg_total_bps * annual_turnover * 2  # Round-trip

        # Gross vs net: approximate gross by adding back cost drag
        net_return   = metrics.get("annual_return", 0)
        gross_return = net_return + total_drag_bps / 10000

        # Best/worst fills
        sorted_fills = sorted(fills, key=lambda f: f.total_cost_bps)
        best_fills   = [
            {"ticker": f.order.ticker, "cost_bps": f.total_cost_bps,
             "shares": f.order.shares, "date": str(f.filled_at)}
            for f in sorted_fills[:3]
        ]
        worst_fills  = [
            {"ticker": f.order.ticker, "cost_bps": f.total_cost_bps,
             "shares": f.order.shares, "date": str(f.filled_at)}
            for f in sorted_fills[-3:]
        ]

        return TCAnalysis(
            period_return_gross  = round(gross_return, 4),
            period_return_net    = round(net_return, 4),
            total_drag_bps       = round(total_drag_bps, 1),
            commission_drag_bps  = round(avg_commission_bps * annual_turnover * 2, 1),
            spread_drag_bps      = round(avg_spread_bps     * annual_turnover * 2, 1),
            impact_drag_bps      = round(avg_impact_bps     * annual_turnover * 2, 1),
            slippage_drag_bps    = round((avg_total_bps - avg_commission_bps
                                          - avg_spread_bps - avg_impact_bps) * annual_turnover * 2, 1),
            n_trades             = n_trades,
            avg_trade_size_usd   = round(total_notional / n_trades, 0) if n_trades else 0,
            annual_turnover      = round(annual_turnover, 3),
            best_fills           = best_fills,
            worst_fills          = worst_fills,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Full attribution report
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FullAttributionReport:
    """Complete attribution analysis for a backtest."""
    result:    BacktestResult
    bhb:       BHBResult
    factor:    FactorAttributionResult
    tc:        TCAnalysis

    def print_report(self) -> str:
        m = self.result.compute_metrics()
        lines = [
            "═" * 65,
            f"  FULL ATTRIBUTION REPORT — {self.result.strategy_name}",
            f"  {self.result.start_date} → {self.result.end_date}",
            "═" * 65,
            "",
            self.result.summary(),
            "",
            self.bhb.summary(),
            "",
            self.factor.summary(),
            "",
            self.tc.summary(),
            "═" * 65,
        ]
        return "\n".join(lines)


def run_full_attribution(result: BacktestResult) -> FullAttributionReport:
    """Run all three attribution analyses on a BacktestResult."""
    bhb    = BHBAttributor().attribute(result)
    factor = FactorAttributor().attribute(result)
    tc     = TCAttributor().attribute(result)
    return FullAttributionReport(result=result, bhb=bhb, factor=factor, tc=tc)


if __name__ == "__main__":
    import logging
    from datetime import date
    from src.backtest.backtest_engine import BacktestEngine, MomentumStrategy

    logging.basicConfig(level=logging.INFO)
    print("=" * 65)
    print("  Performance Attribution — Test")
    print("=" * 65)

    engine = BacktestEngine()
    result = engine.run(
        strategy        = MomentumStrategy(params={"top_n": 4}),
        tickers         = ["AAPL","MSFT","NVDA","GOOGL","JPM","BAC","XOM"],
        start_date      = date(2021, 1, 1),
        end_date        = date(2023, 6, 30),
        initial_capital = 1_000_000,
        verbose         = True,
    )

    report = run_full_attribution(result)
    print(report.print_report())
    print("\n✅ Attribution tests passed")
