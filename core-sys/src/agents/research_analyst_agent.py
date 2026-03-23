"""
AI Hedge Fund — Part 2: Multi-Agent System
============================================
research_analyst_agent.py — AI Research Analyst

The Research Analyst Agent provides deep-dive investment analysis
on individual securities. It differs from the Portfolio Manager Agent
(which makes allocation decisions) in that its sole job is
to build the investment thesis — not to decide position size or risk.

In a real fund:
  PM says "look into NVDA"
  → Research Analyst does the full analysis
  → Returns: thesis, valuation, risks, catalysts, comp set
  → PM uses this to make the allocation decision

This separation is important:
  - Research is slower and more thorough (minutes, not seconds)
  - PM needs fast decisions (seconds)
  - Analyst catches things the PM's quick-scan misses

This agent:
  1. Analyses price history, momentum, and technical signals
  2. Evaluates earnings and revenue trends from price-implied data
  3. Assesses sector and competitive position
  4. Builds a structured investment thesis with bull/bear cases
  5. Computes a price target range
  6. Identifies key catalysts and risk factors

Integrates with your quant-portfolio modules:
  quant-portfolio/06_alpha_research/fama_french/fama_french.py
  quant-portfolio/06_alpha_research/ml_return_prediction/ml_return_prediction.py
  quant-portfolio/06_alpha_research/pairs_trading/pairs_trading.py   (comps)
  quant-portfolio/04_risk_models/var_calculator/var_calculator.py
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("hedge_fund.research_analyst")


# ─────────────────────────────────────────────────────────────────────────────
# Research output structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PriceTarget:
    """Bull/base/bear price target for a security."""
    current_price: float
    bear_target:   float
    base_target:   float
    bull_target:   float
    methodology:   str
    time_horizon:  str = "12 months"

    @property
    def upside_base(self) -> float:
        return (self.base_target - self.current_price) / self.current_price

    @property
    def downside_bear(self) -> float:
        return (self.bear_target - self.current_price) / self.current_price

    @property
    def risk_reward_ratio(self) -> float:
        """Upside / |Downside| — higher is better."""
        downside = abs(self.downside_bear)
        if downside < 1e-4:
            return 0.0
        return self.upside_base / downside

    def to_dict(self) -> dict:
        return {
            "current_price":     self.current_price,
            "bear_target":       round(self.bear_target, 2),
            "base_target":       round(self.base_target, 2),
            "bull_target":       round(self.bull_target, 2),
            "upside_base":       f"{self.upside_base:+.1%}",
            "downside_bear":     f"{self.downside_bear:+.1%}",
            "risk_reward_ratio": round(self.risk_reward_ratio, 2),
            "methodology":       self.methodology,
            "time_horizon":      self.time_horizon,
        }


@dataclass
class ResearchReport:
    """Complete research report on a security."""
    ticker:           str
    analyst:          str
    timestamp:        datetime

    # Quantitative findings
    price_target:     Optional[PriceTarget]
    quant_signals:    Dict[str, Any]
    technical_setup:  Dict[str, Any]
    risk_metrics:     Dict[str, Any]
    relative_metrics: Dict[str, Any]   # vs peers

    # LLM-generated qualitative analysis
    investment_thesis:   str
    bull_case:           str
    bear_case:           str
    key_catalysts:       List[str]
    key_risks:           List[str]
    recommendation:      str            # STRONG BUY / BUY / HOLD / SELL / STRONG SELL
    conviction_score:    float          # 0-1

    # Comparable companies
    peer_comparison:  Dict[str, Dict]

    def to_dict(self) -> dict:
        return {
            "ticker":             self.ticker,
            "analyst":            self.analyst,
            "timestamp":          self.timestamp.isoformat(),
            "price_target":       self.price_target.to_dict() if self.price_target else None,
            "quant_signals":      self.quant_signals,
            "technical_setup":    self.technical_setup,
            "risk_metrics":       self.risk_metrics,
            "investment_thesis":  self.investment_thesis,
            "bull_case":          self.bull_case,
            "bear_case":          self.bear_case,
            "key_catalysts":      self.key_catalysts,
            "key_risks":          self.key_risks,
            "recommendation":     self.recommendation,
            "conviction_score":   self.conviction_score,
            "peer_comparison":    self.peer_comparison,
        }

    def executive_summary(self) -> str:
        pt = self.price_target
        lines = [
            "═" * 65,
            f"  RESEARCH REPORT — {self.ticker}",
            f"  {self.timestamp:%Y-%m-%d %H:%M} | Analyst: {self.analyst}",
            "═" * 65,
            f"  Recommendation : {self.recommendation}",
            f"  Conviction      : {self.conviction_score:.0%}",
        ]
        if pt:
            lines += [
                f"  Price Target    : ${pt.base_target:.2f} base "
                f"(${pt.bear_target:.2f} bear / ${pt.bull_target:.2f} bull)",
                f"  Upside/Downside : {pt.upside_base:+.1%} / {pt.downside_bear:+.1%}",
                f"  Risk/Reward     : {pt.risk_reward_ratio:.1f}x",
            ]
        lines += [
            "─" * 65,
            "  INVESTMENT THESIS:",
            f"  {self.investment_thesis}",
            "─" * 65,
            "  KEY CATALYSTS:",
        ]
        for c in self.key_catalysts:
            lines.append(f"    • {c}")
        lines.append("  KEY RISKS:")
        for r in self.key_risks:
            lines.append(f"    ⚠ {r}")
        lines.append("═" * 65)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Quantitative analysis engine
# ─────────────────────────────────────────────────────────────────────────────

class QuantitativeAnalyzer:
    """
    Computes quantitative metrics for security analysis.

    These are the inputs to the LLM — grounding it in real numbers
    rather than hallucinated facts.
    """

    def __init__(self):
        from src.data.market_data import MarketDataFetcher, FeatureEngineer
        self.fetcher  = MarketDataFetcher()
        self.engineer = FeatureEngineer()

    def _get_prices(self, ticker: str, days: int = 504) -> pd.DataFrame:
        return self.fetcher.get_prices(ticker, days=days)

    def analyse_momentum_and_trend(self, ticker: str) -> Dict[str, Any]:
        """Multi-horizon momentum and trend analysis."""
        df = self._get_prices(ticker, days=504)
        if df.empty:
            return {"error": f"No data for {ticker}"}

        col     = "Adj Close" if "Adj Close" in df.columns else "Close"
        closes  = df[col]
        price   = float(closes.iloc[-1])
        log_ret = np.log(closes / closes.shift(1)).dropna()

        results = {"ticker": ticker, "current_price": round(price, 2)}

        # Returns at multiple horizons
        for h, label in [(1,"1d"), (5,"5d"), (21,"1m"), (63,"3m"),
                         (126,"6m"), (252,"12m")]:
            if len(closes) > h:
                ret = float(closes.iloc[-1] / closes.iloc[-h-1] - 1)
                results[f"return_{label}"] = round(ret * 100, 2)

        # Volatility
        if len(log_ret) >= 21:
            results["vol_21d_ann"]  = round(log_ret.iloc[-21:].std() * math.sqrt(252) * 100, 1)
        if len(log_ret) >= 63:
            results["vol_63d_ann"]  = round(log_ret.iloc[-63:].std() * math.sqrt(252) * 100, 1)
        if len(log_ret) >= 252:
            results["vol_252d_ann"] = round(log_ret.std() * math.sqrt(252) * 100, 1)

        # Distance from 52-week high/low
        if len(closes) >= 252:
            high_52w = float(closes.rolling(252).max().iloc[-1])
            low_52w  = float(closes.rolling(252).min().iloc[-1])
            results["dist_from_52w_high"] = round((price / high_52w - 1) * 100, 1)
            results["dist_from_52w_low"]  = round((price / low_52w - 1) * 100, 1)
            results["52w_high"]           = round(high_52w, 2)
            results["52w_low"]            = round(low_52w, 2)

        # Trend regime (above/below key MAs)
        for ma_window in [20, 50, 100, 200]:
            if len(closes) > ma_window:
                ma = float(closes.rolling(ma_window).mean().iloc[-1])
                results[f"price_vs_ma{ma_window}"] = round((price / ma - 1) * 100, 2)
                results[f"ma{ma_window}"]           = round(ma, 2)

        # Momentum quality: is momentum consistent or erratic?
        if len(log_ret) >= 63:
            r_63 = log_ret.iloc[-63:]
            pos_days = int((r_63 > 0).sum())
            results["pct_positive_days_63d"] = round(pos_days / len(r_63) * 100, 1)
            results["max_1d_gain"]  = round(float(r_63.max()) * 100, 2)
            results["max_1d_loss"]  = round(float(r_63.min()) * 100, 2)

        return results

    def analyse_technical_setup(self, ticker: str) -> Dict[str, Any]:
        """RSI, MACD, Bollinger Bands, volume analysis."""
        df = self._get_prices(ticker, days=252)
        if df.empty:
            return {"error": "No data"}

        col    = "Adj Close" if "Adj Close" in df.columns else "Close"
        close  = df[col]
        volume = df["Volume"] if "Volume" in df.columns else pd.Series(dtype=float)
        high   = df["High"]
        low    = df["Low"]

        result = {}

        # RSI
        delta    = close.diff()
        gain     = delta.clip(lower=0).rolling(14).mean()
        loss     = (-delta).clip(lower=0).rolling(14).mean().clip(lower=1e-10)
        rsi      = 100 - (100 / (1 + gain / loss))
        result["rsi_14"]         = round(float(rsi.iloc[-1]), 1)
        result["rsi_interpretation"] = (
            "OVERBOUGHT" if result["rsi_14"] > 70 else
            "OVERSOLD"   if result["rsi_14"] < 30 else
            "NEUTRAL"
        )

        # MACD
        ema12    = close.ewm(span=12, adjust=False).mean()
        ema26    = close.ewm(span=26, adjust=False).mean()
        macd     = ema12 - ema26
        signal   = macd.ewm(span=9, adjust=False).mean()
        hist     = macd - signal
        result["macd"]            = round(float(macd.iloc[-1]), 4)
        result["macd_signal"]     = round(float(signal.iloc[-1]), 4)
        result["macd_histogram"]  = round(float(hist.iloc[-1]), 4)
        result["macd_crossover"]  = (
            "BULLISH" if hist.iloc[-1] > 0 and hist.iloc[-2] <= 0 else
            "BEARISH" if hist.iloc[-1] < 0 and hist.iloc[-2] >= 0 else
            "NO_CROSSOVER"
        )

        # Bollinger Bands
        sma20    = close.rolling(20).mean()
        std20    = close.rolling(20).std()
        upper    = sma20 + 2 * std20
        lower    = sma20 - 2 * std20
        price    = float(close.iloc[-1])
        bb_pct   = float((price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1] + 1e-6))
        result["bb_pct_b"]        = round(bb_pct * 100, 1)   # 0-100 within bands
        result["bb_upper"]        = round(float(upper.iloc[-1]), 2)
        result["bb_lower"]        = round(float(lower.iloc[-1]), 2)
        result["bb_interpretation"] = (
            "NEAR_UPPER" if bb_pct > 0.8 else
            "NEAR_LOWER" if bb_pct < 0.2 else
            "MID_BAND"
        )

        # Volume
        if not volume.empty and len(volume) >= 20:
            vol_ma20 = float(volume.rolling(20).mean().iloc[-1])
            vol_today = float(volume.iloc[-1])
            result["volume_vs_avg"] = round(vol_today / vol_ma20, 2) if vol_ma20 > 0 else 1.0
            result["volume_trend"]  = (
                "HIGH_VOLUME"   if result["volume_vs_avg"] > 1.5 else
                "LOW_VOLUME"    if result["volume_vs_avg"] < 0.5 else
                "AVERAGE_VOLUME"
            )

        return result

    def analyse_peer_group(
        self, ticker: str, peers: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Compare performance against peer group.

        Identifies relative strength/weakness vs sector.
        In production: use sector ETF constituents or your pairs trading
        cointegration module to identify true peers.
        """
        # Default peer groups by ticker
        default_peers = {
            "AAPL":  ["MSFT", "GOOGL", "META", "AMZN"],
            "MSFT":  ["AAPL", "GOOGL", "ORCL", "CRM"],
            "NVDA":  ["AMD", "INTC", "QCOM", "AVGO"],
            "GOOGL": ["META", "MSFT", "AAPL", "AMZN"],
            "META":  ["SNAP", "GOOGL", "PINS", "TWTR"],
            "JPM":   ["BAC", "GS", "MS", "WFC", "C"],
            "BAC":   ["JPM", "WFC", "C", "GS"],
            "GS":    ["MS", "JPM", "BAC", "BLK"],
            "XOM":   ["CVX", "COP", "SLB"],
            "CVX":   ["XOM", "COP", "EOG"],
        }

        peer_list = peers or default_peers.get(ticker, ["SPY", "QQQ"])
        all_tickers = [ticker] + peer_list

        comparison = {}
        for t in all_tickers:
            df = self._get_prices(t, days=252)
            if df.empty:
                continue
            col = "Adj Close" if "Adj Close" in df.columns else "Close"
            closes = df[col]
            log_ret = np.log(closes / closes.shift(1)).dropna()

            comparison[t] = {
                "return_1m":  round(float(closes.iloc[-1] / closes.iloc[-22] - 1) * 100, 2),
                "return_3m":  round(float(closes.iloc[-1] / closes.iloc[-63] - 1) * 100, 2),
                "return_ytd": round(float(closes.iloc[-1] / closes.iloc[0] - 1) * 100, 2),
                "vol_ann":    round(float(log_ret.std() * math.sqrt(252)) * 100, 1),
                "sharpe_approx": round(
                    float((log_ret.mean() * 252 - 0.05) / (log_ret.std() * math.sqrt(252) + 1e-6)),
                    2
                ),
            }

        return comparison

    def estimate_price_target(
        self, ticker: str, quant_data: Dict
    ) -> PriceTarget:
        """
        Quantitative price target estimation.

        Uses a simple momentum reversion + vol-adjusted range approach.
        In production: replace with your DCF/DDM valuation model.

        Methodology:
          Base = current price × (1 + expected_return_1y)
          Bull = base + 1 standard deviation of annual returns
          Bear = base - 1 standard deviation of annual returns
        """
        price     = quant_data.get("current_price", 100)
        ret_12m   = quant_data.get("return_12m", 0) / 100     # Convert to decimal
        vol_ann   = quant_data.get("vol_252d_ann", 25) / 100  # Convert to decimal

        # Base: assume momentum continues but decays (regression to mean)
        # Mean reversion factor: 50% of trailing 12m return
        expected_return = ret_12m * 0.5

        base  = price * (1 + expected_return)
        bull  = base  * (1 + vol_ann)
        bear  = base  * (1 - vol_ann)

        return PriceTarget(
            current_price = round(price, 2),
            bear_target   = round(bear, 2),
            base_target   = round(base, 2),
            bull_target   = round(bull, 2),
            methodology   = (
                "Momentum reversion (50% decay of trailing 12m return) ± 1σ annual vol. "
                "Note: This is a quantitative first-pass. For precise valuation, "
                "supplement with fundamental analysis."
            ),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Research Analyst Agent
# ─────────────────────────────────────────────────────────────────────────────

class ResearchAnalystAgent(BaseAgent):
    """
    AI Research Analyst for the hedge fund.

    Produces comprehensive investment research reports combining:
      - Quantitative signals (momentum, vol, technical)
      - Peer comparison
      - LLM-generated investment thesis
      - Bull/bear case analysis
      - Price target (quantitative + LLM judgment)

    Communication:
      Receives: research_request (from PM or Coordinator)
      Sends:    research_report (structured report), alerts
    """

    SYSTEM_PROMPT = """You are a senior equity research analyst at a systematic hedge fund.

YOUR ROLE:
Produce rigorous, data-driven investment research that helps the Portfolio Manager
make better allocation decisions. You are NOT making the allocation decision —
that's the PM's job. Your job is to build the investment thesis.

RESEARCH PROCESS:
1. GATHER ALL QUANTITATIVE DATA
   - Use get_price_analysis to understand momentum and trend
   - Use get_technical_setup for entry/exit timing signals
   - Use get_peer_comparison to assess relative strength
   - ALWAYS use tools first before forming a view

2. ASSESS THE QUANTITATIVE PICTURE
   - What story do the numbers tell?
   - Is momentum consistent or erratic?
   - How is this stock positioned relative to peers?
   - What is the risk/reward at current price?

3. BUILD THE INVESTMENT THESIS
   - What is the core reason to own this stock?
   - What would need to be true for the bull case to play out?
   - What are the specific risks that could invalidate the thesis?
   - What are the key near-term catalysts (earnings, product launches, regulatory)?

4. ASSIGN A RECOMMENDATION
   STRONG BUY: Multiple strong signals, compelling risk/reward >2:1, low correlation
   BUY: Positive signals, reasonable risk/reward >1.5:1
   HOLD: Mixed signals, no clear directional edge
   SELL: Deteriorating signals, negative momentum, poor risk/reward
   STRONG SELL: Multiple bearish signals, avoid or short

OUTPUT FORMAT — respond in JSON:
{
  "recommendation": "STRONG BUY|BUY|HOLD|SELL|STRONG SELL",
  "conviction_score": float between 0 and 1,
  "investment_thesis": "2-3 paragraph thesis",
  "bull_case": "1 paragraph specific bull scenario",
  "bear_case": "1 paragraph specific bear scenario with specific risks",
  "key_catalysts": ["catalyst1", "catalyst2", "catalyst3"],
  "key_risks": ["risk1", "risk2", "risk3"],
  "time_horizon": "short (1-4 weeks) | medium (1-6 months) | long (6-18 months)"
}

CRITICAL RULES:
- Base your thesis on the quantitative data — do not invent facts
- Be specific: "MACD histogram crossed positive on [date]" not "looks bullish"
- If data is insufficient, say so explicitly
- Both bull and bear cases must include specific measurable conditions
"""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
    ):
        from src.agents.base_agent import AgentConfig
        cfg_obj = config or AgentConfig(
            name        = "ResearchAnalyst",
            model       = "claude-sonnet-4-6",
            temperature = 0.15,  # Slightly higher — research needs some creativity
            max_tokens  = 4096,
        )
        self.analyzer = QuantitativeAnalyzer()
        super().__init__(cfg_obj)

    def _get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def _get_tools(self) -> List[Tool]:
        from src.agents.base_agent import Tool
        return [
            Tool(
                name = "get_price_analysis",
                func = self._tool_price_analysis,
                description = (
                    "Get comprehensive price history analysis: momentum at multiple horizons, "
                    "volatility, 52-week high/low, moving averages. "
                    "Input: ticker symbol. Returns full quantitative momentum profile."
                ),
                param_schema = {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "description": "Stock ticker symbol"}
                    },
                    "required": ["ticker"]
                }
            ),
            Tool(
                name = "get_technical_setup",
                func = self._tool_technical_setup,
                description = (
                    "Get technical indicator analysis: RSI, MACD, Bollinger Bands, volume. "
                    "Input: ticker symbol. Returns technical signal summary."
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
                name = "get_peer_comparison",
                func = self._tool_peer_comparison,
                description = (
                    "Compare stock performance vs peer group. "
                    "Input: ticker symbol. "
                    "Returns relative performance over 1m, 3m, YTD vs peers."
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
                name = "get_market_regime",
                func = self._tool_market_regime,
                description = (
                    "Get current market regime (bull/bear/crisis/sideways), VIX level, "
                    "and S&P 500 trend. Important context for all recommendations."
                ),
                param_schema = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            ),
            Tool(
                name = "get_sector_performance",
                func = self._tool_sector_performance,
                description = (
                    "Get recent performance of major sector ETFs. "
                    "Helps understand if a stock's move is sector-driven or idiosyncratic. "
                    "Input: anything. Returns sector ETF returns."
                ),
                param_schema = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            ),
        ]

    # ── Tool implementations ──────────────────────────────────────────────────

    def _tool_price_analysis(self, ticker: str) -> str:
        data = self.analyzer.analyse_momentum_and_trend(ticker)
        return json.dumps(data)

    def _tool_technical_setup(self, ticker: str) -> str:
        data = self.analyzer.analyse_technical_setup(ticker)
        return json.dumps(data)

    def _tool_peer_comparison(self, ticker: str) -> str:
        data = self.analyzer.analyse_peer_group(ticker)
        return json.dumps(data)

    def _tool_market_regime(self, **kwargs) -> str:
        from src.data.market_data import MacroDataFetcher, MarketDataFetcher
        try:
            fetcher  = MarketDataFetcher()
            snapshot = fetcher.get_market_snapshot(["SPY", "QQQ", "^VIX", "TLT", "GLD"])
            return json.dumps({
                "regime":         snapshot.regime.value,
                "vix":            snapshot.vix_level,
                "spy_1d":         snapshot.spy_return_1d,
                "spy_5d":         snapshot.returns_5d.get("SPY"),
                "spy_21d":        snapshot.returns_21d.get("SPY"),
                "qqq_1d":         snapshot.returns_1d.get("QQQ"),
                "gold_1d":        snapshot.returns_1d.get("GLD"),
                "tlt_1d":         snapshot.returns_1d.get("TLT"),
                "description":    f"Market is in {snapshot.regime.value} regime",
            })
        except Exception as e:
            return json.dumps({"error": str(e), "regime": "UNKNOWN"})

    def _tool_sector_performance(self, **kwargs) -> str:
        sector_etfs = {
            "Technology":    "XLK",
            "Financials":    "XLF",
            "Healthcare":    "XLV",
            "Energy":        "XLE",
            "Consumer":      "XLY",
            "Industrials":   "XLI",
            "Utilities":     "XLU",
            "Real_Estate":   "XLRE",
            "Materials":     "XLB",
            "Comm_Services": "XLC",
        }
        from src.data.market_data import MarketDataFetcher
        fetcher = MarketDataFetcher()
        result  = {}

        for sector, etf in sector_etfs.items():
            df = fetcher.get_prices(etf, days=30)
            if df.empty:
                continue
            col = "Adj Close" if "Adj Close" in df.columns else "Close"
            closes = df[col]
            result[sector] = {
                "etf":       etf,
                "return_5d":  round(float(closes.iloc[-1] / closes.iloc[-6] - 1) * 100, 2),
                "return_21d": round(float(closes.iloc[-1] / closes.iloc[-22] - 1) * 100, 2),
            }

        return json.dumps(result)

    # ── Main analysis method ──────────────────────────────────────────────────

    def analyse(self, ticker: str, context: str = "") -> ResearchReport:
        """
        Produce a full research report on a security.

        Args:
            ticker  : Stock ticker to analyse
            context : Additional context (market conditions, reason for request)

        Returns:
            ResearchReport with full quantitative analysis and LLM thesis
        """
        logger.info(f"ResearchAnalyst: starting analysis of {ticker}")

        # Get all quantitative data upfront
        momentum_data  = self.analyzer.analyse_momentum_and_trend(ticker)
        technical_data = self.analyzer.analyse_technical_setup(ticker)
        peer_data      = self.analyzer.analyse_peer_group(ticker)
        price_target   = self.analyzer.estimate_price_target(ticker, momentum_data)

        # Risk metrics
        risk_data = {}
        if "vol_21d_ann" in momentum_data:
            price = momentum_data.get("current_price", 100)
            risk_data = {
                "vol_21d_ann":    momentum_data.get("vol_21d_ann"),
                "vol_63d_ann":    momentum_data.get("vol_63d_ann"),
                "max_1d_loss":    momentum_data.get("max_1d_loss"),
                "dist_52w_high":  momentum_data.get("dist_from_52w_high"),
                "dist_52w_low":   momentum_data.get("dist_from_52w_low"),
            }

        # Build user prompt for LLM
        user_prompt = f"""Analyse {ticker} for potential investment.

QUANTITATIVE DATA:
{json.dumps(momentum_data, indent=2)}

TECHNICAL SETUP:
{json.dumps(technical_data, indent=2)}

PEER COMPARISON (vs {list(peer_data.keys())[1:4] if len(peer_data) > 1 else 'N/A'}):
{json.dumps(peer_data, indent=2)}

QUANTITATIVE PRICE TARGET:
Bear: ${price_target.bear_target:.2f} | Base: ${price_target.base_target:.2f} | Bull: ${price_target.bull_target:.2f}
Risk/Reward: {price_target.risk_reward_ratio:.1f}x

ADDITIONAL CONTEXT: {context or 'Standard research request'}

Use the available tools to gather any additional data you need, 
then provide your complete investment thesis in the required JSON format."""

        # Run the LLM analysis
        response_text, tool_calls = self.think(
            user_message = user_prompt,
            use_tools    = True,
            purpose      = f"research_analysis_{ticker}",
        )

        # Parse LLM response
        llm_analysis = self._parse_llm_response(response_text)

        report = ResearchReport(
            ticker           = ticker,
            analyst          = self.name,
            timestamp        = datetime.now(),
            price_target     = price_target,
            quant_signals    = momentum_data,
            technical_setup  = technical_data,
            risk_metrics     = risk_data,
            relative_metrics = peer_data.get(ticker, {}),
            investment_thesis   = llm_analysis.get("investment_thesis", response_text[:500]),
            bull_case           = llm_analysis.get("bull_case", ""),
            bear_case           = llm_analysis.get("bear_case", ""),
            key_catalysts       = llm_analysis.get("key_catalysts", []),
            key_risks           = llm_analysis.get("key_risks", []),
            recommendation      = llm_analysis.get("recommendation", "HOLD"),
            conviction_score    = float(llm_analysis.get("conviction_score", 0.5)),
            peer_comparison     = peer_data,
        )

        logger.info(
            f"Research complete: {ticker} | "
            f"{report.recommendation} | "
            f"conviction={report.conviction_score:.0%}"
        )

        return report

    def _parse_llm_response(self, text: str) -> Dict:
        """Parse JSON from LLM response with fallbacks."""
        import re

        # Try direct JSON parse
        text = text.strip()
        for fence in ["```json", "```JSON", "```"]:
            if fence in text:
                parts = text.split(fence)
                for part in parts:
                    clean = part.strip().rstrip("`").strip()
                    if clean.startswith("{"):
                        text = clean
                        break
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try regex extraction
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Fallback: extract recommendation from text
        rec = "HOLD"
        for r in ["STRONG BUY", "STRONG SELL", "BUY", "SELL", "HOLD"]:
            if r in text.upper():
                rec = r
                break

        return {
            "recommendation":   rec,
            "conviction_score": 0.5,
            "investment_thesis": text[:600],
            "bull_case":        "",
            "bear_case":        "",
            "key_catalysts":    [],
            "key_risks":        [],
        }

    # ── MessageBus handler ────────────────────────────────────────────────────

    def handle_message(self, message) -> Optional[Dict[str, Any]]:
        """Process incoming research requests."""
        subject = message.subject.lower()
        payload = message.payload

        logger.info(
            f"ResearchAnalyst handling: {message.subject} from {message.sender}"
        )

        if "research_request" in subject or "analyse" in subject or "analyze" in subject:
            ticker  = payload.get("ticker", "")
            context = payload.get("context", "")

            if not ticker:
                return {"error": "No ticker provided in research request"}

            report = self.analyse(ticker, context)
            return report.to_dict()

        elif "quick_signal" in subject:
            # Faster analysis without full LLM reasoning
            ticker   = payload.get("ticker", "")
            momentum = self.analyzer.analyse_momentum_and_trend(ticker)
            technical = self.analyzer.analyse_technical_setup(ticker)
            return {
                "ticker":    ticker,
                "momentum":  momentum,
                "technical": technical,
                "timestamp": datetime.now().isoformat(),
            }

        else:
            # Ad-hoc research questions
            ticker = payload.get("ticker", "")
            context = f"Ticker: {ticker}\n{json.dumps(payload)}"
            if ticker:
                # Full analysis
                report = self.analyse(ticker, message.subject)
                return report.to_dict()
            return {"error": "Could not determine research target from message"}


# Import BaseAgent here to avoid circular imports
from src.agents.base_agent import BaseAgent, Tool, AgentConfig


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parents[3]))
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  Research Analyst Agent — Test")
    print("=" * 60)

    import os
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  No LLM API key — testing quantitative analysis only")

        analyzer = QuantitativeAnalyzer()

        print("\n1. Momentum analysis: NVDA")
        momentum = analyzer.analyse_momentum_and_trend("NVDA")
        if "error" not in momentum:
            print(f"   Price: ${momentum.get('current_price')}")
            print(f"   1M:  {momentum.get('return_1m')}%")
            print(f"   3M:  {momentum.get('return_3m')}%")
            print(f"   12M: {momentum.get('return_12m')}%")
            print(f"   Vol: {momentum.get('vol_21d_ann')}% ann")

        print("\n2. Technical setup: NVDA")
        tech = analyzer.analyse_technical_setup("NVDA")
        print(f"   RSI(14): {tech.get('rsi_14')} — {tech.get('rsi_interpretation')}")
        print(f"   MACD crossover: {tech.get('macd_crossover')}")
        print(f"   BB %B: {tech.get('bb_pct_b')}")

        print("\n3. Peer comparison: NVDA vs AMD/INTC/QCOM")
        peers = analyzer.analyse_peer_group("NVDA")
        for t, data in peers.items():
            print(f"   {t}: 1M={data.get('return_1m')}% "
                  f"3M={data.get('return_3m')}% "
                  f"Sharpe={data.get('sharpe_approx')}")

        print("\n✅ Quantitative analysis tests passed (add API key for LLM analysis)")
    else:
        analyst = ResearchAnalystAgent()
        print("\nAnalysing AAPL...")
        report = analyst.analyse("AAPL")
        print(report.executive_summary())
        print(f"\n{analyst.get_metrics()}")
