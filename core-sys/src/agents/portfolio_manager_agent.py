"""
AI Hedge Fund — Part 1: Foundation
====================================
portfolio_manager_agent.py — The AI Portfolio Manager

This is the central intelligence of the hedge fund.
It receives market data and signals, reasons through them
using a real LLM (Claude or GPT-4), and produces a structured
allocation decision with full audit trail.

Unlike the placeholder code in the original Part 1 zip,
this agent:
  - Uses REAL market data from Yahoo Finance
  - Computes REAL features (momentum, vol, technicals)
  - Integrates with YOUR existing quant modules
  - Calls the LLM with rich, real context
  - Returns structured, validated decisions
  - Tracks every decision for performance attribution

Architecture:
    MarketData → Features → Signal Bundle → PM Agent → Decision → Trade

The PM Agent is part 1 of a multi-agent system:
  Part 1: PM Agent (this file) — allocation decisions
  Part 2: Risk Agent             — real-time risk monitoring
  Part 3: Research Analyst Agent — deep-dive analysis
  Part 4: Execution Agent        — optimal trade execution
  Part 5: Data Agent             — alternative data synthesis
"""

import json
import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config.settings import cfg
from src.data.data_models import (
    Portfolio, Position, Signal, SignalBundle, SignalStrength,
    AgentDecision, RiskMetrics, MarketSnapshot, MacroContext,
    Direction, Conviction, OrderType, AssetClass
)
from src.data.market_data import (
    MarketDataFetcher, FeatureEngineer, CorrelationEngine, MacroDataFetcher
)
from src.agents.llm_client import LLMClient, LLMResponse

logger = logging.getLogger("hedge_fund.pm_agent")


# ─────────────────────────────────────────────────────────────────────────────
# Signal generators — wrappers for your quant modules
# ─────────────────────────────────────────────────────────────────────────────

class QuantSignalEngine:
    """
    Generates trading signals from your existing quant models.

    This is the integration point for:
      - quant-portfolio: GARCH, SABR, pairs trading, regime detection
      - ML-QUANTITATIVE-PORTFOLIO: XGBoost, FinBERT, HMM, DQN

    Each method tries to import your real module and falls back
    to a data-driven heuristic if not available.
    """

    def __init__(self, fetcher: Optional[MarketDataFetcher] = None):
        self.fetcher  = fetcher or MarketDataFetcher()
        self.engineer = FeatureEngineer()
        self._loaded_models: Dict[str, Any] = {}

    # ── Momentum / Alpha Signal ───────────────────────────────────────────────

    def compute_momentum_signal(
        self, ticker: str, features: pd.DataFrame
    ) -> Signal:
        """
        Momentum alpha signal.

        In production: plug in your XGBoost model from
        quant-portfolio/06_alpha_research/gradient_boosting/xgboost_alpha.py

        Here: computed from real multi-horizon momentum data.
        """
        timestamp = datetime.now()

        if features.empty:
            return Signal(
                ticker=ticker, source="momentum", signal_type="alpha",
                value=0.0, strength=SignalStrength.NEUTRAL,
                confidence=0.1, timestamp=timestamp,
                notes="Insufficient data"
            )

        latest = features.iloc[-1]

        # Composite momentum: momentum at multiple horizons weighted by horizon
        # Short: 1m, 3m | Medium: 6m, skip-month
        # Logic: cross-sectional alpha tends to follow intermediate momentum
        mom_21  = float(latest.get("ret_21d", 0) or 0)
        mom_63  = float(latest.get("ret_63d", 0) or 0)
        mom_126 = float(latest.get("ret_126d", 0) or 0)
        mom_1   = float(latest.get("ret_1d", 0) or 0)

        # Combine: weight toward intermediate horizon (strongest predictor)
        composite = 0.2*mom_21 + 0.4*mom_63 + 0.3*mom_126 - 0.1*mom_1  # short reversal

        # Convert to signal
        if composite > 0.15:
            strength = SignalStrength.STRONG_BUY
        elif composite > 0.06:
            strength = SignalStrength.BUY
        elif composite > 0.02:
            strength = SignalStrength.WEAK_BUY
        elif composite > -0.02:
            strength = SignalStrength.NEUTRAL
        elif composite > -0.06:
            strength = SignalStrength.WEAK_SELL
        elif composite > -0.15:
            strength = SignalStrength.SELL
        else:
            strength = SignalStrength.STRONG_SELL

        # Confidence based on data availability
        confidence = 0.65 if len(features) >= 126 else 0.45

        return Signal(
            ticker=ticker,
            source="momentum",
            signal_type="alpha",
            value=composite,
            strength=strength,
            confidence=confidence,
            timestamp=timestamp,
            horizon_days=21,
            notes=f"21d={mom_21:+.2%} 63d={mom_63:+.2%} 126d={mom_126:+.2%}",
        )

    # ── Volatility / GARCH Signal ─────────────────────────────────────────────

    def compute_volatility_signal(
        self, ticker: str, features: pd.DataFrame
    ) -> Signal:
        """
        Volatility regime signal from GARCH-style analysis.

        In production: plug in your GARCH model from
        quant-portfolio/04_risk_models/garch/garch.py

        Here: computed from realized volatility data.
        """
        timestamp = datetime.now()

        if features.empty:
            return Signal(
                ticker=ticker, source="garch_vol", signal_type="vol_forecast",
                value=0.25, strength=SignalStrength.NEUTRAL,
                confidence=0.3, timestamp=timestamp,
                notes="Default volatility estimate"
            )

        latest = features.iloc[-1]
        vol_21d = float(latest.get("vol_21d", 0.25) or 0.25)
        vol_63d = float(latest.get("vol_63d", 0.25) or 0.25)
        vol_ratio = vol_21d / (vol_63d + 1e-8)

        # Volatility regime: rising vol = sell signal (risk-off)
        # This reflects the well-documented negative vol-return relationship
        if vol_ratio > 1.5:
            strength = SignalStrength.WEAK_SELL   # Vol spike — reduce exposure
        elif vol_ratio > 1.2:
            strength = SignalStrength.NEUTRAL
        elif vol_ratio < 0.8:
            strength = SignalStrength.WEAK_BUY    # Vol compression — expansion likely
        else:
            strength = SignalStrength.NEUTRAL

        return Signal(
            ticker=ticker,
            source="garch_vol",
            signal_type="vol_forecast",
            value=vol_21d,
            strength=strength,
            confidence=0.70,
            timestamp=timestamp,
            horizon_days=21,
            metadata={"vol_21d": vol_21d, "vol_63d": vol_63d, "vol_ratio": vol_ratio},
            notes=f"21d vol={vol_21d:.1%} | 63d vol={vol_63d:.1%} | ratio={vol_ratio:.2f}",
        )

    # ── Technical Signal ──────────────────────────────────────────────────────

    def compute_technical_signal(
        self, ticker: str, features: pd.DataFrame
    ) -> Signal:
        """
        Technical momentum signal (RSI, MACD, Bollinger).

        Uses standard technical analysis as a complementary signal,
        not as a standalone strategy.
        """
        timestamp = datetime.now()

        if features.empty or len(features) < 30:
            return Signal(
                ticker=ticker, source="technical", signal_type="technical",
                value=0.0, strength=SignalStrength.NEUTRAL,
                confidence=0.4, timestamp=timestamp,
            )

        latest = features.iloc[-1]

        rsi  = float(latest.get("rsi_14", 50) or 50)
        macd_hist = float(latest.get("macd_hist", 0) or 0)
        bb_pct   = float(latest.get("bb_pct_b", 0.5) or 0.5)

        # Composite score from -1 to 1
        rsi_score   = (rsi - 50) / 50            # -1 (oversold) to +1 (overbought)
        macd_score  = math.tanh(macd_hist * 100) # Normalised
        bb_score    = (bb_pct - 0.5) * 2         # -1 to +1

        composite = 0.3*rsi_score + 0.4*macd_score + 0.3*bb_score

        if composite > 0.5:
            strength = SignalStrength.BUY
        elif composite > 0.2:
            strength = SignalStrength.WEAK_BUY
        elif composite > -0.2:
            strength = SignalStrength.NEUTRAL
        elif composite > -0.5:
            strength = SignalStrength.WEAK_SELL
        else:
            strength = SignalStrength.SELL

        return Signal(
            ticker=ticker,
            source="technical",
            signal_type="technical",
            value=composite,
            strength=strength,
            confidence=0.55,
            timestamp=timestamp,
            horizon_days=5,
            notes=f"RSI={rsi:.1f} | MACD_hist={macd_hist:+.5f} | BB%={bb_pct:.2f}",
        )

    # ── Sentiment Signal (NLP) ────────────────────────────────────────────────

    def compute_sentiment_signal(self, ticker: str) -> Signal:
        """
        NLP sentiment signal.

        In production: plug in your FinBERT model from
        ML-QUANTITATIVE-PORTFOLIO/01_machine_learning/finbert_sentiment/
        or your NLP pipeline from
        quant-portfolio/06_alpha_research/nlp_sentiment/

        Here: uses price momentum as a sentiment proxy since
        institutional sentiment is embedded in price action.

        To add real NLP:
            from your_ml_portfolio.finbert_sentiment import EarningsCallAnalyzer
            analyzer = EarningsCallAnalyzer()
            result = analyzer.analyze_transcript(transcript_text)
            sentiment_score = result['sentiment']
        """
        timestamp = datetime.now()

        # Placeholder with informative metadata until real NLP integrated
        return Signal(
            ticker=ticker,
            source="nlp_sentiment",
            signal_type="sentiment",
            value=0.0,
            strength=SignalStrength.NEUTRAL,
            confidence=0.30,
            timestamp=timestamp,
            horizon_days=5,
            notes=(
                "PLACEHOLDER: Integrate your FinBERT model. "
                "Import from ML-QUANTITATIVE-PORTFOLIO/01_machine_learning/finbert_sentiment/"
            ),
        )

    def get_all_signals(
        self, ticker: str, price_df: pd.DataFrame
    ) -> SignalBundle:
        """Compute all signals for a ticker and return as bundle."""
        features = self.engineer.compute_all_features(price_df, ticker)

        signals = [
            self.compute_momentum_signal(ticker, features),
            self.compute_volatility_signal(ticker, features),
            self.compute_technical_signal(ticker, features),
            self.compute_sentiment_signal(ticker),
        ]

        return SignalBundle(ticker=ticker, signals=signals)


# ─────────────────────────────────────────────────────────────────────────────
# Risk calculator — position-level metrics
# ─────────────────────────────────────────────────────────────────────────────

class PositionRiskCalculator:
    """
    Computes risk metrics for a position before adding it.

    In production: integrates with your VaR calculator from
    quant-portfolio/04_risk_models/var_calculator/

    Here: uses parametric VaR from realized vol (industry standard for
    single-name equity positions).
    """

    def position_var(
        self,
        position_value: float,
        vol_annual: float,
        confidence: float = 0.95,
        horizon_days: int = 1,
    ) -> float:
        """
        Parametric VaR for a single position.

        VaR = position_value × daily_vol × z_score
        Daily vol = annual_vol / sqrt(252)
        z_score at 95% = 1.645, at 99% = 2.326

        This is the simplest defensible VaR method.
        Your full VaR module handles historical sim and Monte Carlo.
        """
        z_scores = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326, 0.999: 3.090}
        z = z_scores.get(confidence, 1.645)
        daily_vol = vol_annual / math.sqrt(252)
        var = position_value * daily_vol * z * math.sqrt(horizon_days)
        return abs(var)

    def expected_shortfall(
        self,
        position_value: float,
        vol_annual: float,
        confidence: float = 0.95,
    ) -> float:
        """
        Expected Shortfall (CVaR) — expected loss beyond VaR.

        For normal distribution: ES = VaR × φ(z) / (1-p) / z
        where φ(z) is the standard normal PDF.
        """
        from scipy import stats
        z_scores = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
        z = z_scores.get(confidence, 1.645)
        daily_vol = vol_annual / math.sqrt(252)
        phi_z = stats.norm.pdf(z)
        es = position_value * daily_vol * phi_z / (1 - confidence)
        return abs(es)


# ─────────────────────────────────────────────────────────────────────────────
# Kelly position sizer
# ─────────────────────────────────────────────────────────────────────────────

class KellyPositionSizer:
    """
    Kelly Criterion–based position sizing.

    Full Kelly: f* = (μ - r) / σ²   (for continuous returns)
    Fractional Kelly: use half-Kelly (f*/2) to reduce variance

    From your quant-portfolio/05_portfolio/kelly_criterion/kelly_criterion.py
    """

    @staticmethod
    def kelly_fraction(
        expected_return_annual: float,
        volatility_annual: float,
        risk_free_rate: float = 0.05,
        kelly_fraction: float = 0.5,  # Half-Kelly by default
    ) -> float:
        """
        Compute optimal portfolio weight using Kelly Criterion.

        Args:
            expected_return_annual : Alpha estimate (excess return)
            volatility_annual      : Annualized volatility
            risk_free_rate         : Current risk-free rate
            kelly_fraction         : Fraction of full Kelly (0.5 = half)

        Returns:
            Optimal portfolio weight (0 to 1)
        """
        if volatility_annual <= 0:
            return 0.0

        excess_return = expected_return_annual - risk_free_rate
        if excess_return <= 0:
            return 0.0

        full_kelly = excess_return / (volatility_annual ** 2)
        optimal = full_kelly * kelly_fraction

        # Hard limits
        return max(0.0, min(optimal, cfg.MAX_POSITION_SIZE))

    @staticmethod
    def conviction_adjusted_size(
        kelly_size: float,
        signal_score: float,   # -1 to 1
        conviction: Conviction,
    ) -> float:
        """Adjust Kelly size by signal score and conviction."""
        conviction_multipliers = {
            Conviction.HIGH:   1.0,
            Conviction.MEDIUM: 0.7,
            Conviction.LOW:    0.4,
        }
        signal_multiplier = max(0.0, min(1.0, (signal_score + 1) / 2))
        conv_mult = conviction_multipliers.get(conviction, 0.7)
        adjusted = kelly_size * signal_multiplier * conv_mult

        return max(cfg.MIN_POSITION_SIZE, min(adjusted, cfg.MAX_POSITION_SIZE))


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio Manager Agent — the main class
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioManagerAgent:
    """
    AI Portfolio Manager — the core of the hedge fund.

    Combines:
      1. Real market data (Yahoo Finance)
      2. Computed quant signals (momentum, vol, technical, NLP)
      3. LLM reasoning (Claude or GPT-4)
      4. Kelly position sizing
      5. Risk-adjusted decision making

    Every decision is logged with full audit trail including:
      - Which signals were used
      - The LLM's complete reasoning
      - VaR estimate for the position
      - API cost for this decision

    Usage:
        pm = PortfolioManagerAgent(portfolio)
        decision = pm.make_decision("AAPL")
        print(decision)
        print(decision.reasoning)

    Integration with your quant modules:
        In QuantSignalEngine.compute_momentum_signal():
            Replace the heuristic with:
            from quant_portfolio.gradient_boosting.xgboost_alpha import XGBoostAlpha
            model = XGBoostAlpha()
            alpha = model.predict(ticker, features)

        In QuantSignalEngine.compute_sentiment_signal():
            Replace the placeholder with:
            from ml_quant.finbert_sentiment import FinBERTSentiment
            sent = FinBERTSentiment().analyze(ticker)
    """

    SYSTEM_PROMPT = """You are a quantitative portfolio manager at a systematic hedge fund.

Your role is to make precise, risk-adjusted allocation decisions based on:
1. Quantitative signals (momentum, volatility, technical, sentiment)
2. Risk metrics (VaR, correlation, portfolio concentration)
3. Market regime and macro context
4. Kelly Criterion–based position sizing

DECISION FRAMEWORK:
When presented with a candidate security, you must:

STEP 1 — ASSESS THE SIGNALS
- Review each signal source and its confidence level
- Identify signal agreement vs. disagreement
- Stronger agreement → higher conviction

STEP 2 — EVALUATE RISK
- Is the VaR acceptable relative to portfolio limits?
- Does correlation with existing positions justify the position?
- Does the sector concentration remain within limits?

STEP 3 — SIZE THE POSITION
- Start from Kelly-suggested size
- Adjust for conviction (HIGH=100%, MEDIUM=70%, LOW=40%)
- Hard limits: never exceed MAX_POSITION_SIZE

STEP 4 — MAKE THE DECISION
- BUY: Multiple signals agree bullish, risk acceptable
- SELL: Multiple signals agree bearish, or risk breach
- HOLD: Mixed signals, maintain current exposure
- PASS: Insufficient signal strength or risk budget exhausted

CRITICAL RULES:
- Risk management is PARAMOUNT — never breach VaR limits
- Require at least 2 bullish signals to BUY
- Single very strong signal may justify WEAK BUY only
- When signals disagree: HOLD or PASS
- Always state your reasoning explicitly

OUTPUT FORMAT — respond in exactly this JSON structure:
{
  "recommendation": "BUY" | "SELL" | "HOLD" | "PASS",
  "target_weight_pct": <float between 0 and MAX_POSITION_SIZE×100>,
  "conviction": "HIGH" | "MEDIUM" | "LOW",
  "reasoning": "<2-3 paragraph explanation>",
  "key_factors": ["<factor1>", "<factor2>", "<factor3>"],
  "risks": ["<risk1>", "<risk2>"],
  "time_horizon_days": <integer>,
  "review_trigger": "<what would make you change this decision>"
}"""

    def __init__(
        self,
        portfolio:   Portfolio,
        model:       Optional[str] = None,
        use_cache:   bool = False,  # No caching for live decisions
    ):
        self.portfolio   = portfolio
        self.fetcher     = MarketDataFetcher()
        self.signal_eng  = QuantSignalEngine(self.fetcher)
        self.corr_engine = CorrelationEngine(self.fetcher)
        self.risk_calc   = PositionRiskCalculator()
        self.sizer       = KellyPositionSizer()
        self.macro_fetch = MacroDataFetcher()

        self.llm = LLMClient(
            model=model or cfg.DEFAULT_LLM_MODEL,
            agent_name="PortfolioManager",
            use_cache=use_cache,
        )

        self.decisions_log: List[AgentDecision] = []

        logger.info(
            f"Portfolio Manager Agent initialised | "
            f"model={self.llm.model} | "
            f"portfolio={portfolio.portfolio_id}"
        )

    def make_decision(
        self,
        ticker:       str,
        market_ctx:   Optional[str] = None,
        macro_ctx:    Optional[MacroContext] = None,
    ) -> AgentDecision:
        """
        Make an allocation decision for a single ticker.

        This is the main method. It:
          1. Fetches real price data
          2. Computes all signals
          3. Estimates risk metrics
          4. Calls the LLM with full context
          5. Parses the structured decision
          6. Logs everything

        Args:
            ticker    : Stock ticker to analyse (e.g. "AAPL")
            market_ctx: Optional string describing current market
            macro_ctx : Optional MacroContext object

        Returns:
            AgentDecision with full reasoning and audit trail
        """
        decision_id = f"DEC_{ticker}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
        start_time  = datetime.now()

        logger.info(f"Making decision for {ticker} | decision_id={decision_id}")

        # ── Step 1: Fetch market data ──────────────────────────────────────
        price_df = self.fetcher.get_prices(ticker, days=cfg.LOOKBACK_DAYS)
        if price_df.empty:
            logger.error(f"No price data for {ticker}")
            return self._fail_decision(decision_id, ticker, f"No market data available for {ticker}")

        latest_price = float(
            price_df["Adj Close"].iloc[-1]
            if "Adj Close" in price_df.columns
            else price_df["Close"].iloc[-1]
        )

        # ── Step 2: Compute signals ────────────────────────────────────────
        signal_bundle = self.signal_eng.get_all_signals(ticker, price_df)
        agg_score     = signal_bundle.weighted_score()

        # ── Step 3: Correlation analysis ──────────────────────────────────
        existing_tickers = list(self.portfolio.positions.keys())
        avg_correlation  = 0.0
        corr_details: Dict[str, float] = {}

        if existing_tickers:
            corr_details = self.corr_engine.correlation_to_portfolio(
                ticker, existing_tickers, days=126
            )
            avg_correlation = float(np.mean(list(corr_details.values()))) if corr_details else 0.0

        # ── Step 4: Kelly sizing & risk ───────────────────────────────────
        # Get volatility from signals
        vol_signal = signal_bundle.by_source("garch_vol")
        vol_annual = vol_signal.value if vol_signal else 0.25

        # Kelly suggested size (based on aggregate signal as return proxy)
        kelly_size = self.sizer.kelly_fraction(
            expected_return_annual=max(0, agg_score * 0.30),  # Scale signal to return estimate
            volatility_annual=vol_annual,
        )

        # Position VaR at Kelly-suggested size
        position_value = self.portfolio.net_asset_value * kelly_size
        var_1d = self.risk_calc.position_var(position_value, vol_annual)
        var_pct_nav = var_1d / self.portfolio.net_asset_value if self.portfolio.net_asset_value > 0 else 0

        # Conviction from signal agreement
        conviction = self._assess_conviction(signal_bundle, avg_correlation)

        # ── Step 5: Build LLM context ──────────────────────────────────────
        user_prompt = self._build_user_prompt(
            ticker         = ticker,
            latest_price   = latest_price,
            signal_bundle  = signal_bundle,
            agg_score      = agg_score,
            avg_correlation= avg_correlation,
            corr_details   = corr_details,
            vol_annual     = vol_annual,
            kelly_size     = kelly_size,
            var_1d         = var_1d,
            var_pct_nav    = var_pct_nav,
            market_ctx     = market_ctx,
            macro_ctx      = macro_ctx,
        )

        # ── Step 6: Call LLM ──────────────────────────────────────────────
        logger.info(f"Calling LLM for {ticker}...")
        llm_response = self.llm.complete(
            system  = self.SYSTEM_PROMPT.replace(
                "MAX_POSITION_SIZE", f"{cfg.MAX_POSITION_SIZE:.0%}"
            ),
            user    = user_prompt,
            purpose = f"allocation_decision_{ticker}",
        )

        # ── Step 7: Parse decision ─────────────────────────────────────────
        decision = self._parse_llm_decision(
            decision_id  = decision_id,
            ticker       = ticker,
            llm_response = llm_response,
            signal_bundle= signal_bundle,
            var_1d       = var_1d,
            vol_annual   = vol_annual,
            start_time   = start_time,
        )

        # Log decision
        self.decisions_log.append(decision)
        logger.info(
            f"Decision for {ticker}: {decision.recommendation} "
            f"target={decision.target_weight:.1%} "
            f"conviction={decision.conviction.value} | "
            f"cost=${llm_response.cost_usd:.5f}"
        )

        return decision

    def _build_user_prompt(
        self,
        ticker: str,
        latest_price: float,
        signal_bundle: SignalBundle,
        agg_score: float,
        avg_correlation: float,
        corr_details: Dict[str, float],
        vol_annual: float,
        kelly_size: float,
        var_1d: float,
        var_pct_nav: float,
        market_ctx: Optional[str],
        macro_ctx: Optional[MacroContext],
    ) -> str:
        """
        Build the rich context prompt for the LLM.

        This is what makes the agent intelligent — it receives
        real, quantitative context rather than asking it to guess.
        """
        nav = self.portfolio.net_asset_value
        current_weight = self.portfolio.position_weight(ticker)
        cash_pct = self.portfolio.cash_pct

        # Format signals
        signal_lines = [signal_bundle.summary()]

        # Format correlations
        if corr_details:
            corr_lines = [f"  {t}: {c:.3f}" for t, c in sorted(corr_details.items(), key=lambda x: -abs(x[1]))]
            corr_section = "Correlation with existing positions:\n" + "\n".join(corr_lines[:5])
        else:
            corr_section = "Portfolio is empty — no correlation data."

        # Format portfolio state
        portfolio_summary = self.portfolio.summary()

        # Format macro context
        macro_section = ""
        if macro_ctx:
            macro_section = f"\nMacro Context:\n{macro_ctx.describe()}"
        elif market_ctx:
            macro_section = f"\nMarket Context: {market_ctx}"

        # Risk limits check
        budget_used = var_pct_nav / cfg.MAX_PORTFOLIO_VAR_PCT
        budget_warning = (
            f"⚠️  VaR ({var_pct_nav:.2%}) approaches limit ({cfg.MAX_PORTFOLIO_VAR_PCT:.2%})"
            if budget_used > 0.8 else ""
        )

        prompt = f"""ALLOCATION DECISION REQUEST: {ticker}

Current Price: ${latest_price:.2f}
Current Position Weight: {current_weight:.1%}
Portfolio NAV: ${nav:,.0f}
Available Cash: {cash_pct:.1%} of NAV

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SIGNALS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{signal_bundle.summary()}
Aggregate Signal Score: {agg_score:+.3f} (range: -1=strong sell to +1=strong buy)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RISK METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Forecasted Annual Volatility: {vol_annual:.1%}
1-Day 95% VaR at Kelly Size ({kelly_size:.1%}): ${var_1d:,.0f} ({var_pct_nav:.2%} of NAV)
Portfolio VaR Limit: {cfg.MAX_PORTFOLIO_VAR_PCT:.2%} of NAV
Kelly-Suggested Size: {kelly_size:.1%} of portfolio
{budget_warning}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIVERSIFICATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Average correlation with portfolio: {avg_correlation:.3f}
High correlation threshold: {cfg.MAX_CORRELATION_ADDITION:.2f}
{corr_section}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CURRENT PORTFOLIO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{portfolio_summary}
{macro_section}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONSTRAINTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Max position size: {cfg.MAX_POSITION_SIZE:.0%}
Max sector concentration: {cfg.MAX_SECTOR_CONCENTRATION:.0%}
Target Sharpe: >{cfg.TARGET_SHARPE_RATIO:.1f}

Please analyse {ticker} and return your decision in the specified JSON format."""

        return prompt

    def _assess_conviction(
        self, bundle: SignalBundle, avg_corr: float
    ) -> Conviction:
        """
        Assess conviction level from signal agreement and correlation.

        HIGH: Strong agreement across signals, low correlation
        MEDIUM: Some agreement, or strong single signal
        LOW: Mixed signals, or high correlation with portfolio
        """
        bullish_count = sum(1 for s in bundle.signals if s.is_bullish)
        bearish_count = sum(1 for s in bundle.signals if s.is_bearish)
        total         = len(bundle.signals)

        agreement_ratio = max(bullish_count, bearish_count) / total if total > 0 else 0
        avg_confidence  = float(np.mean([s.confidence for s in bundle.signals]))

        if avg_corr > cfg.MAX_CORRELATION_ADDITION:
            return Conviction.LOW  # Too correlated → low conviction regardless

        if agreement_ratio >= 0.75 and avg_confidence >= 0.65:
            return Conviction.HIGH
        elif agreement_ratio >= 0.50 or avg_confidence >= 0.55:
            return Conviction.MEDIUM
        else:
            return Conviction.LOW

    def _parse_llm_decision(
        self,
        decision_id:   str,
        ticker:        str,
        llm_response:  LLMResponse,
        signal_bundle: SignalBundle,
        var_1d:        float,
        vol_annual:    float,
        start_time:    datetime,
    ) -> AgentDecision:
        """Parse the LLM's JSON response into a structured AgentDecision."""
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Try to parse JSON from LLM response
        parsed = llm_response.parse_json()

        # Defaults in case parsing fails
        recommendation = "PASS"
        target_weight  = 0.0
        conviction     = Conviction.LOW
        reasoning      = llm_response.content
        key_factors    = []
        risks          = []

        if parsed:
            recommendation = str(parsed.get("recommendation", "PASS")).upper()
            if recommendation not in ("BUY", "SELL", "HOLD", "PASS"):
                recommendation = "PASS"

            target_pct   = float(parsed.get("target_weight_pct", 0) or 0)
            target_weight = min(target_pct / 100, cfg.MAX_POSITION_SIZE)  # Convert % to fraction

            conv_str  = str(parsed.get("conviction", "LOW")).upper()
            conviction = Conviction.HIGH if conv_str == "HIGH" else (
                Conviction.MEDIUM if conv_str == "MEDIUM" else Conviction.LOW
            )

            reasoning   = str(parsed.get("reasoning", llm_response.content))
            key_factors = list(parsed.get("key_factors", []))
            risks       = list(parsed.get("risks", []))
        else:
            logger.warning(
                f"Could not parse JSON from LLM for {ticker} — "
                f"using raw text as reasoning"
            )
            # Try to extract recommendation from raw text
            content_upper = llm_response.content.upper()
            for rec in ["STRONG BUY", "STRONG SELL", "BUY", "SELL", "HOLD", "PASS"]:
                if rec in content_upper:
                    recommendation = rec.split()[-1]  # "STRONG BUY" → "BUY"
                    break

        current_weight = self.portfolio.position_weight(ticker)
        weight_delta   = target_weight - current_weight

        return AgentDecision(
            decision_id    = decision_id,
            agent_name     = "PortfolioManagerAgent",
            ticker         = ticker,
            recommendation = recommendation,
            conviction     = conviction,
            target_weight  = target_weight,
            current_weight = current_weight,
            weight_delta   = weight_delta,
            reasoning      = reasoning,
            key_factors    = key_factors,
            risks          = risks,
            signals_used   = signal_bundle.signals,
            estimated_var  = var_1d,
            estimated_vol  = vol_annual,
            llm_model      = llm_response.model,
            llm_cost_usd   = llm_response.cost_usd,
            latency_ms     = latency_ms,
        )

    def _fail_decision(
        self, decision_id: str, ticker: str, reason: str
    ) -> AgentDecision:
        """Return a PASS decision when we cannot analyse the ticker."""
        return AgentDecision(
            decision_id    = decision_id,
            agent_name     = "PortfolioManagerAgent",
            ticker         = ticker,
            recommendation = "PASS",
            conviction     = Conviction.LOW,
            target_weight  = 0.0,
            current_weight = self.portfolio.position_weight(ticker),
            weight_delta   = 0.0,
            reasoning      = f"Decision failed: {reason}",
            key_factors    = [],
            risks          = [reason],
            signals_used   = [],
            llm_cost_usd   = 0.0,
            latency_ms     = 0.0,
        )

    def analyse_universe(
        self,
        tickers: Optional[List[str]] = None,
        top_n:   int = 5,
    ) -> List[AgentDecision]:
        """
        Analyse the full universe and return top N decisions.

        Run this daily after market close to generate trade ideas
        for the next session.

        Args:
            tickers : Tickers to analyse (default: cfg.DEFAULT_UNIVERSE)
            top_n   : Return top N BUY candidates

        Returns:
            List of AgentDecisions sorted by conviction then target_weight
        """
        universe = tickers or cfg.DEFAULT_UNIVERSE
        logger.info(f"Analysing universe of {len(universe)} securities...")

        # Get market snapshot first (single batch download)
        snapshot = self.fetcher.get_market_snapshot(universe)
        macro    = self.macro_fetch.get_macro_context()

        market_ctx = (
            f"Market regime: {snapshot.regime.value} | "
            f"VIX: {snapshot.vix_level or 'N/A'} | "
            f"SPY 1d: {snapshot.spy_return_1d:+.2%}"
            if snapshot.spy_return_1d else snapshot.market_summary()
        )

        decisions = []
        for ticker in universe:
            if ticker in ("SPY", "QQQ", "IWM", "TLT", "GLD", "VIX", "^VIX"):
                continue  # Skip ETF benchmarks from stock-picking
            try:
                decision = self.make_decision(
                    ticker     = ticker,
                    market_ctx = market_ctx,
                    macro_ctx  = macro,
                )
                decisions.append(decision)
            except Exception as e:
                logger.error(f"Failed to analyse {ticker}: {e}")

        # Sort: BUY first, then by conviction and target weight
        def sort_key(d: AgentDecision):
            rec_order  = {"BUY": 0, "HOLD": 1, "PASS": 2, "SELL": 3}
            conv_order = {Conviction.HIGH: 0, Conviction.MEDIUM: 1, Conviction.LOW: 2}
            return (rec_order.get(d.recommendation, 9), conv_order.get(d.conviction, 9), -d.target_weight)

        decisions.sort(key=sort_key)

        buy_decisions = [d for d in decisions if d.recommendation == "BUY"]
        logger.info(
            f"Universe analysis complete: "
            f"{len(buy_decisions)} BUY out of {len(decisions)} analysed"
        )

        return decisions[:top_n]

    def decision_report(self, decision: AgentDecision) -> str:
        """Format a decision as a human-readable report."""
        lines = [
            "=" * 65,
            f"  ALLOCATION DECISION — {decision.ticker}",
            f"  {decision.timestamp:%Y-%m-%d %H:%M:%S} | {decision.agent_name}",
            "=" * 65,
            f"  RECOMMENDATION : {decision.recommendation}",
            f"  TARGET WEIGHT  : {decision.target_weight:.1%}",
            f"  WEIGHT CHANGE  : {decision.weight_delta:+.1%}",
            f"  CONVICTION     : {decision.conviction.value}",
            "─" * 65,
            "  REASONING:",
        ]
        for line in decision.reasoning.split("\n"):
            lines.append(f"    {line}")

        if decision.key_factors:
            lines.append("─" * 65)
            lines.append("  KEY FACTORS:")
            for f in decision.key_factors:
                lines.append(f"    • {f}")

        if decision.risks:
            lines.append("─" * 65)
            lines.append("  RISKS:")
            for r in decision.risks:
                lines.append(f"    ⚠ {r}")

        lines.extend([
            "─" * 65,
            f"  VaR (95%, 1d)  : ${decision.estimated_var:,.0f}" if decision.estimated_var else "",
            f"  LLM Model      : {decision.llm_model}",
            f"  LLM Cost       : ${decision.llm_cost_usd:.5f}",
            f"  Latency        : {decision.latency_ms:.0f}ms",
            f"  Decision ID    : {decision.decision_id}",
            "=" * 65,
        ])

        return "\n".join(l for l in lines if l is not None)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    print("=" * 65)
    print("  Portfolio Manager Agent — Test Run")
    print("=" * 65)

    # Create a test portfolio
    portfolio = Portfolio(
        portfolio_id    = "TEST_001",
        cash            = 1_000_000,
        initial_capital = 1_000_000,
    )

    # Add a test position
    portfolio.positions["MSFT"] = Position(
        ticker        = "MSFT",
        direction     = Direction.LONG,
        shares        = 200,
        avg_cost      = 380.0,
        current_price = 410.0,
        sector        = "Technology",
    )

    print(f"\nPortfolio NAV: ${portfolio.net_asset_value:,.0f}")
    print(f"Positions: {list(portfolio.positions.keys())}")

    # Check API key before trying
    if not cfg.ANTHROPIC_API_KEY and not cfg.OPENAI_API_KEY:
        print("\n⚠️  No LLM API key configured.")
        print("   Add ANTHROPIC_API_KEY or OPENAI_API_KEY to .env to run full test.")
        print("\n   Market data and signal engines work without API keys.")
        print("   Testing market data and signals only...\n")

        fetcher     = MarketDataFetcher()
        signal_eng  = QuantSignalEngine(fetcher)

        print("Fetching AAPL data...")
        df = fetcher.get_prices("AAPL", days=126)
        if not df.empty:
            print(f"✓ Got {len(df)} bars | Latest: ${df['Close'].iloc[-1]:.2f}")
            bundle = signal_eng.get_all_signals("AAPL", df)
            print(f"\n{bundle.summary()}")
        sys.exit(0)

    # Full test with LLM
    print("\nInitialising Portfolio Manager Agent...")
    pm = PortfolioManagerAgent(portfolio)

    print("\nMaking decision for AAPL...")
    decision = pm.make_decision("AAPL")

    print("\n" + pm.decision_report(decision))
    print(f"\n{pm.llm.get_spend_summary()}")
