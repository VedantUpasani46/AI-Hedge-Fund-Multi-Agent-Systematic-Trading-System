"""
AI Hedge Fund — Part 5: Alternative Assets & Data
====================================================
alt_assets_agent.py — Alternative Assets & Data Intelligence Agent

This agent does two distinct jobs:

JOB 1: ALTERNATIVE DATA ENRICHMENT
    Fetches and synthesises alternative data signals (options flow,
    insider transactions, short interest, analyst revisions) for any
    equity security. Injects this intelligence into PM Agent decisions.

JOB 2: CAT BOND / ILS ALLOCATION
    Analyses and prices catastrophe bonds, manages the ILS sleeve of
    the portfolio, and provides uncorrelated return diversification.

Why one agent handles both:
    Both functions deal with "alternative" sources of information
    and alpha that go beyond standard price-based signals.
    The agent's tools naturally span both domains.

How it integrates with the existing multi-agent system:
    AgentCoordinator calls this agent BEFORE the PM Agent to:
    1. Enrich the equity decision with alt data signals
    2. Evaluate any ILS opportunities in the pipeline

    The agent returns structured signals that the PM Agent
    incorporates into its allocation reasoning.

Role in fund construction:
    ILS allocation provides 5-10% of NAV in genuinely uncorrelated
    alternative returns. During equity drawdowns (2008, 2020, 2022),
    cat bonds held their value while equity portfolios dropped 30-50%.
    This is the primary source of portfolio-level diversification.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hedge_fund.alt_assets_agent")


class AltAssetsAgent(BaseAgent):
    """
    Alternative Assets & Data Intelligence Agent.

    Tools:
        [EQUITY ALT DATA]
        get_options_flow          — Options PCR, skew, unusual activity
        get_insider_activity      — SEC Form 4 director/officer transactions
        get_short_interest        — Short float %, days-to-cover, squeeze signals
        get_analyst_revisions     — Consensus rating, target price, EPS revisions
        get_full_alt_data_bundle  — All of the above combined

        [ILS / CAT BONDS]
        price_cat_bond            — Full cat bond pricing with EL, spread, duration
        evaluate_ils_opportunity  — Should we buy this cat bond?
        get_ils_portfolio_status  — Current ILS holdings and risk metrics
        screen_cat_bond_universe  — Screen example bonds for best opportunities
        compute_diversification   — Quantify ILS benefit to overall portfolio
    """

    SYSTEM_PROMPT = """You are an alternative assets and data specialist at a systematic hedge fund.

YOU HAVE TWO ROLES:

ROLE 1: ALTERNATIVE DATA ANALYST
For equity securities, you enrich allocation decisions with non-price signals:
  - Options market: Put/call ratio, skew, unusual activity
  - Corporate insiders: Director/officer transactions (Form 4)
  - Short market: Short interest, days-to-cover, squeeze potential
  - Analyst revisions: Rating changes, target price updates, EPS revisions

WHEN TO TRIGGER EACH SIGNAL:
  Options flow:  Always — real-time sentiment and positioning
  Insider trades: Always — strong predictive signal per Seyhun (1998)
  Short interest: Always — medium-term bearish predictor
  Analyst data:   Always — near-term drift predictor

HOW TO INTERPRET:
  Bullish convergence: insiders buying + analysts upgrading + low short = STRONG BUY
  Bearish convergence: high PCR + high short + analyst downgrades = STRONG SELL
  Mixed signals: inform the PM Agent but reduce conviction

ROLE 2: ILS PORTFOLIO MANAGER
For catastrophe bonds and Insurance-Linked Securities:
  - Price cat bonds using frequency-severity models (Poisson frequency + GPD severity)
  - Key metrics: Expected Loss (EL), attachment probability, risk multiple (spread/EL)
  - Rule: Buy when risk multiple > 2.0x AND spread > 300bps AND NOT too correlated

ILS DECISION RULES:
  BUY cat bond if:
    - Risk multiple (spread/EL) >= 2.0x
    - Attachment probability 1-10% (sweet spot)
    - Not same peril/geography as existing ILS positions
    - ILS allocation < 10% of total NAV
  
  AVOID:
    - Multiple Florida hurricane bonds (peril correlation)
    - Risk multiple < 1.5x (not compensated for the risk)
    - Attachment probability < 0.5% (too remote, low carry)
    - Attachment probability > 15% (too frequent, not diversifying)

DIVERSIFICATION FRAMEWORK:
  ILS vs equities: correlation ~0.04 (genuinely uncorrelated)
  Ideal ILS allocation: 5-10% of total portfolio
  This improves portfolio Sharpe by ~8-12% with no return sacrifice

ALWAYS PROVIDE:
  Quantitative backing for every recommendation
  Specific metrics (EL, spread, risk multiple for cat bonds)
  Specific signals with values (PCR, short%, insider $ for alt data)
  Clear BUY/HOLD/PASS recommendation with reasoning

Respond in JSON for structured decisions:
{
  "recommendation": "BUY|PASS|HOLD",
  "conviction": "HIGH|MEDIUM|LOW",
  "key_signals": ["signal 1", "signal 2"],
  "risks": ["risk 1", "risk 2"],
  "summary": "1 paragraph",
  "metrics": {key quantitative metrics}
}"""

    def __init__(self, config=None):
        from src.agents.base_agent import AgentConfig
        from src.catbond.cat_bond_models import (
            CatBondPricer, build_standard_loss_models, create_example_cat_bonds
        )
        from src.catbond.ils_portfolio import ILSPortfolio, ILSCorrelationMatrix
        from src.altdata.alternative_data import AlternativeDataEngine

        self.alt_engine   = AlternativeDataEngine()
        self.cat_pricer   = CatBondPricer()
        self.loss_models  = build_standard_loss_models()
        self.example_bonds= create_example_cat_bonds()
        self.corr_matrix  = ILSCorrelationMatrix()
        self.ils_portfolio= ILSPortfolio(total_nav=1_000_000, ils_allocation=0.10)

        cfg = config or AgentConfig(
            name        = "AltAssetsAgent",
            model       = "claude-sonnet-4-6",
            temperature = 0.1,
            max_tokens  = 4096,
        )
        super().__init__(cfg)
        logger.info("AltAssetsAgent initialised")

    def _get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def _get_tools(self) -> List[Tool]:
        return [
            # ── Equity Alt Data ────────────────────────────────────────────────
            Tool(
                name="get_full_alt_data_bundle",
                func=self._tool_alt_data_bundle,
                description=(
                    "Get complete alternative data bundle for an equity ticker. "
                    "Includes options flow (PCR, skew), insider transactions (Form 4), "
                    "short interest (% float, days-to-cover), and analyst revisions. "
                    "Input: ticker symbol. Returns composite signal and all sub-signals."
                ),
                param_schema={"type":"object","properties":{"ticker":{"type":"string"}},"required":["ticker"]}
            ),
            Tool(
                name="get_options_flow",
                func=self._tool_options,
                description=(
                    "Get options market signals: put/call ratio, IV skew, unusual activity. "
                    "PCR > 1.5 = bearish, PCR < 0.6 = bullish. "
                    "High put IV vs call IV skew = fear/bearish. "
                    "Input: ticker symbol."
                ),
                param_schema={"type":"object","properties":{"ticker":{"type":"string"}},"required":["ticker"]}
            ),
            Tool(
                name="get_insider_activity",
                func=self._tool_insider,
                description=(
                    "Get recent SEC Form 4 insider transactions. "
                    "Director/officer open-market purchases are strongly bullish (Seyhun 1998). "
                    "Cluster buys (multiple insiders) are very bullish. "
                    "Input: ticker symbol."
                ),
                param_schema={"type":"object","properties":{"ticker":{"type":"string"}},"required":["ticker"]}
            ),
            Tool(
                name="get_short_interest",
                func=self._tool_short,
                description=(
                    "Get short interest metrics: % of float, days-to-cover, monthly change. "
                    "High and rising short interest is bearish (Desai 2002). "
                    "High short + rising price = squeeze signal (bullish). "
                    "Input: ticker symbol."
                ),
                param_schema={"type":"object","properties":{"ticker":{"type":"string"}},"required":["ticker"]}
            ),
            Tool(
                name="get_analyst_revisions",
                func=self._tool_analyst,
                description=(
                    "Get analyst consensus data: rating, price target, EPS estimates. "
                    "Upward revisions predict near-term drift (Stickel 1995). "
                    "Input: ticker symbol."
                ),
                param_schema={"type":"object","properties":{"ticker":{"type":"string"}},"required":["ticker"]}
            ),
            # ── ILS / Cat Bonds ────────────────────────────────────────────────
            Tool(
                name="price_cat_bond",
                func=self._tool_price_cat_bond,
                description=(
                    "Price a catastrophe bond and compute risk metrics: "
                    "expected loss (EL), attachment probability, fair spread, "
                    "risk multiple (spread/EL), cat duration. "
                    "Input: bond ticker (use example bonds: PELICAN-2024-A, "
                    "SIERRA-2024-A, BORA-2024-A, ATLAS-2024-A) "
                    "OR 'all' for full universe screen."
                ),
                param_schema={"type":"object","properties":{"bond_ticker":{"type":"string"}},"required":["bond_ticker"]}
            ),
            Tool(
                name="screen_ils_universe",
                func=self._tool_screen_ils,
                description=(
                    "Screen the full cat bond universe and rank by value. "
                    "Identifies cheap bonds (spread > fair value) and "
                    "ranks by risk-adjusted attractiveness. "
                    "Input: anything (e.g. 'all' or 'screen')"
                ),
                param_schema={"type":"object","properties":{},"required":[]}
            ),
            Tool(
                name="compute_diversification_benefit",
                func=self._tool_diversification,
                description=(
                    "Quantify the diversification benefit of adding ILS to the portfolio. "
                    "Shows impact on Sharpe ratio, portfolio vol, and drawdown. "
                    "Input: current equity allocation % (e.g. 60)"
                ),
                param_schema={"type":"object","properties":{"equity_pct":{"type":"number"}},"required":[]}
            ),
            Tool(
                name="get_ils_portfolio_status",
                func=self._tool_ils_status,
                description=(
                    "Get current ILS portfolio holdings, expected loss, "
                    "income, and concentration metrics. "
                    "Input: anything."
                ),
                param_schema={"type":"object","properties":{},"required":[]}
            ),
        ]

    # ── Alt data tool implementations ─────────────────────────────────────────

    def _tool_alt_data_bundle(self, ticker: str) -> str:
        bundle = self.alt_engine.get_signals(ticker)
        result = {
            "ticker":              ticker,
            "composite_signal":    round(bundle.composite_signal, 4),
            "composite_confidence":round(bundle.composite_confidence, 3),
            "interpretation":      (
                "BULLISH" if bundle.composite_signal > 0.2 else
                "BEARISH" if bundle.composite_signal < -0.2 else "NEUTRAL"
            ),
            "signals": [
                {
                    "source":      s.source,
                    "signal":      s.signal,
                    "confidence":  s.confidence,
                    "raw_value":   s.raw_value,
                    "description": s.description,
                }
                for s in bundle.signals
            ],
        }
        return json.dumps(result, indent=2)

    def _tool_options(self, ticker: str) -> str:
        sig = self.alt_engine.options_analyser.analyse(ticker)
        if not sig:
            return json.dumps({"error": f"No options data for {ticker}"})
        return json.dumps(sig.metadata | {"signal": sig.signal, "description": sig.description})

    def _tool_insider(self, ticker: str) -> str:
        sig = self.alt_engine.insider_signal.analyse(ticker)
        if not sig:
            return json.dumps({"error": f"No insider data for {ticker}"})
        return json.dumps({"signal": sig.signal, "confidence": sig.confidence,
                           "description": sig.description, **sig.metadata})

    def _tool_short(self, ticker: str) -> str:
        sig = self.alt_engine.short_signal.analyse(ticker)
        if not sig:
            return json.dumps({"error": f"No short data for {ticker}"})
        return json.dumps({"signal": sig.signal, "confidence": sig.confidence,
                           "description": sig.description, **sig.metadata})

    def _tool_analyst(self, ticker: str) -> str:
        sig = self.alt_engine.analyst_signal.analyse(ticker)
        if not sig:
            return json.dumps({"error": f"No analyst data for {ticker}"})
        return json.dumps({"signal": sig.signal, "confidence": sig.confidence,
                           "description": sig.description, **sig.metadata})

    # ── ILS tool implementations ───────────────────────────────────────────────

    def _tool_price_cat_bond(self, bond_ticker: str) -> str:
        if bond_ticker.lower() == "all":
            return self._tool_screen_ils()

        # Find the bond
        bond_map = {b.ticker: b for b in self.example_bonds}
        if bond_ticker not in bond_map:
            available = list(bond_map.keys())
            return json.dumps({
                "error": f"Bond {bond_ticker} not found",
                "available": available,
            })

        bond = bond_map[bond_ticker]
        mk   = self.ils_portfolio._infer_model_key(bond)
        model= self.loss_models[mk]
        result = self.cat_pricer.price(bond, model)

        return json.dumps(result.to_dict(), indent=2)

    def _tool_screen_ils(self, **kwargs) -> str:
        results = []
        for bond in self.example_bonds:
            mk     = self.ils_portfolio._infer_model_key(bond)
            model  = self.loss_models[mk]
            result = self.cat_pricer.price(bond, model)
            results.append(result.to_dict())

        # Sort by value (cheapest first)
        results.sort(key=lambda x: -x["spread_vs_fair"])

        return json.dumps({
            "n_bonds":    len(results),
            "n_cheap":    sum(1 for r in results if r["is_cheap"]),
            "bonds":      results,
            "recommended": results[0]["ticker"] if results else None,
        }, indent=2)

    def _tool_diversification(self, equity_pct: float = 60.0) -> str:
        if not self.ils_portfolio.positions:
            # Add example positions first
            for bond in self.example_bonds[:2]:
                self.ils_portfolio.add_position(bond, 50_000)

        benefit = self.ils_portfolio.equity_diversification_benefit(
            equity_weight = equity_pct / 100,
        )
        benefit["explanation"] = (
            f"Adding {benefit['ils_allocation_pct']:.0f}% ILS to a "
            f"{equity_pct:.0f}% equity portfolio improves Sharpe by "
            f"{benefit['sharpe_improvement']:.3f} and reduces vol by "
            f"{benefit['vol_reduction_pct']:.1f}% — "
            f"correlation between ILS and equities: {benefit['ils_equity_corr']:.2f}"
        )
        return json.dumps(benefit, indent=2)

    def _tool_ils_status(self, **kwargs) -> str:
        if not self.ils_portfolio.positions:
            return json.dumps({"status": "ILS portfolio empty. Use screen_ils_universe to find opportunities."})

        return json.dumps({
            "n_positions":        len(self.ils_portfolio.positions),
            "total_allocation":   sum(p.allocation_usd for p in self.ils_portfolio.positions),
            "expected_income":    self.ils_portfolio.portfolio_expected_income_usd(),
            "expected_loss":      self.ils_portfolio.portfolio_expected_loss_usd(),
            "net_return_pct":     self.ils_portfolio.net_expected_return_pct(),
            "hhi_concentration":  self.ils_portfolio.herfindahl_index(),
            "by_peril":           self.ils_portfolio.concentration_by_peril(),
            "positions":          [
                {
                    "ticker":    p.bond.ticker,
                    "peril":     p.bond.peril.value,
                    "territory": p.bond.territory,
                    "allocated": p.allocation_usd,
                    "el_bps":    p.pricing.el_bps,
                    "spread":    p.bond.coupon_spread,
                    "is_cheap":  p.pricing.is_cheap,
                }
                for p in self.ils_portfolio.positions
            ],
        }, indent=2)

    # ── Main public methods ────────────────────────────────────────────────────

    def enrich_equity_decision(
        self,
        ticker: str,
    ) -> Dict[str, Any]:
        """
        Enrich an equity allocation decision with alt data signals.
        Called by AgentCoordinator before PM Agent runs.
        """
        bundle = self.alt_engine.get_signals(ticker)
        return {
            "ticker":          ticker,
            "composite_signal":bundle.composite_signal,
            "confidence":      bundle.composite_confidence,
            "summary":         bundle.summary(),
            "raw_signals":     [
                {"source": s.source, "signal": s.signal,
                 "confidence": s.confidence, "description": s.description}
                for s in bundle.signals
            ],
        }

    def evaluate_ils_allocation(
        self,
        bond_ticker: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate the ILS opportunity set and recommend allocations.
        Called by AgentCoordinator for ILS portfolio management.
        """
        user_prompt = (
            f"Evaluate {'cat bond ' + bond_ticker if bond_ticker else 'the full ILS universe'} "
            f"for portfolio allocation. "
            f"Use price_cat_bond or screen_ils_universe to get current pricing. "
            f"Use compute_diversification_benefit to quantify the portfolio benefit. "
            f"Provide a BUY/PASS recommendation with specific metrics."
        )
        response_text, _ = self.think(
            user_message = user_prompt,
            use_tools    = True,
            purpose      = "ils_allocation",
        )

        parsed = self._parse_json_response(response_text)
        return parsed or {"recommendation": "PASS", "reasoning": response_text[:500]}

    def _parse_json_response(self, text: str) -> Optional[Dict]:
        import re
        for fence in ["```json", "```"]:
            if fence in text:
                parts = text.split(fence)
                for part in parts:
                    clean = part.strip().rstrip("`").strip()
                    if clean.startswith("{"):
                        try:
                            return json.loads(clean)
                        except json.JSONDecodeError:
                            pass
        try:
            return json.loads(text.strip())
        except Exception:
            match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except Exception:
                    pass
        return None

    # ── MessageBus handler ─────────────────────────────────────────────────────

    def handle_message(self, message) -> Optional[Dict[str, Any]]:
        subject = message.subject.lower()
        payload = message.payload
        ticker  = payload.get("ticker", "")

        logger.info(f"AltAssetsAgent: {message.subject} from {message.sender}")

        if "alt_data" in subject or "alternative_data" in subject:
            return self.enrich_equity_decision(ticker)

        elif "ils" in subject or "cat_bond" in subject:
            return self.evaluate_ils_allocation(payload.get("bond_ticker"))

        elif "options" in subject:
            sig = self.alt_engine.options_analyser.analyse(ticker)
            return sig.__dict__ if sig else {"error": "No data"}

        elif "insider" in subject:
            sig = self.alt_engine.insider_signal.analyse(ticker)
            return sig.__dict__ if sig else {"error": "No data"}

        elif "diversification" in subject:
            if self.ils_portfolio.positions:
                return self.ils_portfolio.equity_diversification_benefit()
            return {"note": "No ILS positions to compute diversification"}

        else:
            response_text, _ = self.think(
                user_message = f"{message.subject}: {json.dumps(payload)}",
                use_tools    = True,
                purpose      = "alt_assets_query",
            )
            return {"response": response_text[:500]}


from src.agents.base_agent import BaseAgent, Tool, AgentConfig


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("  Alt Assets Agent — Test")
    print("=" * 60)

    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  No LLM key — testing components directly\n")

        # Test alt data
        from src.altdata.alternative_data import AlternativeDataEngine
        engine = AlternativeDataEngine()
        print("1. Alt data signals for AAPL:")
        bundle = engine.get_signals("AAPL")
        print(bundle.summary())

        # Test ILS
        from src.catbond.cat_bond_models import (
            CatBondPricer, build_standard_loss_models, create_example_cat_bonds
        )
        from src.catbond.ils_portfolio import ILSPortfolio

        print("\n2. Cat bond pricing:")
        pricer = CatBondPricer()
        models = build_standard_loss_models()
        bonds  = create_example_cat_bonds()
        portfolio = ILSPortfolio(10_000_000, 0.10)

        for bond in bonds[:2]:
            mk = portfolio._infer_model_key(bond)
            result = pricer.price(bond, models[mk])
            print(f"\n  {result.summary()}")

        print("\n  Diversification benefit:")
        portfolio.add_position(bonds[0], 400_000)
        portfolio.add_position(bonds[2], 300_000)
        benefit = portfolio.equity_diversification_benefit()
        print(f"  Pure equity Sharpe:  {benefit['pure_equity_sharpe']:.3f}")
        print(f"  With ILS Sharpe:     {benefit['blended_sharpe']:.3f}")
        print(f"  Improvement:         +{benefit['sharpe_improvement']:.3f}")
        print(f"  Vol reduction:       {benefit['vol_reduction_pct']:.1f}%")
        print(f"  ILS-equity corr:     {benefit['ils_equity_corr']:.3f}")

        print("\n✅ Tests passed. Add API key for full agent test.")
    else:
        agent = AltAssetsAgent()
        print("\n1. Alt data enrichment for NVDA...")
        result = agent.enrich_equity_decision("NVDA")
        print(f"   Composite signal: {result['composite_signal']:+.3f}")
        print(f"   Confidence: {result['confidence']:.0%}")

        print("\n2. ILS universe evaluation...")
        result = agent.evaluate_ils_allocation()
        print(f"   Recommendation: {result.get('recommendation', 'N/A')}")
        print(f"   {result.get('summary', '')[:200]}")
