"""
AI Hedge Fund — Part 8: Investor Dashboard
============================================
dashboard_agent.py — LLM-Powered Dashboard Intelligence

The Dashboard Agent adds the intelligence layer on top of the
data API. It does three things that raw numbers cannot:

1. MARKET COMMENTARY GENERATION
   Writes the monthly investor letter commentary in natural
   language. Reads the performance data, interprets what drove
   returns, identifies the main risk events of the month, and
   writes a coherent narrative.

   This is the text that goes in the "Portfolio Commentary"
   section of the investor letter — the part that distinguishes
   a professional fund from a data dump.

2. NATURAL LANGUAGE INVESTOR QUERIES
   Investors can ask questions like:
     "Why is the portfolio down today?"
     "What is our biggest risk right now?"
     "Compare our performance to SPY this quarter"
     "Is the momentum factor working?"

   The agent reads the live data, interprets it, and answers
   in plain English with specific numbers.

3. AUTOMATED ALERTS AND SUMMARIES
   When the market closes, the agent produces a one-paragraph
   end-of-day summary and routes it to the investor portal.
   When a circuit breaker fires, it explains what happened
   in terms a non-quant investor can understand.

Integration:
    Called by the API server (/commentary, /query endpoints)
    Called by the report generator for the letter commentary
    Runs on-demand — not on a background thread
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, date
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hedge_fund.dashboard_agent")


class DashboardAgent(BaseAgent):
    """
    LLM-powered dashboard intelligence agent.

    Turns raw portfolio/risk data into investor-readable insights.
    """

    SYSTEM_PROMPT = """You are the investor relations and portfolio commentary specialist at a systematic hedge fund.

YOUR ROLE:
Bridge the gap between quantitative data and investor communication.
Translate numbers, risk metrics, and factor exposures into clear,
professional language that sophisticated investors can understand.

COMMUNICATION STANDARDS:
- Write like a professional fund manager, not a quant developer
- Be specific: use actual numbers, not vague terms like "performed well"
- Distinguish facts (reported results) from analysis (your interpretation)
- Be honest about underperformance — investors respect transparency
- Never guarantee future performance or make promises

WHEN WRITING COMMENTARY:
1. Lead with the most important point (return, major event)
2. Explain what drove performance (factor attribution, top positions)
3. Address risk: what the metrics say and what they mean
4. Discuss positioning going forward (without making predictions)
5. Keep it to 2-4 short paragraphs — investors are busy

WHEN ANSWERING INVESTOR QUESTIONS:
- Use get_portfolio_data and get_risk_data tools to get current numbers
- Answer the question directly in the first sentence
- Back up with specific data
- If you don't know something, say so clearly

TONE:
- Professional, clear, confident
- Appropriate use of financial terminology (assume sophisticated reader)
- Not overly technical (avoid "kappa" and "characteristic function")
- Not promotional — state facts, let numbers speak

Available tools give you access to:
- Current portfolio (NAV, positions, weights, P&L)
- Risk metrics (VaR, beta, drawdown, factor exposures)
- Performance data (Sharpe, return, vol, monthly table)
- Recent trades and execution data"""

    def __init__(self, api_state=None, config=None):
        from src.agents.base_agent import AgentConfig
        from src.api.api_server import get_state

        self._api_state = api_state or get_state()

        cfg = config or AgentConfig(
            name        = "DashboardAgent",
            model       = "claude-sonnet-4-6",
            temperature = 0.25,    # Slightly higher — commentary benefits from some variation
            max_tokens  = 2048,
        )
        super().__init__(cfg)

    def _get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def _get_tools(self) -> List[Tool]:
        return [
            Tool(
                name = "get_portfolio_data",
                func = self._tool_portfolio,
                description = (
                    "Get current portfolio data: NAV, positions, weights, "
                    "unrealised P&L, sector allocation. "
                    "Always call this before commenting on portfolio performance."
                ),
                param_schema={"type":"object","properties":{},"required":[]}
            ),
            Tool(
                name = "get_risk_data",
                func = self._tool_risk,
                description = (
                    "Get current risk metrics: VaR (95% and 99%), portfolio beta, "
                    "intraday and trailing drawdown, circuit breaker status. "
                    "Call this before commenting on risk."
                ),
                param_schema={"type":"object","properties":{},"required":[]}
            ),
            Tool(
                name = "get_performance_data",
                func = self._tool_performance,
                description = (
                    "Get full performance statistics: total return, annual return, "
                    "Sharpe ratio, Sortino ratio, max drawdown, Calmar ratio, hit rate. "
                    "Call this for performance commentary or investor questions about returns."
                ),
                param_schema={"type":"object","properties":{},"required":[]}
            ),
            Tool(
                name = "get_monthly_returns",
                func = self._tool_monthly,
                description = (
                    "Get monthly return table (year × month grid). "
                    "Useful for discussing performance patterns over time."
                ),
                param_schema={"type":"object","properties":{},"required":[]}
            ),
            Tool(
                name = "get_recent_trades",
                func = self._tool_trades,
                description = (
                    "Get recent execution history: trades, fill prices, "
                    "implementation shortfall. "
                    "Use for questions about positioning changes."
                ),
                param_schema={"type":"object","properties":{},"required":[]}
            ),
            Tool(
                name = "get_factor_exposures",
                func = self._tool_factors,
                description = (
                    "Get current factor betas: market (MKT), size (SMB), "
                    "value (HML), momentum (MOM). "
                    "Use for questions about factor positioning or style drift."
                ),
                param_schema={"type":"object","properties":{},"required":[]}
            ),
        ]

    # ── Tool implementations ──────────────────────────────────────────────────

    def _tool_portfolio(self, **kwargs) -> str:
        return json.dumps(self._api_state.get_portfolio_dict(), indent=2)

    def _tool_risk(self, **kwargs) -> str:
        return json.dumps(self._api_state.get_risk_dict(), indent=2)

    def _tool_performance(self, **kwargs) -> str:
        return json.dumps(self._api_state.get_performance_dict(), indent=2)

    def _tool_monthly(self, **kwargs) -> str:
        return json.dumps(self._api_state.get_monthly_returns(), indent=2)

    def _tool_trades(self, **kwargs) -> str:
        return json.dumps(self._api_state.get_trades(limit=20), indent=2)

    def _tool_factors(self, **kwargs) -> str:
        if self._api_state.factor_monitor:
            port      = self._api_state.get_portfolio_dict()
            positions = {p["ticker"]: p["market_value"] for p in port.get("positions", [])}
            nav       = port.get("nav", 1_000_000)
            snap      = self._api_state.factor_monitor.compute(positions, nav)
            return json.dumps({
                "r_squared":  snap.r_squared,
                "alpha_daily":snap.alpha_estimate,
                "factors": {
                    n: {"beta": e.portfolio_beta, "status": e.status()}
                    for n, e in snap.exposures.items()
                },
            }, indent=2)
        return json.dumps({"note": "Factor monitor not available"})

    # ── Main public methods ───────────────────────────────────────────────────

    def generate_monthly_commentary(self, period: str = None) -> str:
        """
        Generate the monthly investor letter commentary.

        Returns professional prose suitable for the investor letter.
        """
        period = period or datetime.now().strftime("%B %Y")

        prompt = f"""Write the portfolio commentary section for our {period} investor letter.

Use your tools to get the current performance, risk, and positioning data.
Then write 3 paragraphs:

1. Performance overview (how did we do this month, what drove returns)
2. Risk and positioning (what the risk metrics show, how we're positioned)
3. Looking ahead (factor positioning, any notable changes, no predictions)

Style: Professional fund manager letter. Specific numbers. Honest about both
positives and any challenges. 2-3 sentences per paragraph maximum.
Do NOT include a subject line or greeting — just the commentary text itself."""

        response, _ = self.think(
            user_message = prompt,
            use_tools    = True,
            purpose      = "monthly_commentary",
        )
        return response

    def answer_investor_query(self, question: str) -> str:
        """
        Answer an investor question using live data.

        Args:
            question: Natural language question from an investor

        Returns:
            Professional, data-backed answer
        """
        prompt = f"""An investor has asked:

"{question}"

Use your tools to get the relevant data, then answer clearly and professionally.
Lead with the direct answer in the first sentence.
Back up with specific numbers from the data.
Keep the response to 2-4 sentences unless the question requires more detail."""

        response, _ = self.think(
            user_message = prompt,
            use_tools    = True,
            purpose      = "investor_query",
        )
        return response

    def generate_eod_summary(self) -> str:
        """
        Generate an end-of-day portfolio summary.

        Suitable for the daily update email to investors or
        internal team communication.
        """
        prompt = """Generate a concise end-of-day portfolio summary.

Use your tools to get today's data. Include:
- Today's P&L (in $ and %)
- Key movers (top contributors and detractors)
- Risk status (VaR vs limit, any circuit breaker events)
- One notable observation about positioning or market context

Keep it to 4-6 bullet points or 2 short paragraphs.
This goes to investors — professional tone, specific numbers."""

        response, _ = self.think(
            user_message = prompt,
            use_tools    = True,
            purpose      = "eod_summary",
        )
        return response

    def explain_circuit_breaker(self, breaker_name: str, metrics: Dict) -> str:
        """
        Write an investor-friendly explanation of a circuit breaker event.

        These go in the alert email when a hard limit is breached.
        """
        prompt = f"""A risk management circuit breaker was triggered:
Breaker: {breaker_name}
Metrics at time of trigger: {json.dumps(metrics, indent=2)}

Write a clear, calm explanation (2-3 sentences) for investors explaining:
1. What happened (in plain English, not quant jargon)
2. What action was taken automatically
3. That the risk management system is working as designed

Tone: calm and professional. Not alarming. Not dismissive.
This email will go to limited partners."""

        response, _ = self.think(
            user_message = prompt,
            use_tools    = False,   # No tools needed — metrics already in prompt
            purpose      = "circuit_breaker_explanation",
        )
        return response

    # ── MessageBus handler ─────────────────────────────────────────────────────

    def handle_message(self, message) -> Optional[Dict[str, Any]]:
        subject = message.subject.lower()
        payload = message.payload
        logger.info(f"DashboardAgent: {message.subject}")

        if "commentary" in subject:
            period = payload.get("period", datetime.now().strftime("%B %Y"))
            return {"commentary": self.generate_monthly_commentary(period)}

        elif "investor_query" in subject or "question" in subject:
            question = payload.get("question", "")
            return {"answer": self.answer_investor_query(question)}

        elif "eod_summary" in subject:
            return {"summary": self.generate_eod_summary()}

        elif "circuit_breaker" in subject:
            name    = payload.get("breaker", "UNKNOWN")
            metrics = payload.get("metrics", {})
            return {"explanation": self.explain_circuit_breaker(name, metrics)}

        else:
            # Generic query
            response, _ = self.think(
                user_message = f"{message.subject}: {json.dumps(payload)}",
                use_tools    = True,
                purpose      = "dashboard_query",
            )
            return {"response": response}


from src.agents.base_agent import BaseAgent, Tool, AgentConfig


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("  Dashboard Agent — Test")
    print("=" * 60)

    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  No LLM key — showing demo data only\n")
        from src.api.api_server import get_state, _demo_portfolio, _demo_risk, _demo_performance
        state = get_state()
        print("Portfolio:")
        print(json.dumps(_demo_portfolio(), indent=2)[:800])
    else:
        agent = DashboardAgent()

        print("\n1. Answering investor query...")
        answer = agent.answer_investor_query(
            "How is the portfolio performing and what are the main risks right now?"
        )
        print(f"\nAnswer:\n{answer}")

        print("\n2. Generating monthly commentary...")
        commentary = agent.generate_monthly_commentary()
        print(f"\nCommentary:\n{commentary}")
