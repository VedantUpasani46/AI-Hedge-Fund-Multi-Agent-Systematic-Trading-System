"""
AI Hedge Fund — Part 4: Execution Engine
==========================================
execution_agent.py — AI-Driven Execution Agent

The Execution Agent sits between allocation decisions and the broker.
Its job is NOT to decide what to buy or sell (that's the PM Agent).
Its job is to figure out HOW to buy or sell it optimally.

Questions the Execution Agent answers:
    - What algorithm should we use? (IS, TWAP, VWAP, market?)
    - How should we slice this order given current market conditions?
    - Is now a good time to execute or should we wait?
    - Is the market too volatile? Should we use a limit order instead?
    - Are we getting unusual slippage? Should we pause?

This is where the LLM adds real value beyond pure quant:
    Pure Almgren-Chriss gives you the optimal schedule in a stationary world.
    The LLM can read the current market context:
        "VIX just spiked to 35 → widen the execution window"
        "Earnings report in 20 minutes → execute now or hold"
        "Volume is 2x normal → can trade larger slices with less impact"
        "Bid-ask spread has widened significantly → use limit orders"

Communication in the multi-agent system:
    Receives from Coordinator: execute_order
    Sends to OrderManager:     submit child orders
    Sends to PM Agent:         execution_complete (with TCA results)
    Sends to Risk Manager:     position_updated
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hedge_fund.execution_agent")


@dataclass
class ExecutionContext:
    """
    Market context for the Execution Agent's decision-making.
    Built from real-time market data before each execution decision.
    """
    ticker:           str
    current_price:    float
    bid:              Optional[float] = None
    ask:              Optional[float] = None
    spread_bps:       Optional[float] = None
    volume_today:     Optional[float] = None
    volume_vs_avg:    Optional[float] = None   # Today's volume / 20d average
    volatility_today: Optional[float] = None   # Realized vol today
    volatility_21d:   Optional[float] = None   # 21-day realized vol
    vix:              Optional[float] = None
    market_regime:    str = "UNKNOWN"
    minutes_to_close: Optional[float] = None
    has_news:         bool = False
    earnings_days_away: Optional[int] = None

    def urgency_assessment(self) -> str:
        """Determine execution urgency from market context."""
        if self.vix and self.vix > 35:
            return "HIGH"   # High vol → execute quickly, reduce timing risk
        if self.volume_vs_avg and self.volume_vs_avg > 2.0:
            return "HIGH"   # High volume day → good liquidity, execute now
        if self.earnings_days_away is not None and self.earnings_days_away <= 1:
            return "URGENT" # Earnings tomorrow → don't hold overnight risk
        if self.minutes_to_close and self.minutes_to_close < 30:
            return "URGENT" # Near close → MOC or execute now
        if self.spread_bps and self.spread_bps > 20:
            return "LOW"    # Wide spread → be patient, use limit orders
        return "NORMAL"

    def to_prompt_context(self) -> str:
        """Format as LLM-readable context string."""
        lines = [f"Current market context for {self.ticker}:"]
        lines.append(f"  Price: ${self.current_price:.4f}")
        if self.spread_bps:
            lines.append(f"  Bid-ask spread: {self.spread_bps:.1f}bps")
        if self.volume_vs_avg:
            lines.append(
                f"  Volume: {self.volume_vs_avg:.1f}x normal "
                f"({'elevated' if self.volume_vs_avg > 1.5 else 'normal'})"
            )
        if self.volatility_today:
            lines.append(
                f"  Today's vol: {self.volatility_today:.1%} "
                f"(21d: {self.volatility_21d:.1%})"
                if self.volatility_21d else
                f"  Today's vol: {self.volatility_today:.1%}"
            )
        if self.vix:
            regime = "HIGH" if self.vix > 30 else "ELEVATED" if self.vix > 20 else "LOW"
            lines.append(f"  VIX: {self.vix:.1f} ({regime})")
        if self.earnings_days_away is not None:
            lines.append(f"  Earnings in: {self.earnings_days_away} days")
        if self.minutes_to_close:
            lines.append(f"  Minutes to market close: {self.minutes_to_close:.0f}")
        lines.append(f"  Recommended urgency: {self.urgency_assessment()}")
        return "\n".join(lines)


class ExecutionAgent(BaseAgent):
    """
    AI-driven Execution Agent.

    Wraps the OrderManager with LLM reasoning about execution strategy.
    Receives trade instructions from the Coordinator, decides HOW to execute,
    and monitors fills.
    """

    SYSTEM_PROMPT = """You are a professional execution trader at a systematic hedge fund.

YOUR ROLE:
You receive allocation decisions (what to buy/sell) and decide HOW to execute them optimally.
You do not question WHAT to trade — that's already decided by the Portfolio Manager.
Your job is to minimise execution cost and market impact.

EXECUTION DECISION FRAMEWORK:

1. ASSESS MARKET CONDITIONS
   - VIX level and recent volatility
   - Today's volume vs average (liquidity signal)
   - Bid-ask spread (cost of immediacy)
   - Time to market close
   - Any upcoming earnings or events

2. CHOOSE EXECUTION ALGORITHM
   MARKET ORDER:   Use only for very urgent trades or very small sizes (<$5K)
   TWAP:           Use when market is thin, order is large, patience is acceptable
   VWAP:           Use on normal volume days for medium-large orders
   IS (Almgren-Chriss): Use as default for all significant orders — optimal tradeoff
   LIMIT ORDER:    Use when spread is wide or market is very volatile

3. SET EXECUTION PARAMETERS
   - Horizon (minutes): how long to execute over
     Normal:  60-120 minutes for a full session order
     Urgent:  15-30 minutes
     Patient: 2-4 hours
   - Urgency: LOW / NORMAL / HIGH / URGENT
   - Max slippage tolerance: when to pause if market moves against us

4. MONITOR AND ADAPT
   - If getting worse-than-expected fills: alert Risk Manager, pause if needed
   - If vol spikes mid-execution: widen the schedule
   - If volume dries up: pause until liquidity returns

RESPONSE FORMAT for execution recommendations (JSON):
{
  "algo": "IS" | "TWAP" | "VWAP" | "MARKET" | "LIMIT",
  "horizon_minutes": integer,
  "urgency": "LOW" | "NORMAL" | "HIGH" | "URGENT",
  "n_periods": integer,
  "limit_price": float or null,
  "reasoning": "1-2 sentence explanation",
  "risks": ["execution risk 1", "risk 2"],
  "pause_condition": "when to pause execution"
}

CRITICAL RULES:
- Never send orders larger than the daily loss limit allows
- If VIX > 40: increase execution window by 50%, use limit orders
- If spread > 30bps: use limit orders aggressively
- If volume < 30% of normal: reduce order sizes, extend horizon
- Always execute small orders (<$10K) as single MARKET orders — no need for algo
"""

    def __init__(
        self,
        order_manager: Optional["OrderManager"] = None,
        config=None,
    ):
        from src.execution.order_manager import OrderManager
        from src.agents.base_agent import AgentConfig

        self.om = order_manager or OrderManager()
        cfg = config or AgentConfig(
            name        = "ExecutionAgent",
            model       = "claude-sonnet-4-6",
            temperature = 0.05,   # Very low — execution needs consistency
        )
        super().__init__(cfg)
        logger.info("ExecutionAgent initialised")

    def _get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def _get_tools(self) -> List[Tool]:
        return [
            Tool(
                name = "get_market_context",
                func = self._tool_market_context,
                description = (
                    "Get real-time market context for a ticker: "
                    "spread, volume vs average, volatility, VIX, "
                    "minutes to close, earnings proximity. "
                    "Input: ticker symbol"
                ),
                param_schema={"type":"object","properties":{"ticker":{"type":"string"}},"required":["ticker"]}
            ),
            Tool(
                name = "estimate_execution_cost",
                func = self._tool_estimate_cost,
                description = (
                    "Get Almgren-Chriss optimal execution cost estimate. "
                    "Input: JSON with ticker, shares, side, horizon_minutes. "
                    "Returns IS, TWAP, VWAP cost comparison."
                ),
                param_schema={
                    "type":"object",
                    "properties":{
                        "ticker":{"type":"string"},
                        "shares":{"type":"number"},
                        "side":{"type":"string"},
                        "horizon_minutes":{"type":"number"}
                    },
                    "required":["ticker","shares","side"]
                }
            ),
            Tool(
                name = "submit_execution",
                func = self._tool_submit,
                description = (
                    "Submit the execution to the order manager. "
                    "Input: JSON with ticker, side, shares, algo, "
                    "horizon_minutes, urgency. "
                    "Returns order IDs."
                ),
                param_schema={
                    "type":"object",
                    "properties":{
                        "ticker":{"type":"string"},
                        "side":{"type":"string"},
                        "shares":{"type":"number"},
                        "algo":{"type":"string"},
                        "horizon_minutes":{"type":"number"},
                        "urgency":{"type":"string"},
                        "portfolio_nav":{"type":"number"}
                    },
                    "required":["ticker","side","shares"]
                }
            ),
            Tool(
                name = "check_active_orders",
                func = self._tool_active_orders,
                description = "Check status of all active orders. Input: anything.",
                param_schema={"type":"object","properties":{},"required":[]}
            ),
            Tool(
                name = "get_execution_summary",
                func = self._tool_exec_summary,
                description = "Get today's execution performance summary including TCA. Input: anything.",
                param_schema={"type":"object","properties":{},"required":[]}
            ),
        ]

    # ── Tool implementations ──────────────────────────────────────────────────

    def _tool_market_context(self, ticker: str) -> str:
        try:
            import yfinance as yf
            import math
            t  = yf.Ticker(ticker)
            df = t.history(period="30d", interval="1d")

            if df.empty:
                return json.dumps({"error": f"No data for {ticker}"})

            price = float(df["Close"].iloc[-1])
            log_r = (df["Close"] / df["Close"].shift(1)).apply(lambda x: math.log(x) if x > 0 else 0).dropna()
            vol_21d = float(log_r.tail(21).std() * math.sqrt(252))
            vol_today = float(abs(log_r.iloc[-1]))

            vol_adv = float(df["Volume"].tail(21).mean()) if "Volume" in df.columns else 10e6
            vol_today_val = float(df["Volume"].iloc[-1]) if "Volume" in df.columns else vol_adv
            vol_ratio = vol_today_val / vol_adv if vol_adv > 0 else 1.0

            # VIX
            vix_level = None
            try:
                vix_df = yf.download("^VIX", period="1d", progress=False)
                if not vix_df.empty:
                    vix_level = float(vix_df["Close"].iloc[-1])
            except Exception:
                pass

            # Minutes to close (NYSE closes at 16:00 ET)
            from datetime import datetime
            import pytz
            try:
                et_now = datetime.now(pytz.timezone("America/New_York"))
                close_time = et_now.replace(hour=16, minute=0, second=0)
                mins_to_close = (close_time - et_now).total_seconds() / 60
                mins_to_close = max(0, mins_to_close)
            except Exception:
                mins_to_close = 240

            ctx = ExecutionContext(
                ticker           = ticker,
                current_price    = price,
                spread_bps       = 3.0,       # Estimated for large-cap
                volume_today     = vol_today_val,
                volume_vs_avg    = vol_ratio,
                volatility_today = vol_today,
                volatility_21d   = vol_21d,
                vix              = vix_level,
                market_regime    = ("HIGH_VOL" if (vix_level or 20) > 25 else "NORMAL"),
                minutes_to_close = mins_to_close,
            )

            return ctx.to_prompt_context()

        except Exception as e:
            return f"Market context error: {e}"

    def _tool_estimate_cost(
        self, ticker: str, shares: float, side: str,
        horizon_minutes: float = 60
    ) -> str:
        try:
            price = self.om.broker.get_price(ticker) or 100
            adv, daily_vol = self.om._get_market_data(ticker, price)

            params = MarketImpactParams(
                ticker=ticker, price=price,
                daily_vol=daily_vol, adv=adv
            )
            optimiser = AlmgrenChrissOptimiser(params)
            schedules = optimiser.compare_algos(
                shares, side, horizon_minutes, n_periods=12
            )

            result = {
                "ticker": ticker, "shares": shares, "side": side,
                "price": price,
                "notional_usd": round(shares * price),
                "pct_of_adv": round(shares / adv * 100, 2),
                "estimates": {}
            }
            for algo_name, sched in schedules.items():
                result["estimates"][algo_name] = {
                    "cost_bps": round(sched.expected_cost_bps, 2),
                    "cost_usd": round(sched.expected_cost_usd, 2),
                    "kappa":    round(sched.kappa, 4),
                }
            result["recommendation"] = min(
                result["estimates"],
                key=lambda k: result["estimates"][k]["cost_bps"]
            )
            return json.dumps(result, indent=2)

        except Exception as e:
            return f"Cost estimation error: {e}"

    def _tool_submit(
        self,
        ticker: str,
        side: str,
        shares: float,
        algo: str = "IS",
        horizon_minutes: float = 60,
        urgency: str = "NORMAL",
        portfolio_nav: float = 1_000_000,
        decision_id: str = "",
    ) -> str:
        try:
            price = self.om.broker.get_price(ticker) or 100
            weight = (shares * price) / portfolio_nav

            orders = self.om.execute_decision(
                ticker         = ticker,
                side           = side,
                target_weight  = weight,
                portfolio_nav  = portfolio_nav,
                current_weight = 0.0,
                decision_id    = decision_id,
                agent_name     = self.name,
                urgency        = urgency,
                use_algo       = shares > 500,
            )

            return json.dumps({
                "submitted":   len(orders),
                "order_ids":   [o.order_id for o in orders],
                "total_shares": shares,
                "algo":        algo,
                "status":      [o.status.value for o in orders],
            })

        except Exception as e:
            return f"Submission error: {e}"

    def _tool_active_orders(self, **kwargs) -> str:
        return self.om.print_order_book()

    def _tool_exec_summary(self, **kwargs) -> str:
        return json.dumps(self.om.get_execution_summary(), indent=2)

    # ── Main execution method ─────────────────────────────────────────────────

    def execute(
        self,
        ticker:       str,
        side:         str,
        target_weight: float,
        portfolio_nav: float,
        current_weight: float = 0.0,
        decision_id:   str = "",
        context:       str = "",
    ) -> Dict[str, Any]:
        """
        Execute an allocation decision with LLM-driven execution strategy.

        The LLM:
            1. Reads the market context (volume, vol, spread, VIX)
            2. Estimates execution costs for different algos
            3. Chooses the best algo and parameters
            4. Submits to the order manager

        Args:
            ticker:         Security to trade
            side:           BUY or SELL
            target_weight:  Target portfolio weight
            portfolio_nav:  Current NAV
            current_weight: Existing weight
            decision_id:    Originating decision ID
            context:        Any additional context

        Returns:
            Dict with order IDs and execution summary
        """
        price = self.om.broker.get_price(ticker) or 100
        weight_delta = target_weight - current_weight
        dollar_amount = abs(weight_delta) * portfolio_nav
        shares = max(1, round(dollar_amount / price))

        # Build prompt for the LLM
        user_prompt = f"""Execution request: {side} {shares:,} shares of {ticker}

Order details:
  Target weight: {target_weight:.1%}
  Current weight: {current_weight:.1%}
  Weight delta: {weight_delta:+.1%}
  Dollar amount: ${dollar_amount:,.0f}
  Shares: {shares:,}
  Portfolio NAV: ${portfolio_nav:,.0f}
  Decision ID: {decision_id}

Additional context: {context or 'Standard execution request.'}

Please:
1. Use get_market_context to assess current conditions for {ticker}
2. Use estimate_execution_cost to compare IS, TWAP, VWAP for this order
3. Recommend and submit the optimal execution strategy
4. Report the order IDs and estimated costs

Return your recommendation and the submitted order IDs."""

        response_text, tool_calls = self.think(
            user_message = user_prompt,
            use_tools    = True,
            purpose      = f"execution_{ticker}_{side}",
        )

        # Extract order IDs from tool calls
        order_ids = []
        for tc in tool_calls:
            if tc.get("tool") == "submit_execution":
                try:
                    result = json.loads(tc.get("result", "{}"))
                    order_ids.extend(result.get("order_ids", []))
                except Exception:
                    pass

        return {
            "ticker":       ticker,
            "side":         side,
            "shares":       shares,
            "order_ids":    order_ids,
            "llm_reasoning": response_text[:500],
            "timestamp":    datetime.now().isoformat(),
        }

    # ── MessageBus handler ────────────────────────────────────────────────────

    def handle_message(self, message) -> Optional[Dict[str, Any]]:
        """Handle execution requests from the message bus."""
        subject = message.subject.lower()
        payload = message.payload
        logger.info(f"ExecutionAgent: {message.subject} from {message.sender}")

        if "execute" in subject or "trade" in subject:
            ticker        = payload.get("ticker", "")
            side          = payload.get("side", "BUY")
            target_weight = float(payload.get("target_weight", 0))
            nav           = float(payload.get("portfolio_nav", 1_000_000))
            current_w     = float(payload.get("current_weight", 0))
            decision_id   = payload.get("decision_id", "")

            if not ticker:
                return {"error": "No ticker in execution request"}

            result = self.execute(
                ticker         = ticker,
                side           = side,
                target_weight  = target_weight,
                portfolio_nav  = nav,
                current_weight = current_w,
                decision_id    = decision_id,
            )
            return result

        elif "cancel" in subject:
            order_id = payload.get("order_id", "")
            if order_id:
                cancelled = self.om.broker.cancel_order(order_id)
                return {"cancelled": cancelled, "order_id": order_id}
            else:
                n = self.om.broker.cancel_all()
                return {"cancelled_all": n}

        elif "status" in subject or "orders" in subject:
            return self.om.get_execution_summary()

        else:
            return {"error": f"Unknown execution request: {message.subject}"}


from src.agents.base_agent import BaseAgent, Tool, AgentConfig
from src.execution.almgren_chriss import (
    MarketImpactParams, AlmgrenChrissOptimiser, PreTradeEstimator
)
from src.execution.order_manager import OrderManager


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  Execution Agent — Test")
    print("=" * 60)

    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  No LLM API key — testing OrderManager directly")

        from src.execution.order_manager import OrderManager
        om = OrderManager()
        om.connect()

        print("\n  Direct execution test (AAPL, 2% of $1M)...")
        orders = om.execute_decision(
            ticker        = "AAPL",
            side          = "BUY",
            target_weight = 0.02,
            portfolio_nav = 1_000_000,
            use_algo      = True,
        )
        for o in orders[:3]:
            print(f"    {o}")
        print(f"\n  Summary: {om.get_execution_summary()}")
        om.disconnect()
    else:
        agent = ExecutionAgent()
        agent.om.connect()

        print("\n  Running AI execution for AAPL 3%...")
        result = agent.execute(
            ticker        = "AAPL",
            side          = "BUY",
            target_weight = 0.03,
            portfolio_nav = 1_000_000,
            decision_id   = "TEST_DEC_001",
        )
        print(f"\n  Result:")
        print(f"    Orders submitted: {result['order_ids']}")
        print(f"    Reasoning: {result['llm_reasoning'][:300]}")

        agent.om.disconnect()
