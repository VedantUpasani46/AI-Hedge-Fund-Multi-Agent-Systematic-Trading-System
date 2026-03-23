"""
AI Hedge Fund — Part 10: Fund Operations & Compliance
=======================================================
fund_ops_agent.py — Fund Operations Intelligence Agent

The Fund Ops Agent is the operational backbone of the fund.
It runs the daily end-of-day operations workflow automatically
and answers questions about fund administration.

Daily workflow (runs at 4:30pm ET after market close):
    1. Fetch closing prices and compute final NAV
    2. Accrue fees and update investor accounts
    3. Run compliance surveillance (post-trade checks)
    4. Generate end-of-day report (NAV, P&L, compliance status)
    5. Trigger PDF report generation if month-end
    6. Send daily summary to operations team

Monthly workflow (runs on last business day of month):
    1. Calculate final monthly NAV
    2. Crystallise performance fees above high-water mark
    3. Generate monthly investor letters (one per LP)
    4. Prepare regulatory filing data (Form PF summary)
    5. Best execution report for the month

Investor service (on demand):
    - Answer investor queries about their account
    - Generate capital account statements
    - Process subscription/redemption requests
    - Respond to due diligence questionnaires

Tools:
    get_nav_summary          — Current NAV, fees, investor summary
    get_investor_statement   — Individual LP account statement
    run_daily_nav            — Calculate and persist today's NAV
    run_compliance_check     — Post-trade compliance surveillance
    get_compliance_alerts    — Open compliance alerts
    generate_investor_letter — Monthly LP letter with commentary
    process_subscription     — Add new investor capital
    process_redemption       — Handle investor redemption
    get_regulatory_data      — Form PF / 13F / best execution data
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hedge_fund.fund_ops")


class FundOpsAgent(BaseAgent):
    """
    Fund Operations Intelligence Agent.

    Orchestrates daily NAV, compliance surveillance, and investor
    reporting. Handles investor queries and operational decisions.
    """

    SYSTEM_PROMPT = """You are the Chief Operating Officer of a systematic hedge fund.

YOUR RESPONSIBILITIES:

1. FUND ACCOUNTING (daily)
   - Calculate NAV using closing prices
   - Accrue management fees (daily 1/252 of annual rate)
   - Accrue performance fees above high-water mark
   - Update all investor capital accounts
   - Reconcile with prime broker statement

2. COMPLIANCE (daily)
   - Run post-trade surveillance (concentration, leverage, wash sales)
   - Pre-clear personal account trades for supervised persons
   - Monitor and resolve compliance alerts
   - Maintain audit trail for all decisions

3. INVESTOR RELATIONS (on-demand)
   - Answer investor queries about their account
   - Generate capital account statements
   - Process subscriptions and redemptions
   - Produce monthly investor letters
   - Handle due diligence questionnaires

4. REGULATORY REPORTING (quarterly/annual)
   - Generate Form PF data summary
   - Prepare Schedule 13F (if >$100M equity)
   - Best execution quarterly report
   - Note: These are templates requiring legal review before filing

STANDARDS:
- Every NAV is supported by market prices (not estimates)
- Every fee accrual is calculated transparently
- Every compliance alert is documented and tracked
- Investor communications are professional and accurate

When reporting NAV:
  "As of [date], the fund's NAV is $X per share, representing a Y% return
  since inception. Management fees of $Z were accrued today."

When handling compliance alerts:
  State the alert clearly, explain the regulatory implication,
  and recommend specific remediation steps.

Respond in JSON for structured operations:
{
  "action_taken": str,
  "nav_per_share": float,
  "daily_return_pct": float,
  "compliance_status": "CLEAN|WARNING|BREACH",
  "alerts": [str],
  "next_steps": [str]
}"""

    def __init__(
        self,
        nav_engine=None,
        compliance_engine=None,
        portfolio=None,
        config=None,
    ):
        from src.agents.base_agent import AgentConfig
        from src.nav.nav_engine import NAVEngine, InvestmentMandate
        from src.compliance.compliance_engine import ComplianceEngine, InvestmentMandate as Mandate

        cfg = config or AgentConfig(
            name        = "FundOpsAgent",
            model       = "claude-sonnet-4-6",
            temperature = 0.05,
            max_tokens  = 4096,
        )

        self.nav_engine    = nav_engine
        self.compliance    = compliance_engine
        self.portfolio     = portfolio

        # Lazy init if not provided
        if not self.nav_engine:
            self.nav_engine = NAVEngine(
                fund_name     = os.getenv("FUND_NAME", "AI Systematic Fund LP"),
                inception_date= date(2024, 1, 1),
            )
            self.nav_engine.load_history()

        if not self.compliance:
            from src.compliance.compliance_engine import InvestmentMandate, ComplianceEngine
            mandate        = Mandate(fund_name=os.getenv("FUND_NAME", "AI Systematic Fund LP"))
            self.compliance= ComplianceEngine(mandate)

        super().__init__(cfg)
        logger.info("FundOpsAgent initialised")

    def _get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def _get_tools(self) -> List[Tool]:
        return [
            Tool(
                name="get_nav_summary",
                func=self._tool_nav_summary,
                description=(
                    "Get current fund NAV summary: total NAV, NAV per share, "
                    "shares outstanding, accrued fees, investor count, MTD/YTD returns. "
                    "Input: anything."
                ),
                param_schema={"type":"object","properties":{},"required":[]}
            ),
            Tool(
                name="get_investor_statement",
                func=self._tool_investor_statement,
                description=(
                    "Get account statement for a specific investor. "
                    "Input: investor_id (e.g. 'LP001')"
                ),
                param_schema={"type":"object","properties":{"investor_id":{"type":"string"}},"required":["investor_id"]}
            ),
            Tool(
                name="run_daily_nav",
                func=self._tool_run_nav,
                description=(
                    "Calculate and persist today's official NAV. "
                    "Uses current portfolio prices. "
                    "Input: anything."
                ),
                param_schema={"type":"object","properties":{},"required":[]}
            ),
            Tool(
                name="run_compliance_surveillance",
                func=self._tool_compliance,
                description=(
                    "Run post-trade compliance surveillance on current positions. "
                    "Checks concentration, leverage, sector, wash sales, best execution. "
                    "Input: anything."
                ),
                param_schema={"type":"object","properties":{},"required":[]}
            ),
            Tool(
                name="get_compliance_alerts",
                func=self._tool_compliance_alerts,
                description=(
                    "Get open compliance alerts that need resolution. "
                    "Input: severity ('WARNING', 'BREACH', or 'all')"
                ),
                param_schema={
                    "type":"object",
                    "properties":{"severity":{"type":"string","default":"WARNING"}},
                    "required":[]
                }
            ),
            Tool(
                name="get_regulatory_data",
                func=self._tool_regulatory,
                description=(
                    "Get regulatory filing data: Form PF summary, 13F holdings, "
                    "best execution report. "
                    "Input: report_type ('form_pf', '13f', 'best_execution')"
                ),
                param_schema={
                    "type":"object",
                    "properties":{"report_type":{"type":"string"}},
                    "required":["report_type"]
                }
            ),
            Tool(
                name="get_all_investor_summaries",
                func=self._tool_all_investors,
                description=(
                    "Get summary of all investor accounts: names, "
                    "current values, returns, fees. "
                    "Input: anything."
                ),
                param_schema={"type":"object","properties":{},"required":[]}
            ),
        ]

    # ── Tool implementations ──────────────────────────────────────────────────

    def _tool_nav_summary(self, **kwargs) -> str:
        return json.dumps(self.nav_engine.get_fund_summary(), indent=2)

    def _tool_investor_statement(self, investor_id: str) -> str:
        stmt = self.nav_engine.get_investor_statement(investor_id)
        return json.dumps(stmt, indent=2)

    def _tool_run_nav(self, **kwargs) -> str:
        nav_val = self.portfolio.net_asset_value if self.portfolio else 1_000_000
        cash    = getattr(self.portfolio, "cash", nav_val * 0.15) if self.portfolio else nav_val * 0.15
        positions = {}
        if self.portfolio:
            for t, p in self.portfolio.positions.items():
                shares  = getattr(p, "shares", 0)
                price   = getattr(p, "current_price", 0)
                positions[t] = shares * price

        nav = self.nav_engine.calculate_daily_nav(
            portfolio_nav = nav_val,
            cash          = cash,
            positions     = positions,
        )
        return json.dumps(nav.to_dict(), indent=2)

    def _tool_compliance(self, **kwargs) -> str:
        positions = {}
        if self.portfolio:
            nav = self.portfolio.net_asset_value
            for t, p in self.portfolio.positions.items():
                shares  = getattr(p, "shares", 0)
                price   = getattr(p, "current_price", 0)
                sector  = getattr(p, "sector", "Unknown")
                mv      = shares * price
                positions[t] = {
                    "weight": mv / nav if nav > 0 else 0,
                    "sector": sector,
                    "shares": shares,
                }

        nav_val = self.portfolio.net_asset_value if self.portfolio else 1_000_000
        alerts  = self.compliance.daily_surveillance(positions, [], nav_val)
        return json.dumps({
            "n_alerts":  len(alerts),
            "status":    "BREACH" if any(a.severity.value in ("BREACH","CRITICAL") for a in alerts) else
                         "WARNING" if alerts else "CLEAN",
            "alerts":    [a.to_dict() for a in alerts],
        }, indent=2)

    def _tool_compliance_alerts(self, severity: str = "WARNING", **kwargs) -> str:
        from src.compliance.compliance_engine import AlertSeverity
        sev_map = {
            "info":    AlertSeverity.INFO,
            "warning": AlertSeverity.WARNING,
            "breach":  AlertSeverity.BREACH,
            "all":     AlertSeverity.INFO,
        }
        min_sev = sev_map.get(severity.lower(), AlertSeverity.WARNING)
        alerts  = self.compliance.get_open_alerts(min_sev)
        return json.dumps({"n_open": len(alerts), "alerts": alerts[:20]}, indent=2)

    def _tool_regulatory(self, report_type: str) -> str:
        nav_val = self.portfolio.net_asset_value if self.portfolio else 1_000_000
        positions = {}
        if self.portfolio:
            for t, p in self.portfolio.positions.items():
                positions[t] = getattr(p, "shares", 0) * getattr(p, "current_price", 0)

        rt = report_type.lower()
        if rt == "form_pf":
            return json.dumps(self.compliance.generate_form_pf_summary(
                nav=nav_val, gross_exposure=0.80, net_exposure=0.80
            ), indent=2)
        elif rt == "13f":
            return json.dumps(self.compliance.generate_13f_holdings(
                positions=positions, nav=nav_val
            ), indent=2)
        elif rt in ("best_execution", "best_exec", "tca"):
            return json.dumps(self.compliance.generate_best_execution_report(
                trades=[], period_start=date.today(), period_end=date.today()
            ), indent=2)
        return json.dumps({"error": f"Unknown report type: {report_type}"})

    def _tool_all_investors(self, **kwargs) -> str:
        summaries = []
        for investor_id in self.nav_engine.investors:
            s = self.nav_engine.get_investor_statement(investor_id)
            summaries.append(s)
        return json.dumps({
            "n_investors": len(summaries),
            "investors":   summaries,
        }, indent=2)

    # ── Main operations methods ───────────────────────────────────────────────

    def run_eod_operations(self, date_str: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete end-of-day operations workflow.

        1. Calculate official NAV
        2. Update investor accounts
        3. Run compliance surveillance
        4. Generate ops summary
        """
        prompt = f"""Run the complete end-of-day operations workflow for {date_str or 'today'}.

Use your tools in this order:
1. run_daily_nav — calculate today's official NAV
2. run_compliance_surveillance — post-trade compliance check
3. get_compliance_alerts — review any open alerts
4. get_nav_summary — get the final fund summary

Then report:
- Final NAV and NAV per share
- Daily return
- Compliance status and any alerts requiring attention
- Any actions required tomorrow

Return results in JSON format."""

        response, _ = self.think(
            user_message = prompt,
            use_tools    = True,
            purpose      = "eod_operations",
        )
        return self._parse_json_response(response) or {"response": response}

    def answer_investor_query(self, investor_id: str, question: str) -> str:
        """Handle an investor query about their account."""
        prompt = f"""Investor {investor_id} has asked: "{question}"

Use get_investor_statement to get their account details, then answer clearly.
Be specific with numbers. Mention their current return and any fee accruals.
Keep the response professional and concise (2-3 paragraphs max)."""

        response, _ = self.think(
            user_message = prompt,
            use_tools    = True,
            purpose      = "investor_query",
        )
        return response

    def generate_monthly_summary(self) -> Dict[str, Any]:
        """Generate monthly fund operations summary."""
        prompt = """Generate the monthly fund operations summary.

Use get_nav_summary, get_all_investor_summaries, and get_compliance_alerts.
Include:
- Fund NAV and MTD return
- Fee summary (management + performance)
- Investor summary (total capital, n investors)
- Compliance summary (any open alerts)
- Key operational items for next month

Format as JSON with keys: nav, mtd_return, fees_accrued,
investor_count, compliance_status, notes."""

        response, _ = self.think(
            user_message = prompt,
            use_tools    = True,
            purpose      = "monthly_ops_summary",
        )
        return self._parse_json_response(response) or {"response": response}

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
        logger.info(f"FundOpsAgent: {message.subject}")

        if "eod" in subject or "daily_nav" in subject:
            return self.run_eod_operations()

        elif "investor_query" in subject:
            investor_id = payload.get("investor_id", "")
            question    = payload.get("question", "")
            return {"response": self.answer_investor_query(investor_id, question)}

        elif "compliance" in subject:
            result = json.loads(self._tool_compliance())
            return result

        elif "nav" in subject:
            return json.loads(self._tool_run_nav())

        elif "monthly" in subject:
            return self.generate_monthly_summary()

        else:
            response, _ = self.think(
                user_message = f"Fund ops question: {message.subject}\n{json.dumps(payload)}",
                use_tools    = True,
                purpose      = "ops_query",
            )
            return {"response": response[:500]}


from src.agents.base_agent import BaseAgent, Tool, AgentConfig


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 65)
    print("  Fund Ops Agent — Test")
    print("=" * 65)

    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  No LLM key — testing NAV and compliance components directly\n")

        from src.nav.nav_engine import NAVEngine, FeeStructure
        from src.compliance.compliance_engine import ComplianceEngine, InvestmentMandate

        # NAV Engine test
        nav_engine = NAVEngine("Test Fund LP", date(2024, 1, 1))
        nav_engine.add_investor("LP001", "Test Investor A", 500_000, FeeStructure.STANDARD)
        nav_engine.add_investor("LP002", "Test Investor B", 250_000, FeeStructure.FOUNDERS)

        nav = nav_engine.calculate_daily_nav(
            portfolio_nav = 780_000,
            cash          = 130_000,
            positions     = {"AAPL": 400_000, "MSFT": 250_000},
        )
        print(f"NAV: ${nav.net_asset_value:,.2f} | ${nav.nav_per_share:.4f}/share")

        for inv_id in nav_engine.investors:
            stmt = nav_engine.get_investor_statement(inv_id)
            print(f"  {stmt['name']}: ${stmt['current_value']:,.2f} | return: {stmt['total_return_pct']:+.2f}%")

        # Compliance test
        mandate    = InvestmentMandate("Test Fund LP")
        compliance = ComplianceEngine(mandate)
        ok, alerts = compliance.pre_trade_check("AAPL", 0.12, {"AAPL": 0.10}, 1_000_000)
        print(f"\nPre-trade check: {'OK' if ok else 'BLOCKED'} ({len(alerts)} alerts)")

        print("\n✅ Fund ops components working. Add API key for full agent test.")
    else:
        from src.nav.nav_engine import NAVEngine, FeeStructure
        nav_engine = NAVEngine("AI Systematic Fund LP", date(2024, 1, 1))
        nav_engine.add_investor("LP001", "Alpha Capital LP", 1_000_000)
        nav_engine.add_investor("LP002", "Beta Family Office", 500_000, FeeStructure.INSTITUTIONAL)

        agent = FundOpsAgent(nav_engine=nav_engine)

        print("\nRunning EOD operations...")
        result = agent.run_eod_operations()
        print(f"\nEOD Result:")
        print(json.dumps(result, indent=2)[:1000])
