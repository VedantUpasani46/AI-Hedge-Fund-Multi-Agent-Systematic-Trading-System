"""
AI Hedge Fund — Part 10: Fund Operations & Compliance
=======================================================
run_part10.py — Main Entry Point & System Capstone

This is the final part of the AI Hedge Fund system.

Complete 10-part build summary:
    Part 1:  Foundation (data models, LLM client, PM Agent)
    Part 2:  Multi-Agent System (consensus, risk agent, MessageBus)
    Part 3:  RAG & Data Intelligence (SEC EDGAR, ChromaDB)
    Part 4:  Execution Engine (IB API, Almgren-Chriss, OMS, TCA)
    Part 5:  Alternative Assets (cat bonds, ILS, alt data signals)
    Part 6:  Backtesting Engine (walk-forward, attribution, stress)
    Part 7:  Real-Time Risk (live VaR, circuit breakers, factor monitor)
    Part 8:  Investor Dashboard (FastAPI, PDF reports, LLM commentary)
    Part 9:  Cloud Production (Docker, AWS ECS, CI/CD, monitoring)
    Part 10: Fund Operations (NAV engine, compliance, LP reporting)

Total: ~28,000 lines across 10 parts

Usage:
    # Run daily NAV calculation:
    python run_part10.py --daily-nav

    # Run compliance surveillance:
    python run_part10.py --compliance

    # Generate investor statements:
    python run_part10.py --investor-statements

    # Regulatory report data:
    python run_part10.py --regulatory form_pf
    python run_part10.py --regulatory 13f
    python run_part10.py --regulatory best_execution

    # Run Fund Ops Agent (requires LLM key):
    python run_part10.py --agent eod
    python run_part10.py --agent "What is LP001's current return?"

    # Full fund demo (NAV + compliance + statements):
    python run_part10.py --demo

    # Complete system summary:
    python run_part10.py --system-summary
"""

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import cfg, setup_logging

logger = setup_logging()


# ─────────────────────────────────────────────────────────────────────────────
# Setup helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_demo_fund():
    """Build a representative fund with investors for demonstration."""
    from src.nav.nav_engine import NAVEngine, FeeStructure
    from src.compliance.compliance_engine import ComplianceEngine, InvestmentMandate

    engine = NAVEngine(
        fund_name             = "AI Systematic Fund LP",
        inception_date        = date(2024, 1, 1),
        initial_nav_per_share = 1000.0,
    )

    # Add representative investor base
    engine.add_investor("LP001", "Founder Capital LLC",         500_000, FeeStructure.FOUNDERS)
    engine.add_investor("LP002", "Prestige Family Office",      750_000, FeeStructure.STANDARD)
    engine.add_investor("LP003", "University Endowment Fund", 1_000_000, FeeStructure.INSTITUTIONAL)
    engine.add_investor("LP004", "High Net Worth Individual",   250_000, FeeStructure.STANDARD)

    mandate = InvestmentMandate(
        fund_name               = "AI Systematic Fund LP",
        strategy                = "Systematic Long/Short Equity",
        max_single_position_pct = 0.15,
        max_sector_pct          = 0.35,
        max_gross_exposure      = 1.50,
        prohibited_tickers      = [],
    )
    compliance_engine = ComplianceEngine(mandate)

    return engine, compliance_engine


def simulate_nav_history(engine, n_days: int = 60):
    """Simulate NAV history for demonstration."""
    np.random.seed(42)
    nav = 2_500_000.0  # Total committed capital
    start = date.today() - timedelta(days=n_days)

    print(f"  Simulating {n_days} days of NAV history...")
    for i in range(n_days):
        dt   = start + timedelta(days=i)
        # Skip weekends
        if dt.weekday() >= 5:
            continue
        daily_ret = np.random.normal(0.0005, 0.012)
        nav      *= (1 + daily_ret)
        engine.calculate_daily_nav(
            portfolio_nav = nav,
            cash          = nav * 0.15,
            positions     = {
                "AAPL": nav * 0.12,
                "MSFT": nav * 0.11,
                "NVDA": nav * 0.08,
                "JPM":  nav * 0.10,
                "XOM":  nav * 0.07,
            },
            nav_date = dt,
        )

    return nav


def validate_environment() -> bool:
    print("\n" + "═" * 65)
    print("  Part 10: Fund Operations & Compliance — Environment Check")
    print("═" * 65)

    pkgs = [
        ("numpy",    "pip install numpy"),
        ("pandas",   "pip install pandas"),
        ("yfinance", "pip install yfinance"),
    ]

    all_ok = True
    print("\n  Required:")
    for pkg, install in pkgs:
        try:
            __import__(pkg)
            print(f"    ✓ {pkg}")
        except ImportError:
            print(f"    ✗ {pkg} — {install}")
            all_ok = False

    has_llm = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"))
    print(f"\n  LLM key: {'✓' if has_llm else '✗ (needed for agent mode)'}")
    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Command handlers
# ─────────────────────────────────────────────────────────────────────────────

def run_daily_nav():
    print(f"\n{'═'*65}")
    print("  Daily NAV Calculation")
    print(f"{'═'*65}")

    engine, _ = make_demo_fund()
    total_nav = simulate_nav_history(engine, n_days=5)

    nav = engine.calculate_daily_nav(
        portfolio_nav = total_nav,
        cash          = total_nav * 0.15,
        positions     = {
            "AAPL": total_nav * 0.12,
            "MSFT": total_nav * 0.11,
            "NVDA": total_nav * 0.08,
            "JPM":  total_nav * 0.10,
            "XOM":  total_nav * 0.07,
        },
    )

    print(f"\n  Official NAV [{nav.nav_date}]")
    print(f"  {'─'*50}")
    print(f"  Gross Asset Value     : ${nav.gross_asset_value:>15,.2f}")
    print(f"  Accrued Fees          : ${nav.total_liabilities:>15,.2f}")
    print(f"  Net Asset Value       : ${nav.net_asset_value:>15,.2f}")
    print(f"  NAV per Share         : ${nav.nav_per_share:>15.6f}")
    print(f"  Shares Outstanding    : {nav.shares_outstanding:>16.4f}")
    print(f"  {'─'*50}")
    print(f"  Daily Return          : {nav.daily_return_pct:>+14.4%}")
    print(f"  MTD Return            : {nav.mtd_return_pct:>+14.4%}")
    print(f"  YTD Return            : {nav.ytd_return_pct:>+14.4%}")
    print(f"  {'─'*50}")
    print(f"  Mgmt Fee Accrual      : ${nav.mgmt_fee_accrual:>15,.2f}")
    print(f"  Perf Fee Accrual      : ${nav.perf_fee_accrual:>15,.2f}")
    print(f"  Other Expenses        : ${nav.other_expenses:>15,.2f}")


def run_compliance():
    print(f"\n{'═'*65}")
    print("  Compliance Surveillance")
    print(f"{'═'*65}")

    from src.compliance.compliance_engine import ComplianceEngine, InvestmentMandate

    mandate = InvestmentMandate(
        fund_name               = "AI Systematic Fund LP",
        max_single_position_pct = 0.15,
        max_sector_pct          = 0.35,
        max_gross_exposure      = 1.50,
    )
    engine = ComplianceEngine(mandate)

    positions = {
        "AAPL": {"weight": 0.12, "sector": "Technology",  "shares": 500},
        "MSFT": {"weight": 0.11, "sector": "Technology",  "shares": 300},
        "NVDA": {"weight": 0.08, "sector": "Technology",  "shares": 150},
        "JPM":  {"weight": 0.10, "sector": "Financials",  "shares": 800},
        "XOM":  {"weight": 0.07, "sector": "Energy",      "shares": 600},
    }

    # Test a pre-trade check
    print("\n  Pre-Trade Checks:")
    test_trades = [
        ("AAPL",  0.14, "approaching limit"),
        ("MSFT",  0.18, "breach"),
        ("NVDA",  0.09, "OK"),
    ]
    for ticker, weight, expected in test_trades:
        ok, alerts = engine.pre_trade_check(ticker, weight, {k: v["weight"] for k, v in positions.items()}, 1_000_000)
        status = "✓ OK" if ok else "✗ BLOCKED"
        print(f"  {status:<12} {ticker} @ {weight:.0%}  ({expected})")
        for a in alerts:
            print(f"             [{a.severity.value}] {a.message[:70]}")

    # Daily surveillance
    print("\n  Daily Surveillance:")
    alerts = engine.daily_surveillance(positions, [], 2_500_000)

    summary = engine.compliance_summary()
    status_icon = {"CLEAN": "🟢", "WARNING": "🟡", "BREACH": "🔴"}.get(summary["status"], "⚪")
    print(f"\n  {status_icon} Status: {summary['status']}")
    print(f"  Open alerts:  {summary['open_alerts']}")
    print(f"  Breaches:     {summary['breaches']}")
    print(f"  Warnings:     {summary['warnings']}")

    if alerts:
        print(f"\n  Alert details:")
        for a in alerts[:5]:
            icon = "❌" if a.severity.value == "BREACH" else "⚠️"
            print(f"  {icon} [{a.severity.value}] {a.message[:70]}")


def run_investor_statements():
    print(f"\n{'═'*65}")
    print("  Investor Capital Account Statements")
    print(f"{'═'*65}")

    engine, _ = make_demo_fund()
    simulate_nav_history(engine, n_days=30)

    for inv_id, investor in engine.investors.items():
        stmt = engine.get_investor_statement(inv_id)
        sign = "+" if stmt["total_return_pct"] >= 0 else ""
        print(f"\n  ── {stmt['name']} ({inv_id}) ──────────────────────")
        print(f"  Called Capital    : ${stmt['called_capital']:>12,.2f}")
        print(f"  Current Value     : ${stmt['current_value']:>12,.2f}")
        print(f"  Unrealised Gain   : ${stmt['unrealised_gain']:>+12,.2f}")
        print(f"  Total Return      : {sign}{stmt['total_return_pct']:>11.2f}%")
        print(f"  MOIC              : {stmt['moic']:>12.3f}×")
        print(f"  Shares            : {stmt['shares_held']:>15.6f}")
        print(f"  NAV/Share         : ${stmt['nav_per_share']:>12.6f}")
        print(f"  Fee Structure     : {stmt['fee_structure']:>15}")
        print(f"  Accrued Mgmt Fee  : ${stmt['accrued_mgmt_fee']:>12,.2f}")
        print(f"  Accrued Perf Fee  : ${stmt['accrued_perf_fee']:>12,.2f}")


def run_regulatory(report_type: str):
    print(f"\n{'═'*65}")
    print(f"  Regulatory Data — {report_type.upper()}")
    print(f"{'═'*65}")

    from src.compliance.compliance_engine import ComplianceEngine, InvestmentMandate

    mandate = InvestmentMandate("AI Systematic Fund LP")
    engine  = ComplianceEngine(mandate)

    positions = {
        "AAPL": 300_000, "MSFT": 275_000, "NVDA": 200_000,
        "JPM":  250_000, "XOM":  175_000,
    }

    rt = report_type.lower().replace("-", "_").replace(" ", "_")

    if rt == "form_pf":
        data = engine.generate_form_pf_summary(
            nav=2_500_000, gross_exposure=0.80, net_exposure=0.78
        )
    elif rt == "13f":
        data = engine.generate_13f_holdings(positions, 2_500_000)
    elif rt in ("best_execution", "best_exec", "tca"):
        sample_trades = [
            {"ticker": "AAPL", "side": "BUY", "quantity": 100, "is_bps": 3.2,
             "commission": 0.50, "status": "FILLED", "created_at": datetime.now().isoformat()},
            {"ticker": "MSFT", "side": "BUY", "quantity": 50,  "is_bps": 5.8,
             "commission": 0.25, "status": "FILLED", "created_at": datetime.now().isoformat()},
            {"ticker": "NVDA", "side": "SELL","quantity": 30,  "is_bps": 8.1,
             "commission": 0.15, "status": "FILLED", "created_at": datetime.now().isoformat()},
        ]
        data = engine.generate_best_execution_report(
            trades=sample_trades,
            period_start=date.today().replace(day=1),
            period_end=date.today(),
        )
    else:
        print(f"  Unknown report type: {report_type}")
        print("  Available: form_pf, 13f, best_execution")
        return

    print(f"\n{json.dumps(data, indent=2)[:1200]}")
    if len(json.dumps(data)) > 1200:
        print("  ... (truncated)")


def run_agent_query(query: str):
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  No LLM key — running demo mode")
        if "eod" in query.lower():
            run_daily_nav()
            run_compliance()
        else:
            run_investor_statements()
        return

    from src.agents.fund_ops_agent import FundOpsAgent
    from src.nav.nav_engine import NAVEngine, FeeStructure
    from src.data.data_models import Portfolio

    nav_engine = NAVEngine("AI Systematic Fund LP", date(2024, 1, 1))
    nav_engine.add_investor("LP001", "Alpha Capital LP", 1_000_000)
    nav_engine.add_investor("LP002", "Beta Family Office", 500_000, FeeStructure.INSTITUTIONAL)
    simulate_nav_history(nav_engine, n_days=10)

    agent = FundOpsAgent(nav_engine=nav_engine)

    print(f"\n  Query: {query}\n")

    if query.lower() == "eod":
        result = agent.run_eod_operations()
        print(json.dumps(result, indent=2)[:1000])
    elif query.lower() == "monthly":
        result = agent.generate_monthly_summary()
        print(json.dumps(result, indent=2)[:800])
    else:
        result = agent.answer_investor_query("LP001", query)
        print(f"\n  Response:\n{result[:800]}")


def run_demo():
    print(f"\n{'╔'+'═'*58+'╗'}")
    print(f"{'║'+'  PART 10: FUND OPERATIONS & COMPLIANCE'.center(58)+'║'}")
    print(f"{'╚'+'═'*58+'╝'}")

    print("\n[1/4] Daily NAV Calculation...")
    run_daily_nav()

    print("\n[2/4] Compliance Surveillance...")
    run_compliance()

    print("\n[3/4] Investor Capital Account Statements...")
    run_investor_statements()

    print("\n[4/4] Regulatory Data (Form PF summary)...")
    run_regulatory("form_pf")


def print_system_summary():
    print(f"\n{'╔'+'═'*63+'╗'}")
    print(f"{'║'+'  AI HEDGE FUND — COMPLETE SYSTEM SUMMARY'.center(63)+'║'}")
    print(f"{'╚'+'═'*63+'╝'}")

    parts = [
        (1,  "Foundation",          "Portfolio, data models, LLM client, PM Agent",             "~3,500"),
        (2,  "Multi-Agent System",  "Consensus protocol, Risk Agent, MessageBus",                "~8,400"),
        (3,  "RAG & Knowledge Base","SEC EDGAR ingestion, ChromaDB, vector search",              "~3,900"),
        (4,  "Execution Engine",    "IB TWS API, Almgren-Chriss OMS, TCA",                       "~3,400"),
        (5,  "Alternative Assets",  "Cat bonds, ILS portfolio, alt data signals",                "~3,200"),
        (6,  "Backtesting Engine",  "Event-driven, walk-forward, attribution, stress",           "~4,000"),
        (7,  "Real-Time Risk",      "Live VaR, circuit breakers, factor monitor",                "~2,400"),
        (8,  "Investor Dashboard",  "FastAPI REST+WS, PDF reports, LLM commentary",             "~2,300"),
        (9,  "Cloud Production",    "Docker, AWS ECS, CI/CD, structured logging",                "~2,900"),
        (10, "Fund Operations",     "NAV engine, compliance, LP reporting",                      "~3,200"),
    ]

    total_lines = 0
    print(f"\n  {'Part':<6} {'Module':<25} {'Description':<40} {'Lines':>7}")
    print(f"  {'─'*80}")
    for num, module, desc, lines in parts:
        print(f"  {num:<6} {module:<25} {desc:<40} {lines:>7}")
        total_lines += int(lines.replace("~", "").replace(",", ""))
    print(f"  {'─'*80}")
    print(f"  {'TOTAL':<32} {'':<40} ~{total_lines:,d}")

    print(f"\n  Key design decisions:")
    items = [
        ("Architecture",     "Event-driven, zero look-ahead bias by construction"),
        ("Consensus",        "3-agent unanimous BUY = full Kelly; lone BUY = 5% max"),
        ("RAG sources",      "SEC EDGAR + FOMC minutes + Yahoo News (all free)"),
        ("Execution",        "Almgren-Chriss sinh closed-form, wired to IB paper trading"),
        ("Backtesting",      "Walk-forward validation; Deflated Sharpe Ratio reported"),
        ("Risk",             "7 circuit breakers; background thread every 30s"),
        ("ILS pricing",      "Poisson(λ) × GPD severity reusing EVT module"),
        ("Production",       "Multi-stage Docker, ECS rolling deploy, CloudWatch alarms"),
        ("Compliance",       "Pre-trade + post-trade; 13D/13F/Form PF templates"),
        ("NAV",              "High-water mark; daily fee accrual; per-LP accounts"),
    ]
    print(f"\n  {'Category':<18} Detail")
    print(f"  {'─'*65}")
    for category, detail in items:
        print(f"  {category:<18} {detail}")

    print(f"\n  To run any part:")
    print(f"    pip install numpy pandas scipy yfinance reportlab")
    print(f"    python run_part1.py --demo")
    print(f"    python run_part6.py --backtest momentum AAPL MSFT NVDA GOOGL JPM")
    print(f"    python run_part8.py --server        # API at localhost:8000")
    print(f"    python run_part10.py --demo")

    print(f"\n  To run with full LLM intelligence:")
    print(f"    export ANTHROPIC_API_KEY=sk-ant-...")
    print(f"    python run_part2.py --scan --top 5")
    print(f"    python run_part10.py --agent eod")

    print(f"\n  For cloud deployment:")
    print(f"    docker-compose up -d")
    print(f"    python run_part9.py --deploy  # Requires AWS credentials")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AI Hedge Fund — Part 10: Fund Operations & Compliance"
    )
    parser.add_argument("--daily-nav",             action="store_true")
    parser.add_argument("--compliance",            action="store_true")
    parser.add_argument("--investor-statements",   action="store_true")
    parser.add_argument("--regulatory",            type=str,
                        help="form_pf | 13f | best_execution")
    parser.add_argument("--agent",                 type=str,
                        help="'eod', 'monthly', or natural language query")
    parser.add_argument("--demo",                  action="store_true")
    parser.add_argument("--system-summary",        action="store_true")
    parser.add_argument("--validate",              action="store_true")
    args = parser.parse_args()

    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + "  AI HEDGE FUND — PART 10: FUND OPERATIONS".center(58) + "║")
    print("║" + f"  {datetime.now():%Y-%m-%d %H:%M:%S}".center(58) + "║")
    print("╚" + "═"*58 + "╝")

    validate_environment()

    if args.validate:                           return
    elif getattr(args, "daily_nav", False):     run_daily_nav()
    elif args.compliance:                       run_compliance()
    elif getattr(args, "investor_statements", False): run_investor_statements()
    elif args.regulatory:                       run_regulatory(args.regulatory)
    elif args.agent:                            run_agent_query(args.agent)
    elif getattr(args, "system_summary", False):print_system_summary()
    elif args.demo:                             run_demo()
    else:
        print("\n  No command — running full demo + system summary")
        run_demo()
        print_system_summary()

    print("\n" + "═"*65)
    print("  ✅  AI Hedge Fund — All 10 Parts Complete.")
    print("═"*65)


if __name__ == "__main__":
    main()
