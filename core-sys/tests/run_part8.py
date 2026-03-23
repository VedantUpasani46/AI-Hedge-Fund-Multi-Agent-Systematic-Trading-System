"""
AI Hedge Fund — Part 8: Investor Dashboard
============================================
run_part8.py — Main Entry Point

Usage:
    # Start the API server (no LLM needed for data endpoints):
    python run_part8.py --server

    # Start server on a custom port:
    python run_part8.py --server --port 8080

    # Generate a PDF monthly letter (demo data):
    python run_part8.py --report monthly

    # Generate daily risk report:
    python run_part8.py --report daily

    # Test all API endpoints locally (runs server + curl tests):
    python run_part8.py --test-api

    # Answer an investor question (requires LLM key):
    python run_part8.py --query "How is the portfolio performing?"

    # Generate commentary (requires LLM key):
    python run_part8.py --commentary

    # End-of-day summary (requires LLM key):
    python run_part8.py --eod

    # Full demo (generate PDFs + show API endpoints):
    python run_part8.py --demo
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import cfg, setup_logging

logger = setup_logging()


def make_test_portfolio():
    from src.data.data_models import Portfolio, Position, Direction

    portfolio = Portfolio("FUND_001", cash=650_000, initial_capital=1_000_000)
    for ticker, shares, avg, curr, sector in [
        ("AAPL", 500, 180.0, 195.0, "Technology"),
        ("MSFT", 300, 380.0, 415.0, "Technology"),
        ("NVDA", 120, 450.0, 485.0, "Technology"),
        ("JPM",  800, 185.0, 198.0, "Financials"),
        ("XOM",  600, 105.0, 112.0, "Energy"),
    ]:
        portfolio.positions[ticker] = Position(
            ticker, Direction.LONG, shares, avg, curr, sector=sector
        )
    return portfolio


def validate_environment() -> bool:
    print("\n" + "═" * 65)
    print("  Part 8: Investor Dashboard — Environment Check")
    print("═" * 65)

    has_llm = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"))

    core = [
        ("numpy",      "pip install numpy"),
        ("pandas",     "pip install pandas"),
        ("reportlab",  "pip install reportlab"),
        ("yfinance",   "pip install yfinance"),
    ]
    server = [
        ("fastapi",    "pip install fastapi"),
        ("uvicorn",    "pip install uvicorn"),
        ("pydantic",   "pip install pydantic"),
    ]
    optional = [
        ("matplotlib", "pip install matplotlib  # for charts in PDFs"),
    ]

    all_ok = True
    print("\n  Core (PDF generation):")
    for pkg, install in core:
        try:
            __import__(pkg)
            print(f"    ✓ {pkg}")
        except ImportError:
            print(f"    ✗ {pkg} — {install}")
            all_ok = False

    print("\n  Server (API):")
    for pkg, install in server:
        try:
            __import__(pkg)
            print(f"    ✓ {pkg}")
        except ImportError:
            print(f"    ○ {pkg} — {install}")

    print("\n  Optional:")
    for pkg, install in optional:
        try:
            __import__(pkg)
            print(f"    ✓ {pkg}")
        except ImportError:
            print(f"    ○ {pkg} — {install}")

    print(f"\n  LLM key: {'✓' if has_llm else '✗ (needed for commentary / queries)'}")
    return all_ok


def start_server(port: int = 8000):
    """Start the FastAPI server with test portfolio."""
    print(f"\n{'═'*65}")
    print(f"  Starting API Server on port {port}")
    print(f"{'═'*65}")

    try:
        from src.api.api_server import create_app, get_state
        import uvicorn
    except ImportError as e:
        print(f"\n  ✗ Missing dependency: {e}")
        print("  Run: pip install fastapi uvicorn")
        return

    portfolio = make_test_portfolio()
    state     = get_state()
    state.set_portfolio(portfolio)

    app = create_app(portfolio=portfolio)

    print(f"\n  Dashboard API running:")
    print(f"    http://localhost:{port}/")
    print(f"    http://localhost:{port}/docs          ← Interactive API docs")
    print(f"    http://localhost:{port}/portfolio     ← Current portfolio")
    print(f"    http://localhost:{port}/risk          ← Live risk snapshot")
    print(f"    http://localhost:{port}/performance   ← Performance stats")
    print(f"    http://localhost:{port}/trades        ← Recent trades")
    print(f"    ws://localhost:{port}/ws/risk         ← WebSocket stream")
    print(f"\n  Default API key: dev-key-1")
    print(f"  Header: X-API-Key: dev-key-1\n")

    try:
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
    except KeyboardInterrupt:
        print("\n  Server stopped.")


def test_api(port: int = 8000):
    """Start server and test all endpoints."""
    import subprocess
    import threading

    # Start server in background
    from src.api.api_server import create_app, get_state
    portfolio = make_test_portfolio()
    state     = get_state()
    state.set_portfolio(portfolio)
    app       = create_app(portfolio=portfolio)

    def run_server():
        try:
            import uvicorn
            uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")
        except Exception:
            pass

    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    time.sleep(2)   # Wait for startup

    print(f"\n  Testing API endpoints (http://localhost:{port})...\n")

    try:
        import requests as req
        headers = {"X-API-Key": "dev-key-1"}
        base    = f"http://localhost:{port}"

        endpoints = [
            ("GET", "/"),
            ("GET", "/portfolio"),
            ("GET", "/portfolio/history?days=30"),
            ("GET", "/portfolio/sector-allocation"),
            ("GET", "/risk"),
            ("GET", "/performance"),
            ("GET", "/performance/monthly"),
            ("GET", "/trades"),
        ]

        all_passed = True
        for method, ep in endpoints:
            try:
                r = req.get(f"{base}{ep}", headers=headers, timeout=5)
                status = "✓" if r.status_code == 200 else "✗"
                if r.status_code != 200:
                    all_passed = False
                print(f"  {status} {method} {ep:<40} {r.status_code}")
            except Exception as e:
                print(f"  ✗ {method} {ep:<40} ERROR: {e}")
                all_passed = False

        if all_passed:
            print("\n  ✅ All endpoints passing")
        else:
            print("\n  ⚠  Some endpoints failed")

    except ImportError:
        print("  Install requests to run API tests: pip install requests")


def generate_report(report_type: str):
    """Generate a PDF report."""
    print(f"\n{'═'*65}")
    print(f"  Generating {report_type.upper()} Report")
    print(f"{'═'*65}")

    from src.reports.pdf_generator import generate_fund_report
    from src.api.api_server import (
        _demo_portfolio, _demo_risk, _demo_performance, _generate_demo_nav_history
    )

    nav_history = _generate_demo_nav_history(1_000_000, 90)

    path = generate_fund_report(
        portfolio_data   = _demo_portfolio(),
        performance_data = _demo_performance(),
        risk_data        = _demo_risk(),
        nav_history      = nav_history,
        report_type      = report_type,
        commentary       = (
            "The portfolio delivered solid absolute and risk-adjusted returns this month. "
            "Technology holdings were the primary driver of positive performance, while "
            "energy positions provided meaningful diversification benefit.\n\n"
            "Risk metrics remained well within target ranges throughout the period. "
            "Portfolio beta averaged 0.91, and daily VaR stayed comfortably below "
            "the 2.0% NAV limit on every trading day.\n\n"
            "We maintained our systematic approach without discretionary overrides. "
            "The momentum signal continued to identify opportunities in large-cap technology, "
            "while our mean-reversion overlay captured several intraday dislocations."
        ) if report_type == "monthly" else "",
    )

    print(f"\n  ✓ Report generated: {path}")
    print(f"  Size: {Path(path).stat().st_size / 1024:.1f} KB")


def run_query(question: str):
    """Answer an investor question using the Dashboard Agent."""
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  No LLM key — cannot generate response")
        return

    from src.agents.dashboard_agent import DashboardAgent
    agent = DashboardAgent()

    print(f"\n  Q: {question}\n")
    answer = agent.answer_investor_query(question)
    print(f"  A: {answer}")


def generate_commentary():
    """Generate monthly commentary."""
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  No LLM key — showing template only")
        print("\nTemplate commentary:")
        print(
            "[Month] was [positive/challenging] for the portfolio. "
            "[Primary driver of returns]. The [factor/sector] contributed "
            "[X]% to total returns.\n\n"
            "Risk metrics remained [within/near] target ranges throughout "
            "the period. Portfolio beta averaged [X] versus our target of [Y], "
            "and daily VaR stayed [below/within] the [X]% limit on [all/most] trading days.\n\n"
            "Positioning for [next month] remains [description]. We expect "
            "[factor/theme] to continue [description]."
        )
        return

    from src.agents.dashboard_agent import DashboardAgent
    agent       = DashboardAgent()
    commentary  = agent.generate_monthly_commentary()
    print(f"\n  Monthly Commentary:\n\n{commentary}")


def generate_eod():
    """Generate end-of-day summary."""
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  No LLM key — showing demo summary")
        from src.api.api_server import _demo_risk, _demo_portfolio
        risk = _demo_risk()
        port = _demo_portfolio()
        print(f"\nEnd-of-Day Summary — {datetime.now():%Y-%m-%d}")
        print(f"• NAV: ${port['nav']:,.2f}")
        print(f"• Daily P&L: {risk['daily_pnl_pct']*100:+.2f}%")
        print(f"• VaR (95%): {risk['var_95_pct']*100:.2f}% of NAV")
        print(f"• Beta: {risk['portfolio_beta']:.2f}")
        print(f"• Risk level: {risk['risk_level']}")
        return

    from src.agents.dashboard_agent import DashboardAgent
    agent   = DashboardAgent()
    summary = agent.generate_eod_summary()
    print(f"\n  End-of-Day Summary:\n\n{summary}")


def run_demo():
    print(f"\n{'╔'+'═'*58+'╗'}")
    print(f"{'║'+'  PART 8: INVESTOR DASHBOARD DEMO'.center(58)+'║'}")
    print(f"{'╚'+'═'*58+'╝'}")

    print("\n[1/3] Generating monthly investor letter PDF...")
    generate_report("monthly")

    print("\n[2/3] Generating daily risk report PDF...")
    generate_report("daily_risk")

    print("\n[3/3] API server endpoints (start with --server to run live):")
    print("  GET /portfolio     → Current NAV and positions")
    print("  GET /risk          → Live VaR, beta, drawdown")
    print("  GET /performance   → Sharpe, returns, drawdown")
    print("  GET /trades        → Execution history")
    print("  WS  /ws/risk       → Real-time risk stream")
    print("  GET /docs          → Interactive Swagger docs")
    print("\n  Start with: python run_part8.py --server")


def main():
    parser = argparse.ArgumentParser(
        description="AI Hedge Fund — Part 8: Investor Dashboard"
    )
    parser.add_argument("--server",     action="store_true", help="Start API server")
    parser.add_argument("--port",       type=int, default=8000, help="Server port")
    parser.add_argument("--test-api",   action="store_true", help="Test API endpoints")
    parser.add_argument("--report",     choices=["monthly","daily_risk"], help="Generate PDF")
    parser.add_argument("--query",      type=str, help="Investor query (LLM)")
    parser.add_argument("--commentary", action="store_true", help="Monthly commentary")
    parser.add_argument("--eod",        action="store_true", help="EOD summary")
    parser.add_argument("--demo",       action="store_true", help="Full demo")
    parser.add_argument("--validate",   action="store_true", help="Check environment")
    args = parser.parse_args()

    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + "  AI HEDGE FUND — PART 8: INVESTOR DASHBOARD".center(58) + "║")
    print("║" + f"  {datetime.now():%Y-%m-%d %H:%M:%S}".center(58) + "║")
    print("╚" + "═"*58 + "╝")

    validate_environment()

    if args.validate:       return
    elif args.server:       start_server(args.port)
    elif args.test_api:     test_api(args.port)
    elif args.report:       generate_report(args.report)
    elif args.query:        run_query(args.query)
    elif args.commentary:   generate_commentary()
    elif args.eod:          generate_eod()
    elif args.demo:         run_demo()
    else:
        print("\n  No command — running demo")
        run_demo()

    print("\n✅ Part 8 complete.")
    print("   Next: Part 9 — Cloud Production (AWS, Docker, CI/CD)")


if __name__ == "__main__":
    main()
