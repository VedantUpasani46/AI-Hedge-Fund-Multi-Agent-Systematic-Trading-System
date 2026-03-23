"""
AI Hedge Fund — Part 7: Real-Time Risk Management
===================================================
run_part7.py — Main Entry Point

Usage:
    # Start live risk monitor (runs until Ctrl+C):
    python run_part7.py --monitor

    # One-shot risk snapshot (no background thread):
    python run_part7.py --snapshot

    # Factor exposure report:
    python run_part7.py --factors

    # Circuit breaker status:
    python run_part7.py --circuit-breakers

    # Position reduction plan (simulate a VaR breach):
    python run_part7.py --reduction-plan

    # Full dashboard (snapshot + factors + liquidity):
    python run_part7.py --dashboard

    # Run Realtime Risk Agent with LLM:
    python run_part7.py --agent "Is the portfolio at risk today?"

    # Full demo:
    python run_part7.py --demo
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import cfg, setup_logging

logger = setup_logging()


def make_test_portfolio():
    """Build a representative test portfolio."""
    from src.data.data_models import Portfolio, Position, Direction

    portfolio = Portfolio("FUND_001", cash=650_000, initial_capital=1_000_000)

    test_positions = [
        ("AAPL", 500,  180.0, 195.0, "Technology"),
        ("MSFT", 300,  380.0, 415.0, "Technology"),
        ("NVDA", 120,  450.0, 485.0, "Technology"),
        ("JPM",  800,  185.0, 198.0, "Financials"),
        ("XOM",  600,  105.0, 112.0, "Energy"),
    ]
    for ticker, shares, avg_cost, current, sector in test_positions:
        portfolio.positions[ticker] = Position(
            ticker, Direction.LONG, shares, avg_cost, current, sector=sector
        )
    return portfolio


def validate_environment() -> bool:
    print("\n" + "═" * 65)
    print("  Part 7: Real-Time Risk Management — Environment Check")
    print("═" * 65)

    has_llm = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"))

    pkgs = [
        ("numpy",   "pip install numpy"),
        ("pandas",  "pip install pandas"),
        ("scipy",   "pip install scipy"),
        ("yfinance","pip install yfinance"),
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

    print(f"\n  LLM key: {'✓' if has_llm else '✗ (optional — for agent mode)'}")
    return all_ok


def run_snapshot():
    """One-shot risk snapshot."""
    print(f"\n{'═'*65}")
    print("  Live Risk Snapshot")
    print(f"{'═'*65}")

    from src.risk.live_risk_engine import LiveRiskEngine

    portfolio = make_test_portfolio()
    engine    = LiveRiskEngine(portfolio, poll_interval_seconds=999)

    print("\n  Fetching live prices and computing risk metrics...")
    snap = engine._compute_snapshot()

    if snap:
        print("\n" + engine.dashboard_summary())
    else:
        print("  Could not compute snapshot — check internet connection")


def run_factor_report():
    """Factor exposure report."""
    print(f"\n{'═'*65}")
    print("  Factor Exposure Report")
    print(f"{'═'*65}")

    from src.risk.factor_monitor import FactorMonitor

    portfolio   = make_test_portfolio()
    monitor     = FactorMonitor(
        target_exposures={
            "MKT": {"min": 0.5, "max": 1.3, "target": 0.9},
            "MOM": {"min": -0.1, "max": 0.5},
            "SMB": {"min": -0.3, "max": 0.3},
        }
    )
    positions   = {
        t: getattr(p, "shares", 0) * getattr(p, "current_price", 0)
        for t, p in portfolio.positions.items()
    }
    nav         = portfolio.net_asset_value

    print(f"\n  Portfolio NAV: ${nav:,.0f}")
    print(f"  Positions: {list(portfolio.positions.keys())}")
    print(f"\n  Computing factor exposures...")

    fsnap = monitor.compute(positions, nav)
    print("\n" + fsnap.summary())

    if fsnap.has_breaches():
        print("\n  ❌ Factor exposure breaches detected")
    elif fsnap.has_warnings():
        print("\n  ⚠  Factor exposure warnings")
    else:
        print("\n  ✓  All factor exposures within target ranges")


def run_circuit_breakers():
    """Show circuit breaker status."""
    print(f"\n{'═'*65}")
    print("  Circuit Breaker Status")
    print(f"{'═'*65}")

    from src.risk.live_risk_engine import build_circuit_breakers

    cbs = build_circuit_breakers(
        max_daily_loss_pct    = 0.02,
        max_var_pct           = 0.02,
        max_intraday_drawdown = 0.015,
        max_margin_utilisation= 0.80,
        max_beta              = 1.5,
    )

    print(f"\n  {'Name':<28} {'Severity':<8} {'Status':<12} Description")
    print(f"  {'─'*75}")
    for cb in cbs:
        status = "TRIGGERED" if cb.triggered else "OK"
        print(f"  {cb.name:<28} {cb.severity:<8} {status:<12} {cb.description}")


def run_reduction_plan():
    """Simulate a VaR breach and show the reduction plan."""
    print(f"\n{'═'*65}")
    print("  Position Reduction Plan (simulated VaR breach)")
    print(f"{'═'*65}")

    from src.risk.live_risk_engine import LiveRiskEngine

    portfolio = make_test_portfolio()
    engine    = LiveRiskEngine(portfolio, poll_interval_seconds=999)

    snap = engine._compute_snapshot()
    if not snap:
        print("  No price data — cannot compute reduction plan")
        return

    # Simulate: pretend VaR is 2.5% (above 2.0% limit)
    print(f"\n  Current portfolio: ${snap.nav:,.0f}")
    print(f"  Current VaR (95%): {snap.var_95_pct:.2%} of NAV")
    print(f"  VaR limit:         2.00%")

    if snap.var_95_pct > 0.020:
        print(f"\n  ❌ VaR BREACH — reduction required")
    else:
        print(f"\n  ✓  VaR within limits (limit is 2.0%)")

    print(f"\n  Reduction plan (target VaR: 1.8%):")
    print(f"  {'Ticker':<8} {'Current%':>10} {'Target%':>9} {'Sell Shares':>12} {'VaR Relief':>12}")
    print(f"  {'─'*55}")

    sorted_pos = sorted(
        snap.positions.items(),
        key=lambda x: x[1].var_95_1d,
        reverse=True,
    )
    for ticker, pos in sorted_pos[:4]:
        target_w = pos.portfolio_weight * 0.75
        shares_sell = pos.shares * 0.25
        var_relief  = pos.var_95_1d * 0.25
        print(
            f"  {ticker:<8} {pos.portfolio_weight*100:>9.1f}% "
            f"{target_w*100:>8.1f}% "
            f"{shares_sell:>12.0f} "
            f"${var_relief:>10,.0f}"
        )


def run_live_monitor():
    """Start continuous live monitoring (Ctrl+C to stop)."""
    print(f"\n{'═'*65}")
    print("  Live Risk Monitor — Press Ctrl+C to stop")
    print(f"{'═'*65}")

    from src.risk.live_risk_engine import LiveRiskEngine

    portfolio = make_test_portfolio()
    engine    = LiveRiskEngine(portfolio, poll_interval_seconds=15)

    # Circuit breaker alert
    def on_alert(name, info):
        print(f"\n  {'='*50}")
        print(f"  ❌ CIRCUIT BREAKER: {name}")
        print(f"  {json.dumps(info, indent=2)}")
        print(f"  {'='*50}")

    engine.register_alert_callback(on_alert)
    engine.start()

    print(f"\n  Monitoring started. Updating every 15 seconds...")
    print(f"  Portfolio: {list(portfolio.positions.keys())}\n")

    try:
        while True:
            time.sleep(15)
            if engine.current_snapshot:
                print(engine.current_snapshot.summary_line())
    except KeyboardInterrupt:
        print("\n\n  Stopping monitor...")
    finally:
        engine.stop()
        print("  Monitor stopped.")


def run_full_dashboard():
    """Full dashboard: snapshot + factors + circuit breakers."""
    print(f"\n{'╔'+'═'*58+'╗'}")
    print(f"{'║'+'  RISK MANAGEMENT DASHBOARD'.center(58)+'║'}")
    print(f"{'╚'+'═'*58+'╝'}")

    run_snapshot()
    run_factor_report()
    run_circuit_breakers()


def run_agent(query: str):
    """Run the Realtime Risk Agent with an LLM query."""
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  No LLM key — running snapshot instead")
        run_snapshot()
        return

    from src.agents.realtime_risk_agent import RealtimeRiskAgent

    portfolio = make_test_portfolio()
    agent     = RealtimeRiskAgent(portfolio)

    print(f"\n  Query: {query}\n")
    response_text, _ = agent.think(
        user_message = query,
        use_tools    = True,
        purpose      = "risk_dashboard_query",
    )
    print(f"  Response:\n{response_text[:1500]}")


def run_demo():
    print(f"\n{'╔'+'═'*58+'╗'}")
    print(f"{'║'+'  PART 7: REAL-TIME RISK MANAGEMENT DEMO'.center(58)+'║'}")
    print(f"{'╚'+'═'*58+'╝'}")

    print("\n[1/4] Live risk snapshot...")
    run_snapshot()

    print("\n[2/4] Factor exposures...")
    run_factor_report()

    print("\n[3/4] Circuit breaker definitions...")
    run_circuit_breakers()

    print("\n[4/4] Position reduction plan...")
    run_reduction_plan()


def main():
    parser = argparse.ArgumentParser(
        description="AI Hedge Fund — Part 7: Real-Time Risk Management"
    )
    parser.add_argument("--monitor",         action="store_true", help="Start live monitor")
    parser.add_argument("--snapshot",        action="store_true", help="One-shot risk snapshot")
    parser.add_argument("--factors",         action="store_true", help="Factor exposure report")
    parser.add_argument("--circuit-breakers",action="store_true", help="Circuit breaker status")
    parser.add_argument("--reduction-plan",  action="store_true", help="Position reduction plan")
    parser.add_argument("--dashboard",       action="store_true", help="Full dashboard")
    parser.add_argument("--agent",           type=str,            help="LLM risk query")
    parser.add_argument("--demo",            action="store_true", help="Full demo")
    parser.add_argument("--validate",        action="store_true", help="Validate environment")
    args = parser.parse_args()

    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + "  AI HEDGE FUND — PART 7: REAL-TIME RISK".center(58) + "║")
    print("║" + f"  {datetime.now():%Y-%m-%d %H:%M:%S}".center(58) + "║")
    print("╚" + "═"*58 + "╝")

    validate_environment()

    if args.validate:         return
    elif args.monitor:        run_live_monitor()
    elif args.snapshot:       run_snapshot()
    elif args.factors:        run_factor_report()
    elif getattr(args, "circuit_breakers", False): run_circuit_breakers()
    elif getattr(args, "reduction_plan", False):   run_reduction_plan()
    elif args.dashboard:      run_full_dashboard()
    elif args.agent:          run_agent(args.agent)
    elif args.demo:           run_demo()
    else:
        print("\n  No command — running full dashboard")
        run_full_dashboard()

    print("\n✅ Part 7 complete.")
    print("   Next: Part 8 — Investor Dashboard (FastAPI + React)")


if __name__ == "__main__":
    main()
