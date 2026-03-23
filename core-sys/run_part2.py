"""
AI Hedge Fund — Part 2: Multi-Agent System
============================================
run_part2.py — Main entry point

Demonstrates the full multi-agent system:
  - Research Analyst Agent  (deep security analysis)
  - Risk Manager Agent      (pre-trade and portfolio risk)
  - Portfolio Manager Agent (allocation decisions)
  - Agent Coordinator       (consensus protocol)
  - Message Bus             (agent-to-agent communication)

Usage:
    # Full multi-agent consensus on a single ticker:
    python run_part2.py --ticker AAPL

    # Fast mode (skip Research Analyst):
    python run_part2.py --ticker NVDA --fast

    # Scan the full universe:
    python run_part2.py --scan --top 5

    # Test only risk manager (no LLM needed):
    python run_part2.py --risk-only --ticker AAPL

    # Test message bus:
    python run_part2.py --test-bus

    # Show decision history:
    python run_part2.py --history
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import cfg, setup_logging

logger = setup_logging()


def validate_environment() -> bool:
    """Quick environment check."""
    print("\n" + "═" * 60)
    print("  Part 2: Multi-Agent System — Environment Check")
    print("═" * 60)

    import os
    has_llm = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"))

    pkgs = [("yfinance","pip install yfinance"),
            ("numpy","pip install numpy"),
            ("pandas","pip install pandas"),
            ("scipy","pip install scipy")]

    all_ok = True
    for pkg, install in pkgs:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} — {install}")
            all_ok = False

    print(f"\n  LLM key: {'✓ configured' if has_llm else '✗ not configured (required for agents)'}")

    from src.comms.message_bus import MessageBus
    bus = MessageBus()
    print(f"  ✓ MessageBus ready")

    return has_llm and all_ok


def test_message_bus():
    """Test the message bus independently."""
    print("\n" + "═" * 60)
    print("  MessageBus Test")
    print("═" * 60)

    from src.comms.message_bus import MessageBus, Message, MessageType, Priority

    bus = MessageBus()

    # 1: Publish + consume
    print("\n1. Publish / Consume cycle...")
    msg = Message.create(
        sender    = "PortfolioManager",
        recipient = "RiskManager",
        subject   = "pre_trade_check",
        payload   = {"ticker": "AAPL", "proposed_weight": 0.08},
        priority  = Priority.HIGH,
    )
    msg_id = bus.publish(msg)
    consumed = bus.consume("RiskManager")
    assert any(m.message_id == msg_id for m in consumed), "Message not consumed"
    for m in consumed:
        bus.ack(m.message_id)
    print(f"  ✓ Published and consumed {len(consumed)} message(s)")

    # 2: Alert broadcast
    print("\n2. Alert broadcast...")
    alert_id = bus.broadcast_alert(
        sender  = "RiskManager",
        subject = "VAR_WARNING",
        payload = {"var_pct": 1.8, "limit": 2.0},
    )
    for agent in ["PortfolioManager", "ResearchAnalyst", "AgentCoordinator"]:
        msgs = bus.consume(agent)
        for m in msgs:
            bus.ack(m.message_id)
    print(f"  ✓ Alert broadcast to all agents")

    # 3: Stats
    stats = bus.get_stats()
    print(f"\n3. Bus stats: published={stats['total_published']}, "
          f"delivered={stats['total_delivered']}")

    print("\n✅ MessageBus tests passed")


def test_risk_manager(ticker: str = "AAPL"):
    """Test Risk Manager Agent without LLM."""
    print("\n" + "═" * 60)
    print(f"  Risk Manager Test — {ticker}")
    print("═" * 60)

    from src.data.data_models import Portfolio, Position, Direction
    from src.agents.risk_manager_agent import RiskManagerAgent, RiskEngine

    port = Portfolio("RISK_TEST", cash=850000, initial_capital=1000000)
    port.positions["MSFT"] = Position(
        "MSFT", Direction.LONG, 200, 380.0, 415.0, sector="Technology"
    )
    port.positions["JPM"] = Position(
        "JPM", Direction.LONG, 500, 180.0, 195.0, sector="Financials"
    )

    rm = RiskManagerAgent(port)

    print(f"\n  Portfolio NAV: ${port.net_asset_value:,.0f}")
    print(f"  Positions: {list(port.positions.keys())}")

    # Test pre-trade check
    print(f"\n  Pre-trade check: BUY {ticker} 10%...")
    check = rm.pre_trade_check(ticker, proposed_weight=0.10, current_weight=0.0)
    print(check.summary())

    # Test oversized position
    print(f"\n  Pre-trade check: BUY {ticker} 25% (should fail)...")
    check2 = rm.pre_trade_check(ticker, proposed_weight=0.25, current_weight=0.0)
    print(check2.summary())

    # Portfolio risk summary
    print("\n  Portfolio Risk Summary:")
    summary = json.loads(rm._tool_portfolio_risk_summary())
    for k, v in summary.items():
        if v is not None:
            print(f"    {k}: {v}")

    # Stress tests
    print("\n  Stress Tests:")
    stress = json.loads(rm._tool_stress_test())
    for scenario, result in stress.items():
        print(f"    {scenario}: "
              f"${result['pnl_usd']:>12,.0f} "
              f"({result['pnl_pct_nav']:+.1f}% NAV) "
              f"— {result['severity']}")

    print(f"\n  {rm.get_metrics()}")
    print("\n✅ Risk Manager tests passed")


def run_full_consensus(ticker: str, fast_mode: bool = False):
    """Run the complete multi-agent consensus."""
    print("\n" + "═" * 60)
    print(f"  Multi-Agent Consensus — {ticker}")
    print(f"  Mode: {'Fast (no research)' if fast_mode else 'Full (with research)'}")
    print("═" * 60)

    from src.data.data_models import Portfolio, Position, Direction
    from src.agents.agent_coordinator import AgentCoordinator

    # Sample portfolio
    portfolio = Portfolio(
        portfolio_id    = "FUND_001",
        cash            = cfg.INITIAL_CAPITAL * 0.80,
        initial_capital = cfg.INITIAL_CAPITAL,
    )
    portfolio.positions["MSFT"] = Position(
        "MSFT", Direction.LONG, 200, 380.0, 415.0, sector="Technology"
    )
    portfolio.positions["JPM"] = Position(
        "JPM", Direction.LONG, 500, 180.0, 195.0, sector="Financials"
    )

    print(f"\n  Portfolio NAV: ${portfolio.net_asset_value:,.0f}")
    print(f"  Existing positions: {list(portfolio.positions.keys())}")

    coordinator = AgentCoordinator(portfolio)

    print(f"\n  Starting consensus for {ticker}...")
    print("  (This may take 1-5 minutes)\n")

    result = coordinator.decide(ticker, fast_mode=fast_mode)

    print(result.summary())

    # Save result
    output_path = (
        Path(__file__).parent / "logs" /
        f"consensus_{ticker}_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"\n  Saved to: {output_path}")

    print(f"\n  Total LLM cost: ${result.total_llm_cost:.4f}")


def run_universe_scan(top_n: int = 5):
    """Scan the full universe."""
    print("\n" + "═" * 60)
    print(f"  Universe Scan — Top {top_n} Opportunities")
    print("═" * 60)

    from src.data.data_models import Portfolio
    from src.agents.agent_coordinator import AgentCoordinator

    portfolio = Portfolio(
        portfolio_id    = "FUND_001",
        cash            = cfg.INITIAL_CAPITAL,
        initial_capital = cfg.INITIAL_CAPITAL,
    )

    coordinator = AgentCoordinator(portfolio)

    print(f"\n  Scanning {len(cfg.DEFAULT_UNIVERSE)} securities (fast mode)...")
    print("  Estimated cost: $0.10–$0.50 | Time: 5–15 minutes\n")

    results = coordinator.scan_universe(fast_mode=True, top_n=top_n)

    print(f"\n  Top {top_n} Results:")
    print("  " + "─" * 60)
    for i, r in enumerate(results, 1):
        flag = "🟢" if r.final_decision == "BUY" else "🔴" if r.final_decision == "SELL" else "⚪"
        print(f"  {i}. {flag} {r.ticker:<6} | {r.final_decision:<5} | "
              f"Weight: {r.final_weight:.1%} | "
              f"Votes: {r.buy_votes}/{r.total_votes} BUY | "
              f"Conf: {r.avg_confidence:.0%}")

    print(f"\n  Total spend: ${coordinator.total_spend():.4f}")
    print(coordinator.print_daily_report())


def show_history():
    """Show recent decision history."""
    from src.agents.agent_coordinator import DecisionDatabase
    db = DecisionDatabase()
    stats = db.decision_stats(days=30)
    print("\n═" * 60)
    print("  Decision History (Last 30 Days)")
    print("═" * 60)
    if not stats:
        print("  No decisions recorded yet.")
    else:
        for decision_type, data in stats.items():
            print(f"  {decision_type:<12} {data['count']} decisions | "
                  f"avg conf {data['avg_confidence'] or 0:.0%} | "
                  f"${data['total_cost'] or 0:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="AI Hedge Fund — Part 2: Multi-Agent System"
    )
    parser.add_argument("--ticker",    default="AAPL",  help="Ticker to analyse")
    parser.add_argument("--fast",      action="store_true", help="Fast mode (skip research)")
    parser.add_argument("--scan",      action="store_true", help="Scan full universe")
    parser.add_argument("--top",       type=int, default=5,  help="Top N for scan")
    parser.add_argument("--risk-only", action="store_true", help="Risk tests only (no LLM)")
    parser.add_argument("--test-bus",  action="store_true", help="Test message bus")
    parser.add_argument("--history",   action="store_true", help="Show decision history")
    parser.add_argument("--validate",  action="store_true", help="Validate environment only")
    args = parser.parse_args()

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + "  AI HEDGE FUND — PART 2: MULTI-AGENT SYSTEM".center(58) + "║")
    print("║" + f"  {datetime.now():%Y-%m-%d %H:%M:%S}".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    if args.history:
        show_history()
        return

    if args.test_bus:
        test_message_bus()
        return

    if args.risk_only:
        test_risk_manager(args.ticker)
        return

    agent_ready = validate_environment()

    if args.validate:
        sys.exit(0 if agent_ready else 1)

    if not agent_ready:
        print("\n  Running component tests (no LLM key configured)...")
        test_message_bus()
        test_risk_manager(args.ticker)
        print("\n  Add ANTHROPIC_API_KEY or OPENAI_API_KEY to .env for full agents.")
        return

    if args.scan:
        run_universe_scan(top_n=args.top)
    else:
        run_full_consensus(ticker=args.ticker, fast_mode=args.fast)

    print("\n✅ Part 2 complete.")
    print("   Next: Part 3 — RAG & Data Intelligence (earnings transcripts, SEC filings)")


if __name__ == "__main__":
    main()
