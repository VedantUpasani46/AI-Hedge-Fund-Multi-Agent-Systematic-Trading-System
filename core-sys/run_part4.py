"""
AI Hedge Fund — Part 4: Execution Engine
==========================================
run_part4.py — Main Entry Point

Demonstrates:
  - Almgren-Chriss optimal execution scheduling
  - IB broker interface (paper trading or simulation)
  - Order lifecycle management and TCA
  - Execution Agent with LLM-driven algo selection

Usage:
    # Test AC optimiser (no broker needed):
    python run_part4.py --ac-test AAPL 10000 BUY

    # Full execution test (simulation mode):
    python run_part4.py --execute AAPL BUY 5

    # With IB paper trading (must have TWS running on port 7497):
    python run_part4.py --execute AAPL BUY 5 --ib

    # Compare execution algorithms:
    python run_part4.py --compare NVDA 5000 BUY

    # Run full Execution Agent with LLM:
    python run_part4.py --agent AAPL BUY 5

    # Show today's execution summary:
    python run_part4.py --summary
"""

import argparse
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import cfg, setup_logging

logger = setup_logging()


def validate_environment() -> bool:
    print("\n" + "═" * 60)
    print("  Part 4: Execution Engine — Environment Check")
    print("═" * 60)

    import os
    has_llm = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"))

    pkgs = [
        ("yfinance",  "pip install yfinance"),
        ("numpy",     "pip install numpy"),
        ("pandas",    "pip install pandas"),
    ]
    optional = [
        ("ib_insync", "pip install ib_insync  # For IB live/paper trading"),
        ("pytz",      "pip install pytz       # For market hours"),
    ]

    all_ok = True
    print("\n  Core packages:")
    for pkg, install in pkgs:
        try:
            __import__(pkg)
            print(f"    ✓ {pkg}")
        except ImportError:
            print(f"    ✗ {pkg} — {install}")
            all_ok = False

    print("\n  Optional packages:")
    for pkg, install in optional:
        try:
            __import__(pkg)
            print(f"    ✓ {pkg}")
        except ImportError:
            print(f"    ○ {pkg} — {install}")

    print(f"\n  LLM key: {'✓' if has_llm else '✗ (needed for Execution Agent)'}")
    return all_ok


def test_almgren_chriss(ticker: str, shares: float, side: str):
    """Demonstrate Almgren-Chriss optimiser directly."""
    print(f"\n{'═'*60}")
    print(f"  Almgren-Chriss Optimisation — {side} {shares:,.0f} {ticker}")
    print(f"{'═'*60}")

    from src.execution.almgren_chriss import (
        MarketImpactParams, AlmgrenChrissOptimiser
    )
    from src.data.market_data import MarketDataFetcher
    import numpy as np

    fetcher = MarketDataFetcher()
    df = fetcher.get_prices(ticker, days=30)

    if df.empty:
        print(f"  Could not fetch data for {ticker}")
        return

    col   = "Adj Close" if "Adj Close" in df.columns else "Close"
    price = float(df[col].iloc[-1])
    log_r = np.log(df[col] / df[col].shift(1)).dropna()
    vol   = float(log_r.tail(21).std())
    adv   = float(df["Volume"].tail(21).mean()) if "Volume" in df.columns else 10_000_000

    print(f"\n  Market data:")
    print(f"    Price:      ${price:.2f}")
    print(f"    Daily vol:  {vol:.2%}")
    print(f"    ADV:        {adv/1e6:.1f}M shares")
    print(f"    Order size: {shares/adv*100:.2f}% of ADV")

    params    = MarketImpactParams(ticker=ticker, price=price, daily_vol=vol, adv=adv)
    optimiser = AlmgrenChrissOptimiser(params)

    print(f"\n  η (temporary impact): {params.eta:.2e}")
    print(f"  γ (permanent impact): {params.gamma:.2e}")
    print(f"  Cost per 1% ADV:      {params.cost_of_trading_1pct_adv():.1f}bps")

    # Compare all algos
    print(f"\n  Algorithm comparison ({shares:,.0f} shares, 60-min horizon):")
    print(f"  {'Algo':<8} {'Cost (bps)':>12} {'Cost ($)':>12} {'κ':>8}")
    print(f"  {'─'*44}")

    schedules = optimiser.compare_algos(shares, side, horizon_minutes=60, n_periods=12)
    for algo_name, sched in schedules.items():
        print(
            f"  {algo_name:<8} {sched.expected_cost_bps:>12.2f} "
            f"${sched.expected_cost_usd:>10,.0f} "
            f"{sched.kappa:>8.4f}"
        )

    # Show IS schedule
    is_sched = schedules["IS"]
    print(f"\n  Optimal IS schedule (κ={is_sched.kappa:.4f}):")
    print(f"  {'Period':<8} {'Time':<10} {'Trade':>8} {'Remaining':>12}")
    print(f"  {'─'*42}")
    for i in range(len(is_sched.trade_list)):
        print(
            f"  {i+1:<8} "
            f"{is_sched.timestamps[i].strftime('%H:%M'):<10} "
            f"{is_sched.trade_list[i]:>8.0f} "
            f"{is_sched.inventory[i]:>12.0f}"
        )

    notional = shares * price
    print(f"\n  Total notional: ${notional:,.0f}")
    print(f"  Optimal cost:   ${is_sched.expected_cost_usd:,.2f} "
          f"({is_sched.expected_cost_bps:.2f}bps)")
    print(f"  vs TWAP:        "
          f"{'savings' if is_sched.expected_cost_bps < schedules['TWAP'].expected_cost_bps else 'more expensive'} "
          f"by {abs(is_sched.expected_cost_bps - schedules['TWAP'].expected_cost_bps):.2f}bps")


def run_execution_test(ticker: str, side: str, weight_pct: float, use_ib: bool = False):
    """Test full execution pipeline."""
    print(f"\n{'═'*60}")
    print(f"  Execution Test — {side} {ticker} {weight_pct:.0f}% of NAV")
    print(f"{'═'*60}")

    from src.execution.ib_broker import IBBroker, IBConfig
    from src.execution.order_manager import OrderManager
    from src.execution.order_models import OrderStatus

    config = IBConfig(port=7497 if use_ib else 7497)
    broker = IBBroker(config)
    om     = OrderManager(broker=broker)

    connected = om.connect()
    print(f"\n  Mode: {om.broker.mode} | Connected: {connected}")

    account = om.broker.get_account_state()
    nav     = account.net_liquidation
    print(f"  NAV: ${nav:,.0f}")

    print(f"\n  Executing {side} {ticker} {weight_pct:.0f}%...")
    orders = om.execute_decision(
        ticker         = ticker,
        side           = side,
        target_weight  = weight_pct / 100,
        portfolio_nav  = nav,
        current_weight = 0.0,
        decision_id    = f"TEST_{ticker}_{datetime.now():%H%M%S}",
        agent_name     = "TestRunner",
        urgency        = "NORMAL",
        use_algo       = True,
    )

    print(f"\n  Submitted {len(orders)} order(s):")
    for o in orders:
        print(f"    {o.order_id}: {o.status.value} | "
              f"{o.side.value} {o.quantity:.0f} shares")

    import time; time.sleep(0.3)

    print(f"\n  After fill:")
    for o in orders:
        if o.avg_fill_price:
            print(
                f"    {o.order_id}: FILLED @ ${o.avg_fill_price:.4f} | "
                f"commission=${o.commission:.2f} | "
                f"IS={o.implementation_shortfall_bps:+.1f}bps"
                if o.implementation_shortfall_bps else
                f"    {o.order_id}: {o.status.value}"
            )

    summary = om.get_execution_summary()
    print(f"\n  Today's summary:")
    print(f"    Orders: {summary['total_orders']}")
    print(f"    Filled: {summary['filled']}")
    if summary.get("tca"):
        tca = summary["tca"]
        print(f"    Avg IS (bps): {tca.get('avg_is_bps', 0):.2f}")

    om.disconnect()


def compare_algos(ticker: str, shares: float, side: str):
    """Detailed algorithm comparison."""
    print(f"\n{'═'*60}")
    print(f"  Algorithm Comparison — {side} {shares:,.0f} {ticker}")
    print(f"{'═'*60}")

    from src.execution.almgren_chriss import (
        MarketImpactParams, AlmgrenChrissOptimiser, PreTradeEstimator
    )
    from src.data.market_data import MarketDataFetcher
    import numpy as np

    fetcher = MarketDataFetcher()
    df = fetcher.get_prices(ticker, days=30)
    col   = "Adj Close" if "Adj Close" in df.columns else "Close"
    price = float(df[col].iloc[-1])
    log_r = np.log(df[col] / df[col].shift(1)).dropna()
    vol   = float(log_r.tail(21).std())
    adv   = float(df["Volume"].tail(21).mean()) if "Volume" in df.columns else 10_000_000

    params = MarketImpactParams(ticker=ticker, price=price, daily_vol=vol, adv=adv)

    print(f"\n  Testing across different horizons:")
    print(f"  {'Horizon':<10} {'IS (bps)':>10} {'TWAP (bps)':>12} {'VWAP (bps)':>12} {'IS savings':>12}")
    print(f"  {'─'*58}")

    for horizon in [15, 30, 60, 120, 240]:
        optimiser = AlmgrenChrissOptimiser(params)
        schedules = optimiser.compare_algos(shares, side, horizon, n_periods=max(3, int(horizon/5)))
        is_bps   = schedules["IS"].expected_cost_bps
        twap_bps = schedules["TWAP"].expected_cost_bps
        vwap_bps = schedules["VWAP"].expected_cost_bps
        savings  = twap_bps - is_bps
        print(
            f"  {horizon:<10}min {is_bps:>10.2f} {twap_bps:>12.2f} "
            f"{vwap_bps:>12.2f} {savings:>12.2f}"
        )

    estimator = PreTradeEstimator()
    est = estimator.estimate(ticker, shares, side, price, vol, adv, 60.0)
    print(f"\n  Pre-trade recommendation: {est.algo_recommended.value}")
    print(f"  Estimated total cost: ${est.total_estimated_cost_usd:,.2f}")
    print(f"  Participation rate: {est.participation_rate:.2%} of ADV")


def show_summary():
    """Show execution performance summary."""
    from src.execution.order_manager import ExecutionDatabase
    db  = ExecutionDatabase()
    tca = db.get_tca_summary(days=30)
    today_orders = db.get_today_orders()

    print(f"\n{'═'*60}")
    print("  Execution Performance Summary")
    print(f"{'═'*60}")
    print(f"\n  Today's orders: {len(today_orders)}")
    if today_orders:
        for o in today_orders[:5]:
            print(f"    {o['order_id']}: {o['side']} {o['quantity']:.0f} {o['ticker']} "
                  f"| {o['status']}")

    if tca:
        print(f"\n  30-day TCA:")
        print(f"    Orders analysed: {tca.get('n_orders', 0)}")
        print(f"    Avg IS (bps):    {tca.get('avg_is_bps', 0):.2f}")
        print(f"    vs VWAP (bps):   {tca.get('avg_vs_vwap_bps', 0):.2f}")
        print(f"    Total cost:      {tca.get('avg_total_cost_bps', 0):.2f}bps avg")
    else:
        print("\n  No TCA data yet — run some executions first")


def run_agent(ticker: str, side: str, weight_pct: float):
    """Run full Execution Agent with LLM."""
    print(f"\n{'═'*60}")
    print(f"  Execution Agent (LLM) — {side} {ticker} {weight_pct:.0f}%")
    print(f"{'═'*60}")

    import os
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\n  ⚠️  No LLM key — falling back to OrderManager test")
        run_execution_test(ticker, side, weight_pct)
        return

    from src.agents.execution_agent import ExecutionAgent
    agent = ExecutionAgent()
    agent.om.connect()

    print(f"\n  Running AI execution for {ticker}...")
    result = agent.execute(
        ticker        = ticker,
        side          = side,
        target_weight = weight_pct / 100,
        portfolio_nav = 1_000_000,
        decision_id   = f"AGENT_TEST_{datetime.now():%H%M%S}",
    )

    print(f"\n  Orders submitted: {result.get('order_ids', [])}")
    print(f"\n  LLM reasoning:\n  {result.get('llm_reasoning', '')[:500]}")
    print(f"\n  {agent.get_metrics()}")
    agent.om.disconnect()


def main():
    parser = argparse.ArgumentParser(
        description="AI Hedge Fund — Part 4: Execution Engine"
    )
    parser.add_argument("--ac-test",   nargs=3, metavar=("TICKER","SHARES","SIDE"),
                        help="Test Almgren-Chriss optimiser")
    parser.add_argument("--execute",   nargs=3, metavar=("TICKER","SIDE","PCT"),
                        help="Execute order (BUY AAPL 5 = buy 5%% of NAV)")
    parser.add_argument("--compare",   nargs=3, metavar=("TICKER","SHARES","SIDE"),
                        help="Compare execution algorithms")
    parser.add_argument("--agent",     nargs=3, metavar=("TICKER","SIDE","PCT"),
                        help="Run LLM Execution Agent")
    parser.add_argument("--summary",   action="store_true", help="Show execution summary")
    parser.add_argument("--ib",        action="store_true", help="Use IB (port 7497)")
    parser.add_argument("--validate",  action="store_true", help="Validate environment")
    args = parser.parse_args()

    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + "  AI HEDGE FUND — PART 4: EXECUTION ENGINE".center(58) + "║")
    print("║" + f"  {datetime.now():%Y-%m-%d %H:%M:%S}".center(58) + "║")
    print("╚" + "═"*58 + "╝")

    validate_environment()

    if args.validate:
        return

    if args.ac_test:
        test_almgren_chriss(args.ac_test[0], float(args.ac_test[1]), args.ac_test[2])
    elif args.compare:
        compare_algos(args.compare[0], float(args.compare[1]), args.compare[2])
    elif args.execute:
        run_execution_test(args.execute[0], args.execute[1], float(args.execute[2]), args.ib)
    elif args.agent:
        run_agent(args.agent[0], args.agent[1], float(args.agent[2]))
    elif args.summary:
        show_summary()
    else:
        # Default demo
        print("\n  No command — running default demo")
        test_almgren_chriss("AAPL", 10000, "BUY")
        run_execution_test("AAPL", "BUY", 5.0)

    print("\n✅ Part 4 complete.")
    print("   Next: Part 5 — Alternative Data Integration")


if __name__ == "__main__":
    main()
