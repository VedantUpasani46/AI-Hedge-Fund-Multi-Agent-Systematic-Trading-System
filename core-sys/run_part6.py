"""
AI Hedge Fund — Part 6: Backtesting Engine
============================================
run_part6.py — Main Entry Point

Usage:
    # Run a simple backtest:
    python run_part6.py --backtest momentum AAPL MSFT NVDA GOOGL JPM

    # Walk-forward validation:
    python run_part6.py --walk-forward momentum AAPL MSFT NVDA GOOGL JPM BAC XOM

    # Stress test a strategy:
    python run_part6.py --stress-test momentum AAPL MSFT NVDA GOOGL JPM

    # Full attribution report:
    python run_part6.py --attribution momentum AAPL MSFT NVDA GOOGL JPM

    # Compare two strategies:
    python run_part6.py --compare momentum mean_reversion AAPL MSFT NVDA GOOGL JPM

    # Mean reversion strategy:
    python run_part6.py --backtest mean_reversion AAPL MSFT NVDA GOOGL JPM BAC

    # Run Backtest Agent (requires LLM key):
    python run_part6.py --agent "Run walk-forward on momentum for AAPL MSFT NVDA GOOGL"

    # Full demo:
    python run_part6.py --demo
"""

import argparse
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import cfg, setup_logging

logger = setup_logging()

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "JPM", "BAC", "XOM", "JNJ", "PG"]
DEFAULT_START   = date(2019, 1, 1)
DEFAULT_END     = date(2023, 12, 31)


def validate_environment() -> bool:
    print("\n" + "═" * 65)
    print("  Part 6: Backtesting Engine — Environment Check")
    print("═" * 65)

    import os
    has_llm = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"))

    pkgs = [
        ("numpy",   "pip install numpy"),
        ("pandas",  "pip install pandas"),
        ("scipy",   "pip install scipy"),
        ("yfinance","pip install yfinance"),
    ]
    optional_pkgs = [
        ("pyarrow", "pip install pyarrow  # for parquet cache"),
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

    print("\n  Optional:")
    for pkg, install in optional_pkgs:
        try:
            __import__(pkg)
            print(f"    ✓ {pkg}")
        except ImportError:
            print(f"    ○ {pkg} — {install}")

    print(f"\n  LLM key: {'✓' if has_llm else '✗ (optional — for agent mode)'}")
    return all_ok


def run_backtest(strategy: str, tickers: list, start: date = None, end: date = None):
    print(f"\n{'═'*65}")
    print(f"  Backtest — {strategy.upper()} | {len(tickers)} tickers")
    print(f"{'═'*65}")

    from src.backtest.backtest_engine import (
        BacktestEngine, MomentumStrategy, MeanReversionStrategy
    )

    cls_map = {"momentum": MomentumStrategy, "mean_reversion": MeanReversionStrategy}
    cls = cls_map.get(strategy.lower())
    if not cls:
        print(f"  Unknown strategy: {strategy}. Available: {list(cls_map.keys())}")
        return

    engine = BacktestEngine()
    result = engine.run(
        strategy        = cls(),
        tickers         = tickers,
        start_date      = start or DEFAULT_START,
        end_date        = end   or DEFAULT_END,
        initial_capital = 1_000_000,
        verbose         = True,
    )
    print(result.summary())
    print("\nMonthly Returns (net of costs):")
    print(result.monthly_returns().to_string())


def run_walk_forward(strategy: str, tickers: list):
    print(f"\n{'═'*65}")
    print(f"  Walk-Forward Validation — {strategy.upper()}")
    print(f"{'═'*65}")

    from src.backtest.backtest_engine import MomentumStrategy, MeanReversionStrategy
    from src.backtest.walk_forward import WalkForwardEngine, WFConfig, WFValidationType

    cls_map = {"momentum": MomentumStrategy, "mean_reversion": MeanReversionStrategy}
    cls = cls_map.get(strategy.lower(), MomentumStrategy)

    param_grids = {
        "momentum":       {"top_n": [3, 5, 7], "lookback": [126, 189, 252]},
        "mean_reversion": {"entry_z": [1.5, 2.0, 2.5], "max_hold_days": [5, 10, 15]},
    }

    wf = WalkForwardEngine()
    result = wf.run(
        strategy_class  = cls,
        tickers         = tickers,
        start_date      = date(2018, 1, 1),
        end_date        = date(2023, 12, 31),
        initial_capital = 1_000_000,
        config          = WFConfig(n_folds=4, validation_type=WFValidationType.ANCHORED),
        param_grid      = param_grids.get(strategy.lower()),
    )
    print("\n" + result.summary())


def run_stress_test(strategy: str, tickers: list):
    print(f"\n{'═'*65}")
    print(f"  Stress Testing — {strategy.upper()}")
    print(f"{'═'*65}")

    from src.backtest.backtest_engine import BacktestEngine, MomentumStrategy, MeanReversionStrategy
    from src.backtest.stress_testing import StressTester

    cls_map = {"momentum": MomentumStrategy, "mean_reversion": MeanReversionStrategy}
    cls = cls_map.get(strategy.lower(), MomentumStrategy)

    engine = BacktestEngine()
    result = engine.run(
        strategy=cls(), tickers=tickers,
        start_date=date(2018, 1, 1), end_date=date(2023, 12, 31),
        initial_capital=1_000_000, verbose=True,
    )

    tester = StressTester()
    print("\n" + tester.full_stress_report(result, n_mc_sims=5_000))


def run_attribution(strategy: str, tickers: list):
    print(f"\n{'═'*65}")
    print(f"  Performance Attribution — {strategy.upper()}")
    print(f"{'═'*65}")

    from src.backtest.backtest_engine import BacktestEngine, MomentumStrategy, MeanReversionStrategy
    from src.attribution.performance_attribution import run_full_attribution

    cls_map = {"momentum": MomentumStrategy, "mean_reversion": MeanReversionStrategy}
    cls = cls_map.get(strategy.lower(), MomentumStrategy)

    engine = BacktestEngine()
    result = engine.run(
        strategy=cls(), tickers=tickers,
        start_date=date(2020, 1, 1), end_date=date(2023, 12, 31),
        initial_capital=1_000_000, verbose=True,
    )

    report = run_full_attribution(result)
    print(report.print_report())


def compare_strategies(strategy_a: str, strategy_b: str, tickers: list):
    print(f"\n{'═'*65}")
    print(f"  Strategy Comparison — {strategy_a} vs {strategy_b}")
    print(f"{'═'*65}")

    from src.backtest.backtest_engine import (
        BacktestEngine, MomentumStrategy, MeanReversionStrategy
    )
    cls_map = {"momentum": MomentumStrategy, "mean_reversion": MeanReversionStrategy}

    engine = BacktestEngine()
    results = {}
    for name in [strategy_a, strategy_b]:
        cls = cls_map.get(name.lower(), MomentumStrategy)
        r   = engine.run(
            strategy=cls(), tickers=tickers,
            start_date=DEFAULT_START, end_date=DEFAULT_END,
            initial_capital=1_000_000, verbose=False,
        )
        results[name] = r

    keys = ["annual_return", "annual_volatility", "sharpe_ratio",
            "sortino_ratio", "max_drawdown", "calmar_ratio",
            "hit_rate", "annual_turnover", "avg_cost_bps", "n_trades"]

    print(f"\n  {'Metric':<24} {strategy_a:>16} {strategy_b:>16}")
    print(f"  {'─'*58}")
    for key in keys:
        a_val = results[strategy_a].compute_metrics().get(key, 0)
        b_val = results[strategy_b].compute_metrics().get(key, 0)
        fmt = ".2%" if "return" in key or "drawdown" in key or "vol" in key or "turnover" in key or "hit" in key else ".3f"
        print(f"  {key:<24} {a_val:>16{fmt}} {b_val:>16{fmt}}")


def run_agent(query: str):
    import os
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  No LLM key — falling back to direct backtest")
        run_backtest("momentum", DEFAULT_TICKERS)
        return

    from src.agents.backtest_agent import BacktestAgent
    agent = BacktestAgent()

    print(f"\n  Query: {query}")
    response_text, _ = agent.think(
        user_message = query,
        use_tools    = True,
        purpose      = "backtest_research",
    )
    print(f"\n  Response:\n{response_text[:1500]}")


def run_demo():
    print(f"\n{'╔'+'═'*58+'╗'}")
    print(f"{'║'+'  PART 6: BACKTESTING ENGINE — FULL DEMO'.center(58)+'║'}")
    print(f"{'╚'+'═'*58+'╝'}")

    tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "JPM", "BAC", "XOM", "JNJ"]

    print("\n[1/4] Running momentum strategy backtest...")
    run_backtest("momentum", tickers)

    print("\n[2/4] Running stress tests...")
    run_stress_test("momentum", tickers[:5])

    print("\n[3/4] Comparing strategies...")
    compare_strategies("momentum", "mean_reversion", tickers[:6])

    print("\n[4/4] Walk-forward validation (4 folds)...")
    run_walk_forward("momentum", tickers[:6])


def main():
    parser = argparse.ArgumentParser(
        description="AI Hedge Fund — Part 6: Backtesting Engine"
    )
    parser.add_argument("--backtest",    nargs="+",
                        help="--backtest <strategy> <ticker1> <ticker2> ...")
    parser.add_argument("--walk-forward",nargs="+",
                        help="--walk-forward <strategy> <ticker1> ...")
    parser.add_argument("--stress-test", nargs="+",
                        help="--stress-test <strategy> <ticker1> ...")
    parser.add_argument("--attribution", nargs="+",
                        help="--attribution <strategy> <ticker1> ...")
    parser.add_argument("--compare",     nargs="+",
                        help="--compare <strategy_a> <strategy_b> <ticker1> ...")
    parser.add_argument("--agent",       type=str,
                        help="Natural language query for BacktestAgent")
    parser.add_argument("--demo",        action="store_true")
    parser.add_argument("--validate",    action="store_true")
    args = parser.parse_args()

    from datetime import datetime
    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + "  AI HEDGE FUND — PART 6: BACKTESTING ENGINE".center(58) + "║")
    print("║" + f"  {datetime.now():%Y-%m-%d %H:%M:%S}".center(58) + "║")
    print("╚" + "═"*58 + "╝")

    validate_environment()

    if args.validate:
        return
    elif args.backtest:
        run_backtest(args.backtest[0], args.backtest[1:])
    elif getattr(args, "walk_forward", None):
        run_walk_forward(args.walk_forward[0], args.walk_forward[1:])
    elif getattr(args, "stress_test", None):
        run_stress_test(args.stress_test[0], args.stress_test[1:])
    elif args.attribution:
        run_attribution(args.attribution[0], args.attribution[1:])
    elif args.compare:
        compare_strategies(args.compare[0], args.compare[1], args.compare[2:])
    elif args.agent:
        run_agent(args.agent)
    elif args.demo:
        run_demo()
    else:
        print("\n  No command — running quick demo")
        run_backtest("momentum", DEFAULT_TICKERS[:6])

    print("\n✅ Part 6 complete.")
    print("   Next: Part 7 — Real-Time Risk Management")


if __name__ == "__main__":
    main()
