"""
AI Hedge Fund — Part 1: Foundation
====================================
run_part1.py — Main entry point

Run this script to:
  1. Validate your environment (API keys, dependencies)
  2. Test market data fetching (real OHLCV from Yahoo Finance)
  3. Compute signals for a candidate security
  4. Run the Portfolio Manager Agent (requires LLM API key)
  5. Print a full decision report

Usage:
    # Full run (requires LLM API key in .env):
    python run_part1.py

    # Data-only mode (no API key needed):
    python run_part1.py --data-only

    # Analyse a specific ticker:
    python run_part1.py --ticker NVDA

    # Analyse the full default universe:
    python run_part1.py --universe --top 5

    # Cost estimate only:
    python run_part1.py --cost-check
"""

import argparse
import logging
import sys
import json
from datetime import datetime
from pathlib import Path

# Ensure src/ is in the Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import cfg, setup_logging, write_env_template
from src.data.data_models import Portfolio, Position, Direction, Conviction
from src.data.market_data import (
    MarketDataFetcher, FeatureEngineer, CorrelationEngine, MacroDataFetcher
)

logger = setup_logging()


# ─────────────────────────────────────────────────────────────────────────────
# Environment validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_environment() -> bool:
    """
    Check all dependencies and configuration.
    Returns True if ready for full operation, False if partial only.
    """
    print("\n" + "═" * 60)
    print("  Environment Validation")
    print("═" * 60)

    all_ok = True

    # ── Python packages ──────────────────────────────────────────────────
    required = [("yfinance", "pip install yfinance"),
                ("numpy",    "pip install numpy"),
                ("pandas",   "pip install pandas"),
                ("scipy",    "pip install scipy")]

    optional = [("anthropic", "pip install anthropic"),
                ("openai",    "pip install openai")]

    print("\n  Core packages:")
    for pkg, install in required:
        try:
            __import__(pkg)
            print(f"    ✓ {pkg}")
        except ImportError:
            print(f"    ✗ {pkg} — Install: {install}")
            all_ok = False

    print("\n  LLM packages (at least one required for agent):")
    llm_ok = False
    for pkg, install in optional:
        try:
            __import__(pkg)
            print(f"    ✓ {pkg}")
            llm_ok = True
        except ImportError:
            print(f"    ○ {pkg} not installed — {install}")

    if not llm_ok:
        print("    ⚠  No LLM SDK installed. Agent will not run.")

    # ── API Keys ─────────────────────────────────────────────────────────
    print("\n  API Keys:")
    print(f"    {'✓' if cfg.ANTHROPIC_API_KEY else '✗'} ANTHROPIC_API_KEY {'(set)' if cfg.ANTHROPIC_API_KEY else '(not set)'}")
    print(f"    {'✓' if cfg.OPENAI_API_KEY else '✗'} OPENAI_API_KEY {'(set)' if cfg.OPENAI_API_KEY else '(not set)'}")
    print(f"    {'✓' if cfg.FRED_API_KEY else '○'} FRED_API_KEY {'(set — macro data enabled)' if cfg.FRED_API_KEY else '(optional — will use market proxies)'}")
    print(f"    {'✓' if cfg.POLYGON_API_KEY else '○'} POLYGON_API_KEY {'(set)' if cfg.POLYGON_API_KEY else '(optional — using Yahoo Finance)'}")

    agent_ready = llm_ok and (cfg.ANTHROPIC_API_KEY or cfg.OPENAI_API_KEY)

    # ── Portfolio config ──────────────────────────────────────────────────
    print(f"\n  {cfg.portfolio_summary()}")

    # ── .env file ─────────────────────────────────────────────────────────
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        write_env_template()
        print(f"\n  ⚠  Created .env.example — rename to .env and add your API keys")
        all_ok = False
    else:
        print(f"\n  ✓ .env file found")

    print()
    return agent_ready


# ─────────────────────────────────────────────────────────────────────────────
# Test market data
# ─────────────────────────────────────────────────────────────────────────────

def test_market_data(tickers=None) -> bool:
    """Fetch real market data and compute features for a set of tickers."""
    test_tickers = tickers or ["AAPL", "MSFT", "GOOGL", "JPM", "SPY"]

    print("\n" + "═" * 60)
    print("  Market Data Test")
    print("═" * 60)

    fetcher  = MarketDataFetcher()
    engineer = FeatureEngineer()

    try:
        for ticker in test_tickers[:3]:
            print(f"\n  Fetching {ticker}...")
            df = fetcher.get_prices(ticker, days=252)
            if df.empty:
                print(f"    ✗ No data returned for {ticker}")
                return False

            latest = df["Close"].iloc[-1]
            ret_1d = df["Close"].pct_change().iloc[-1]
            ret_21d = df["Close"].pct_change(21).iloc[-1]
            vol = df["Close"].pct_change().std() * (252 ** 0.5)

            print(f"    ✓ {len(df)} bars | Price: ${latest:.2f} | "
                  f"1d: {ret_1d:+.2%} | 21d: {ret_21d:+.2%} | "
                  f"Ann.Vol: {vol:.1%}")

            # Feature computation
            features = engineer.compute_all_features(df, ticker)
            if features.empty:
                print(f"    ⚠ Feature computation returned empty DataFrame")
            else:
                non_null = features.count().sum() / (features.shape[0] * features.shape[1])
                print(f"    ✓ Features: {features.shape[1]} cols × {features.shape[0]} rows | "
                      f"Completeness: {non_null:.0%}")

        # Market snapshot
        print(f"\n  Building market snapshot ({len(test_tickers)} tickers)...")
        snapshot = fetcher.get_market_snapshot(test_tickers)
        print(f"  ✓ {snapshot.market_summary()}")

        if snapshot.top_movers(3):
            print(f"  Top movers: {[(t, f'{r:+.2%}') for t, r in snapshot.top_movers(3)]}")

        # Correlation
        print(f"\n  Computing correlations...")
        corr_engine = CorrelationEngine(fetcher)
        corr = corr_engine.average_correlation("AAPL", ["MSFT", "GOOGL", "JPM"])
        print(f"  ✓ AAPL avg correlation to MSFT/GOOGL/JPM: {corr:.3f}")

        # Macro
        print(f"\n  Building macro context...")
        macro = MacroDataFetcher()
        ctx = macro.get_macro_context()
        print(f"  ✓ {ctx.describe()}")

        print("\n  ✅ Market data tests passed")
        return True

    except Exception as e:
        print(f"\n  ✗ Market data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Run PM Agent
# ─────────────────────────────────────────────────────────────────────────────

def run_pm_agent(ticker: str = "AAPL") -> None:
    """Run the Portfolio Manager Agent for a single ticker."""

    print("\n" + "═" * 60)
    print(f"  Portfolio Manager Agent — {ticker}")
    print("═" * 60)

    from src.agents.portfolio_manager_agent import PortfolioManagerAgent

    # Create sample portfolio (in production: load from database)
    portfolio = Portfolio(
        portfolio_id    = "FUND_001",
        cash            = cfg.INITIAL_CAPITAL * 0.85,   # 85% cash (early stage)
        initial_capital = cfg.INITIAL_CAPITAL,
        inception_date  = datetime.now().date(),
    )

    # Add one existing position to make correlation checks interesting
    portfolio.positions["MSFT"] = Position(
        ticker        = "MSFT",
        direction     = Direction.LONG,
        shares        = 300,
        avg_cost      = 380.0,
        current_price = 415.0,
        sector        = "Technology",
        entry_date    = datetime.now().date(),
    )

    print(f"\n  Portfolio: {portfolio.portfolio_id}")
    print(f"  NAV: ${portfolio.net_asset_value:,.0f}")
    print(f"  Positions: {list(portfolio.positions.keys())}")

    # Initialise agent
    print(f"\n  Initialising PM Agent ({cfg.DEFAULT_LLM_MODEL})...")
    pm = PortfolioManagerAgent(portfolio)

    # Make decision
    print(f"  Analysing {ticker}...\n")
    decision = pm.make_decision(ticker)

    # Print report
    print(pm.decision_report(decision))

    # Save decision to JSON
    output_path = Path(__file__).parent / "logs" / f"decision_{ticker}_{datetime.now():%Y%m%d_%H%M%S}.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(decision.to_dict(), f, indent=2)
    print(f"\n  Decision saved to: {output_path}")

    # Cost summary
    print(f"\n  {pm.llm.get_spend_summary()}")


def run_universe_analysis(top_n: int = 5) -> None:
    """Analyse the full default universe."""
    print("\n" + "═" * 60)
    print(f"  Universe Analysis — Top {top_n} Opportunities")
    print("═" * 60)

    from src.agents.portfolio_manager_agent import PortfolioManagerAgent

    portfolio = Portfolio(
        portfolio_id    = "FUND_001",
        cash            = cfg.INITIAL_CAPITAL,
        initial_capital = cfg.INITIAL_CAPITAL,
    )

    pm = PortfolioManagerAgent(portfolio)

    print(f"\n  Analysing {len(cfg.DEFAULT_UNIVERSE)} securities...")
    print("  (This will take several minutes and cost ~$0.50-$2.00 in LLM API calls)")
    print("  Press Ctrl+C to cancel\n")

    decisions = pm.analyse_universe(top_n=top_n)

    print(f"\n  Top {top_n} Opportunities:")
    print("  " + "─" * 60)

    for i, d in enumerate(decisions, 1):
        flag = "🟢" if d.recommendation == "BUY" else "🔴" if d.recommendation == "SELL" else "⚪"
        print(f"  {i}. {flag} {d.ticker:<6} | {d.recommendation:<4} | "
              f"Target: {d.target_weight:.1%} | "
              f"Conviction: {d.conviction.value}")
        if d.key_factors:
            print(f"       → {d.key_factors[0]}")

    print(f"\n  {pm.llm.get_spend_summary()}")


def cost_estimate() -> None:
    """Print estimated costs for running the system."""
    print("\n" + "═" * 60)
    print("  Cost Estimate")
    print("═" * 60)
    print(f"\n  Model: {cfg.DEFAULT_LLM_MODEL}")
    print(f"  Estimated tokens per decision: ~3,000 (in) + ~500 (out)")
    print()

    from src.agents.llm_client import MODEL_COSTS, estimate_cost
    model = cfg.DEFAULT_LLM_MODEL
    in_tok, out_tok = 3000, 500

    cost_per_decision = estimate_cost(model, in_tok, out_tok)
    print(f"  Cost per decision        : ${cost_per_decision:.5f}")
    print(f"  Cost for 10 decisions/day: ${cost_per_decision*10:.4f}")
    print(f"  Cost per month (10/day)  : ${cost_per_decision*10*30:.2f}")
    print(f"  Universe scan (30 stocks): ${cost_per_decision*30:.3f}")

    print()
    print("  Compare to:")
    print("    Bloomberg Terminal  : $25,000/year = $68.49/day")
    print("    Quant PM salary     : $300,000/year = $821/day")
    print(f"    This system at 10/day : ${cost_per_decision*10*365:.2f}/year")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AI Hedge Fund — Part 1: Foundation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_part1.py                    # Full run (AAPL, requires API key)
  python run_part1.py --ticker NVDA      # Analyse NVDA
  python run_part1.py --data-only        # Test data only (no API key needed)
  python run_part1.py --universe --top 5 # Analyse universe, return top 5
  python run_part1.py --cost-check       # Show cost estimate
        """
    )
    parser.add_argument("--ticker",     default="AAPL", help="Ticker to analyse")
    parser.add_argument("--data-only",  action="store_true", help="Run data tests only (no LLM)")
    parser.add_argument("--universe",   action="store_true", help="Analyse full universe")
    parser.add_argument("--top",        type=int, default=5, help="Top N for universe mode")
    parser.add_argument("--cost-check", action="store_true", help="Print cost estimate")
    parser.add_argument("--validate",   action="store_true", help="Validate environment only")
    args = parser.parse_args()

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + "  AI HEDGE FUND — PART 1: FOUNDATION".center(58) + "║")
    print("║" + f"  {datetime.now():%Y-%m-%d %H:%M:%S}".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    # Always validate environment first
    agent_ready = validate_environment()

    if args.validate:
        sys.exit(0 if agent_ready else 1)

    if args.cost_check:
        cost_estimate()
        sys.exit(0)

    # Market data always runs (no API key needed)
    data_ok = test_market_data()
    if not data_ok:
        print("\n✗ Market data failed. Check your internet connection.")
        sys.exit(1)

    if args.data_only:
        print("\n✅ Data-only mode complete. Add LLM API key to .env to run agents.")
        sys.exit(0)

    if not agent_ready:
        print("\n⚠️  LLM agent not available (no API key configured).")
        print("   Add ANTHROPIC_API_KEY or OPENAI_API_KEY to your .env file.")
        print("   Market data and signals work without an API key.\n")
        sys.exit(0)

    if args.universe:
        run_universe_analysis(top_n=args.top)
    else:
        run_pm_agent(ticker=args.ticker)

    print("\n✅ Part 1 complete.")
    print("   Next: Part 2 — Multi-Agent System (Risk Agent + Research Analyst)")
    print()


if __name__ == "__main__":
    main()
