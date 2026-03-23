"""
AI Hedge Fund — Part 5: Alternative Assets & Data
====================================================
run_part5.py — Main Entry Point

Demonstrates:
  1. Cat bond pricing (Almgren-Chriss frequency-severity model)
  2. ILS portfolio construction and diversification analysis
  3. Alternative data signals (options, insiders, short interest, analysts)
  4. Alt Assets Agent integration with the multi-agent system

Usage:
    # Price a cat bond:
    python run_part5.py --price-bond PELICAN-2024-A

    # Screen all cat bonds:
    python run_part5.py --screen-ils

    # ILS portfolio report:
    python run_part5.py --ils-portfolio

    # Diversification benefit analysis:
    python run_part5.py --diversification

    # Alternative data for a ticker:
    python run_part5.py --alt-data AAPL

    # Rank universe by alt data:
    python run_part5.py --rank-universe AAPL MSFT NVDA GOOGL JPM

    # Full demo:
    python run_part5.py --demo

    # Run Alt Assets Agent (requires LLM key):
    python run_part5.py --agent AAPL
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
    print("\n" + "═" * 60)
    print("  Part 5: Alternative Assets & Data — Environment Check")
    print("═" * 60)

    import os
    has_llm = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"))

    core_pkgs = [
        ("numpy",   "pip install numpy"),
        ("pandas",  "pip install pandas"),
        ("scipy",   "pip install scipy"),
        ("yfinance","pip install yfinance"),
    ]
    optional = [
        ("requests","pip install requests"),
    ]

    all_ok = True
    print("\n  Core packages:")
    for pkg, install in core_pkgs:
        try:
            __import__(pkg)
            print(f"    ✓ {pkg}")
        except ImportError:
            print(f"    ✗ {pkg} — {install}")
            all_ok = False

    print(f"\n  LLM key: {'✓' if has_llm else '✗ (optional — needed for agent mode)'}")
    return all_ok


def price_bond(bond_ticker: str):
    """Price a specific cat bond."""
    print(f"\n{'═'*65}")
    print(f"  Cat Bond Pricing — {bond_ticker}")
    print(f"{'═'*65}")

    from src.catbond.cat_bond_models import (
        CatBondPricer, build_standard_loss_models, create_example_cat_bonds
    )
    from src.catbond.ils_portfolio import ILSPortfolio

    bonds   = {b.ticker: b for b in create_example_cat_bonds()}
    pricer  = CatBondPricer()
    models  = build_standard_loss_models()
    portfolio = ILSPortfolio()

    if bond_ticker not in bonds:
        print(f"  Bond not found. Available: {list(bonds.keys())}")
        return

    bond   = bonds[bond_ticker]
    mk     = portfolio._infer_model_key(bond)
    model  = models[mk]

    print(f"\n  Running Monte Carlo (100K scenarios)...")
    result = pricer.price(bond, model)
    print(f"\n{result.summary()}")

    print(f"\n  Loss Exceedance Curve:")
    oep = result.oep_curve
    for rp, loss in list(oep.items())[:5]:
        print(f"    {rp:<30}: {loss:.3f}")


def screen_ils():
    """Screen the full cat bond universe."""
    print(f"\n{'═'*65}")
    print("  Cat Bond Universe Screen")
    print(f"{'═'*65}")

    from src.catbond.cat_bond_models import (
        CatBondPricer, build_standard_loss_models, create_example_cat_bonds
    )
    from src.catbond.ils_portfolio import ILSPortfolio

    bonds   = create_example_cat_bonds()
    pricer  = CatBondPricer()
    models  = build_standard_loss_models()
    portfolio = ILSPortfolio()

    print(f"\n  Pricing {len(bonds)} bonds...\n")
    print(f"  {'Ticker':<18} {'Peril':<14} {'EL':>6} {'Spread':>8} {'Fair':>8} "
          f"{'Diff':>8} {'Multiple':>8} {'Rating':<6}")
    print(f"  {'─'*75}")

    results = []
    for bond in bonds:
        mk     = portfolio._infer_model_key(bond)
        model  = models[mk]
        result = pricer.price(bond, model)
        results.append((bond, result))
        val_str = f"+{result.spread_vs_fair:.0f}" if result.is_cheap else f"{result.spread_vs_fair:.0f}"
        print(
            f"  {bond.ticker:<18} {bond.peril.value:<14} "
            f"{result.el_bps:>5.1f}bp "
            f"{bond.coupon_spread:>7}bp "
            f"{result.fair_spread_bps:>7.0f}bp "
            f"{val_str:>8}bp "
            f"{result.risk_multiple:>7.1f}x "
            f"{bond.rating:<6}"
        )

    print(f"\n  Best value: {max(results, key=lambda x: x[1].spread_vs_fair)[0].ticker}")
    print(f"  Worst value: {min(results, key=lambda x: x[1].spread_vs_fair)[0].ticker}")


def show_ils_portfolio():
    """Show ILS portfolio with example allocations."""
    print(f"\n{'═'*65}")
    print("  ILS Portfolio Analysis")
    print(f"{'═'*65}")

    from src.catbond.cat_bond_models import create_example_cat_bonds
    from src.catbond.ils_portfolio import ILSPortfolio

    portfolio = ILSPortfolio(total_nav=10_000_000, ils_allocation=0.10)
    bonds = create_example_cat_bonds()

    allocations = {
        "PELICAN-2024-A": 400_000,
        "SIERRA-2024-A":  300_000,
        "BORA-2024-A":    200_000,
        "ATLAS-2024-A":   100_000,
    }
    for bond in bonds:
        portfolio.add_position(bond, allocations.get(bond.ticker, 100_000))

    print(portfolio.portfolio_report())

    import numpy as np
    print("\n  Running correlated loss simulation (50K scenarios)...")
    losses = portfolio.simulate_portfolio_loss(50_000)
    print(f"  Mean annual loss:    ${np.mean(losses):>10,.0f}")
    print(f"  Std annual loss:     ${np.std(losses):>10,.0f}")
    print(f"  VaR 99%:             ${np.percentile(losses, 99):>10,.0f}")
    print(f"  CVaR 99%:            ${np.mean(losses[losses >= np.percentile(losses, 99)]):>10,.0f}")
    print(f"  Portfolio Sharpe:    {portfolio.portfolio_sharpe():>10.3f}")


def show_diversification():
    """Show portfolio-level diversification benefit of ILS."""
    print(f"\n{'═'*65}")
    print("  ILS Diversification Benefit Analysis")
    print(f"{'═'*65}")

    from src.catbond.cat_bond_models import create_example_cat_bonds
    from src.catbond.ils_portfolio import ILSPortfolio

    portfolio = ILSPortfolio(total_nav=10_000_000, ils_allocation=0.10)
    bonds = create_example_cat_bonds()
    portfolio.add_position(bonds[0], 500_000)
    portfolio.add_position(bonds[2], 300_000)
    portfolio.add_position(bonds[3], 200_000)

    print(f"\n  ILS allocation: 10% of NAV")
    print(f"  Testing against different equity allocations:\n")

    print(f"  {'Equity%':>8} {'Bonds%':>7} {'ILS%':>6} | "
          f"{'No-ILS Sharpe':>14} {'With-ILS Sharpe':>15} {'Improvement':>12} | "
          f"{'Vol Δ':>8}")
    print(f"  {'─'*80}")

    for eq_pct in [40, 50, 60, 70, 80]:
        benefit = portfolio.equity_diversification_benefit(
            equity_weight = eq_pct / 100,
            equity_vol    = 0.16,
        )
        bond_pct = 90 - eq_pct
        print(
            f"  {eq_pct:>7}% {bond_pct:>6}% {benefit['ils_allocation_pct']:>5.0f}% | "
            f"{benefit['pure_equity_sharpe']:>14.3f} "
            f"{benefit['blended_sharpe']:>15.3f} "
            f"{benefit['sharpe_improvement']:>+12.3f} | "
            f"{-benefit['vol_reduction_pct']:>+7.1f}%"
        )

    print(f"\n  ILS-equity correlation: "
          f"{portfolio.corr_matrix.equity_ils_correlation('SPY'):.2f} "
          f"(source: Swiss Re ILS data 2002-2023)")


def show_alt_data(ticker: str):
    """Show alternative data signals for a ticker."""
    print(f"\n{'═'*65}")
    print(f"  Alternative Data Signals — {ticker}")
    print(f"{'═'*65}")

    from src.altdata.alternative_data import AlternativeDataEngine
    engine = AlternativeDataEngine()

    print(f"\n  Fetching signals for {ticker}...")
    bundle = engine.get_signals(ticker)

    print(f"\n{bundle.summary()}")
    print(f"\n  Composite: {bundle.composite_signal:+.3f} "
          f"({'BULLISH' if bundle.composite_signal > 0.2 else 'BEARISH' if bundle.composite_signal < -0.2 else 'NEUTRAL'}) | "
          f"Confidence: {bundle.composite_confidence:.0%}")


def rank_universe(tickers: list):
    """Rank tickers by composite alt data signal."""
    print(f"\n{'═'*65}")
    print(f"  Universe Alt Data Ranking — {len(tickers)} tickers")
    print(f"{'═'*65}")

    from src.altdata.alternative_data import AlternativeDataEngine
    engine = AlternativeDataEngine()

    print(f"\n  Fetching signals for {len(tickers)} tickers...\n")
    print(f"  {'Ticker':<8} {'Signal':>8} {'Conf':>6} {'Direction':<12} Summary")
    print(f"  {'─'*65}")

    results = []
    for ticker in tickers:
        bundle = engine.get_signals(ticker)
        direction = "BULLISH" if bundle.composite_signal > 0.2 else "BEARISH" if bundle.composite_signal < -0.2 else "NEUTRAL"
        results.append((ticker, bundle.composite_signal, bundle.composite_confidence, direction))

    results.sort(key=lambda x: x[1] * x[2], reverse=True)
    for t, sig, conf, direction in results:
        print(f"  {t:<8} {sig:>+8.3f} {conf:>5.0%} {direction:<12}")


def run_full_demo():
    """Full Part 5 demonstration."""
    print(f"\n{'╔'+'═'*58+'╗'}")
    print(f"{'║'+'  PART 5: ALTERNATIVE ASSETS & DATA DEMO'.center(58)+'║'}")
    print(f"{'╚'+'═'*58+'╝'}")

    screen_ils()
    show_ils_portfolio()
    show_diversification()
    show_alt_data("AAPL")


def run_agent(ticker: str):
    """Run full Alt Assets Agent with LLM."""
    import os
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  No LLM key — running component tests")
        show_alt_data(ticker)
        price_bond("PELICAN-2024-A")
        return

    from src.agents.alt_assets_agent import AltAssetsAgent
    agent = AltAssetsAgent()

    print(f"\n1. Alt data enrichment for {ticker}...")
    result = agent.enrich_equity_decision(ticker)
    print(f"   Signal: {result['composite_signal']:+.3f} | Conf: {result['confidence']:.0%}")
    print(f"\n{result['summary']}")

    print(f"\n2. ILS universe evaluation...")
    ils_result = agent.evaluate_ils_allocation()
    print(f"   Recommendation: {ils_result.get('recommendation', 'N/A')}")
    if "summary" in ils_result:
        print(f"   {ils_result['summary'][:300]}")


def main():
    parser = argparse.ArgumentParser(
        description="AI Hedge Fund — Part 5: Alternative Assets & Data"
    )
    parser.add_argument("--price-bond",    metavar="TICKER", help="Price a cat bond")
    parser.add_argument("--screen-ils",    action="store_true", help="Screen ILS universe")
    parser.add_argument("--ils-portfolio", action="store_true", help="ILS portfolio report")
    parser.add_argument("--diversification", action="store_true", help="Diversification analysis")
    parser.add_argument("--alt-data",      metavar="TICKER", help="Alt data for equity")
    parser.add_argument("--rank-universe", nargs="+", help="Rank tickers by alt data")
    parser.add_argument("--demo",          action="store_true", help="Full demo")
    parser.add_argument("--agent",         metavar="TICKER", help="Run Alt Assets Agent")
    parser.add_argument("--validate",      action="store_true", help="Validate environment")
    args = parser.parse_args()

    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + "  AI HEDGE FUND — PART 5: ALT ASSETS & DATA".center(58) + "║")
    print("║" + f"  {datetime.now():%Y-%m-%d %H:%M:%S}".center(58) + "║")
    print("╚" + "═"*58 + "╝")

    validate_environment()

    if args.validate:         return
    if args.price_bond:       price_bond(args.price_bond)
    elif args.screen_ils:     screen_ils()
    elif args.ils_portfolio:  show_ils_portfolio()
    elif args.diversification:show_diversification()
    elif args.alt_data:       show_alt_data(args.alt_data)
    elif args.rank_universe:  rank_universe(args.rank_universe)
    elif args.agent:          run_agent(args.agent)
    elif args.demo:           run_full_demo()
    else:
        print("\n  No command specified — running full demo")
        run_full_demo()

    print("\n✅ Part 5 complete.")
    print("   Next: Part 6 — Backtesting Engine")


if __name__ == "__main__":
    main()
