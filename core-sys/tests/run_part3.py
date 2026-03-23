"""
AI Hedge Fund — Part 3: RAG & Data Intelligence
=================================================
run_part3.py — Main Entry Point

Demonstrates:
  1. Document ingestion from real sources (SEC EDGAR, Yahoo Finance)
  2. Vector store indexing (ChromaDB or SQLite fallback)
  3. Semantic search over financial documents
  4. LLM-synthesised answers grounded in real documents
  5. Data Intelligence Agent integration with the multi-agent system

Usage:
    # Ingest documents for a ticker:
    python run_part3.py --ingest AAPL

    # Query the knowledge base:
    python run_part3.py --query "Apple revenue guidance" --ticker AAPL

    # Full sentiment analysis:
    python run_part3.py --sentiment AAPL

    # Investment signals extraction:
    python run_part3.py --signals NVDA

    # Earnings call summary:
    python run_part3.py --earnings MSFT

    # Ingest multiple tickers:
    python run_part3.py --ingest-universe AAPL MSFT NVDA GOOGL

    # Full demo (ingest + query + sentiment):
    python run_part3.py --demo AAPL

    # Status of knowledge base:
    python run_part3.py --status
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
    """Check all dependencies for Part 3."""
    print("\n" + "═" * 60)
    print("  Part 3: RAG & Data Intelligence — Environment Check")
    print("═" * 60)

    import os
    has_llm = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"))

    # Core packages
    core_pkgs = [
        ("yfinance",  "pip install yfinance"),
        ("numpy",     "pip install numpy"),
        ("pandas",    "pip install pandas"),
        ("requests",  "pip install requests"),
    ]

    # Part 3 specific packages
    p3_pkgs = [
        ("sentence_transformers", "pip install sentence-transformers"),
        ("chromadb",             "pip install chromadb"),
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

    print("\n  Part 3 packages (recommended):")
    for pkg, install in p3_pkgs:
        try:
            __import__(pkg.replace("-", "_"))
            print(f"    ✓ {pkg}")
        except ImportError:
            print(f"    ○ {pkg} not installed — {install}")
            print(f"      (will use fallback — full capability requires this)")

    print(f"\n  LLM key: {'✓ configured' if has_llm else '✗ not set (queries will return raw chunks)'}")

    # Show knowledge base status
    try:
        from src.rag.rag_engine import RAGEngine
        engine = RAGEngine()
        stats  = engine.store_stats()
        print(f"\n  Knowledge base: {stats['total_chunks']:,} chunks indexed")
        if stats['indexed_tickers']:
            print(f"  Indexed tickers: {', '.join(stats['indexed_tickers'])}")
        else:
            print("  Knowledge base is empty — run --ingest to add documents")
    except Exception as e:
        print(f"\n  Knowledge base: error ({e})")

    return all_ok


def ingest_ticker(ticker: str, refresh: bool = False):
    """Ingest all documents for a ticker."""
    print(f"\n{'═'*60}")
    print(f"  Ingesting Documents — {ticker}")
    print(f"{'═'*60}")

    from src.rag.rag_engine import RAGEngine
    engine = RAGEngine()

    if engine.is_ticker_indexed(ticker) and not refresh:
        count = engine.store.count(ticker)
        print(f"\n  {ticker} already indexed: {count} chunks")
        print("  Use --refresh to force re-ingestion")
        return

    print(f"\n  Fetching documents from:")
    print(f"    • Yahoo Finance news")
    print(f"    • SEC EDGAR filings (10-K, 10-Q, 8-K)")
    print(f"    • Earnings-related 8-K filings\n")

    start = datetime.now()
    stats = engine.ingest_ticker(ticker, refresh=refresh)
    elapsed = (datetime.now() - start).total_seconds()

    print(f"\n  Results:")
    print(f"    Documents fetched: {stats['documents']}")
    print(f"    Chunks indexed:    {stats['chunks']}")
    print(f"    Errors:            {stats.get('errors', 0)}")
    print(f"    Time elapsed:      {elapsed:.1f}s")
    print(f"    Total in store:    {engine.store.count():,}")


def ingest_universe(tickers: list):
    """Ingest documents for multiple tickers."""
    print(f"\n{'═'*60}")
    print(f"  Ingesting Universe — {len(tickers)} tickers")
    print(f"{'═'*60}")

    from src.rag.document_fetchers import DocumentFetchOrchestrator
    from src.rag.rag_engine import RAGEngine

    engine = RAGEngine()
    total  = {"documents": 0, "chunks": 0, "errors": 0}

    for ticker in tickers:
        print(f"\n  [{ticker}]...")
        stats = engine.ingest_ticker(ticker)
        total["documents"] += stats.get("documents", 0)
        total["chunks"]    += stats.get("chunks", 0)
        total["errors"]    += stats.get("errors", 0)
        print(f"    {stats['documents']} docs → {stats['chunks']} chunks")

    print(f"\n  Universe ingestion complete:")
    print(f"    Total documents: {total['documents']}")
    print(f"    Total chunks:    {total['chunks']}")
    print(f"    Errors:          {total['errors']}")
    print(f"    Total in store:  {engine.store.count():,}")


def run_query(query: str, ticker: str = None, use_llm: bool = True):
    """Run a natural language query against the knowledge base."""
    print(f"\n{'═'*60}")
    print(f"  RAG Query")
    print(f"{'═'*60}")
    print(f"\n  Query: {query}")
    if ticker:
        print(f"  Ticker filter: {ticker}")

    from src.rag.rag_engine import RAGEngine
    engine = RAGEngine()

    if ticker and not engine.is_ticker_indexed(ticker):
        print(f"\n  {ticker} not indexed — ingesting now...")
        engine.ingest_ticker(ticker)

    result = engine.query(
        query_text    = query,
        ticker_filter = [ticker] if ticker else None,
        use_llm       = use_llm,
        top_k         = 5,
    )

    print(f"\n  Retrieved {result.num_sources} relevant chunks")
    print(f"  {'═'*50}")

    if not use_llm or not result.synthesis:
        print("\n  RAW RETRIEVED CHUNKS:")
        for r in result.retrieved:
            dt = r.chunk.published_at or r.chunk.filing_date
            dt_str = dt.strftime("%Y-%m-%d") if dt else "Unknown"
            print(f"\n  [{r.rank}] Similarity: {r.similarity:.3f}")
            print(f"       {r.chunk.ticker} | {r.chunk.doc_type.value} | {dt_str}")
            print(f"       {r.chunk.title}")
            print(f"       {r.chunk.text[:300]}...")
    else:
        print(f"\n  ANSWER (LLM synthesis from {result.num_sources} sources):")
        print(f"  {'-'*50}")
        print(result.synthesis)
        print(f"\n  Cost: ${result.cost_usd:.5f} | Latency: {result.latency_ms:.0f}ms")

    # Save result
    output_path = (
        Path(__file__).parent / "logs" /
        f"rag_query_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"\n  Saved to: {output_path}")


def run_sentiment(ticker: str):
    """Run sentiment analysis for a ticker."""
    print(f"\n{'═'*60}")
    print(f"  Sentiment Analysis — {ticker}")
    print(f"{'═'*60}")

    from src.rag.rag_engine import RAGEngine
    engine = RAGEngine()

    if not engine.is_ticker_indexed(ticker):
        print(f"  {ticker} not indexed — ingesting now...")
        engine.ingest_ticker(ticker)

    print(f"\n  Analysing sentiment in {engine.store.count(ticker)} chunks...")
    result = engine.analyse_sentiment(ticker)

    if result.synthesis:
        print(f"\n  SENTIMENT ANALYSIS:")
        print(f"  {'-'*50}")
        print(result.synthesis)
        print(f"\n  Sources: {result.num_sources} documents")
        print(f"  Cost: ${result.cost_usd:.5f}")
    else:
        print("  No sentiment data available")


def run_signals(ticker: str):
    """Extract investment signals for a ticker."""
    print(f"\n{'═'*60}")
    print(f"  Investment Signal Extraction — {ticker}")
    print(f"{'═'*60}")

    from src.rag.rag_engine import RAGEngine
    engine = RAGEngine()

    if not engine.is_ticker_indexed(ticker):
        print(f"  {ticker} not indexed — ingesting now...")
        engine.ingest_ticker(ticker)

    print(f"  Extracting investment signals...")
    result = engine.extract_investment_signals(ticker)

    if result.synthesis:
        print(f"\n  INVESTMENT SIGNALS:")
        print(f"  {'-'*50}")
        print(result.synthesis)
        print(f"\n  Cost: ${result.cost_usd:.5f}")
    else:
        print("  No signals extracted")


def run_earnings(ticker: str):
    """Summarise earnings call for a ticker."""
    print(f"\n{'═'*60}")
    print(f"  Earnings Call Summary — {ticker}")
    print(f"{'═'*60}")

    from src.rag.rag_engine import RAGEngine
    engine = RAGEngine()

    if not engine.is_ticker_indexed(ticker):
        print(f"  {ticker} not indexed — ingesting now...")
        engine.ingest_ticker(ticker)

    print(f"  Summarising earnings call...")
    result = engine.summarise_earnings_call(ticker)

    if result.synthesis:
        print(f"\n  EARNINGS SUMMARY:")
        print(f"  {'-'*50}")
        print(result.synthesis)
        print(f"\n  Cost: ${result.cost_usd:.5f}")
    else:
        print("  No earnings data found")


def run_demo(ticker: str):
    """Full demonstration: ingest + query + sentiment + signals."""
    print(f"\n{'╔'+'═'*58+'╗'}")
    print(f"{'║'+'  FULL DEMO — ' + ticker:{'║'}<60}{'║'}")
    print(f"{'╚'+'═'*58+'╝'}")

    import os
    has_llm = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"))

    from src.rag.rag_engine import RAGEngine
    engine = RAGEngine()

    # Step 1: Ingest
    print(f"\nStep 1/4: Ingesting documents for {ticker}...")
    stats = engine.ingest_ticker(ticker)
    print(f"  → {stats['chunks']} chunks indexed")

    if engine.store.count(ticker) == 0:
        print(f"  ✗ No documents found for {ticker}. Check internet connection.")
        return

    # Step 2: Simple retrieval (no LLM)
    print(f"\nStep 2/4: Semantic retrieval (no LLM)...")
    result = engine.query(
        f"{ticker} revenue earnings growth",
        ticker_filter = [ticker],
        use_llm       = False,
        top_k         = 3,
    )
    print(f"  → Found {result.num_sources} relevant chunks")
    if result.retrieved:
        r = result.retrieved[0]
        print(f"  → Top result (sim={r.similarity:.2f}): {r.chunk.text[:200]}...")

    if not has_llm:
        print(f"\n  (Add LLM API key for Steps 3-4 with synthesis)")
        return

    # Step 3: Sentiment
    print(f"\nStep 3/4: Sentiment analysis...")
    sentiment = engine.analyse_sentiment(ticker)
    if sentiment.synthesis:
        preview = sentiment.synthesis[:400]
        print(f"  {preview}...")
        print(f"  Cost: ${sentiment.cost_usd:.5f}")

    # Step 4: Investment signals
    print(f"\nStep 4/4: Investment signal extraction...")
    signals = engine.extract_investment_signals(ticker)
    if signals.synthesis:
        preview = signals.synthesis[:400]
        print(f"  {preview}...")
        print(f"  Cost: ${signals.cost_usd:.5f}")

    print(f"\n✅ Demo complete. Total chunks: {engine.store.count():,}")


def show_status():
    """Show knowledge base status."""
    print(f"\n{'═'*60}")
    print("  Knowledge Base Status")
    print(f"{'═'*60}")

    from src.rag.rag_engine import RAGEngine
    engine = RAGEngine()
    stats  = engine.store_stats()

    print(f"\n  Total chunks:    {stats['total_chunks']:,}")
    print(f"  Embedding model: {stats['embedding_model']}")
    print(f"  Embedding dims:  {stats['embedding_dims']}")
    print(f"\n  Indexed tickers ({len(stats['indexed_tickers'])}):")

    if stats['indexed_tickers']:
        for ticker in stats['indexed_tickers']:
            count = engine.store.count(ticker)
            print(f"    {ticker:<8} {count:>5} chunks")
    else:
        print("    (none — run --ingest to add documents)")


def main():
    parser = argparse.ArgumentParser(
        description="AI Hedge Fund — Part 3: RAG & Data Intelligence"
    )
    parser.add_argument("--ingest",          metavar="TICKER",   help="Ingest documents for ticker")
    parser.add_argument("--ingest-universe", nargs="+",          help="Ingest multiple tickers")
    parser.add_argument("--refresh",         action="store_true",help="Force re-ingestion")
    parser.add_argument("--query",           metavar="TEXT",     help="Natural language query")
    parser.add_argument("--ticker",          metavar="TICKER",   help="Filter query by ticker")
    parser.add_argument("--no-llm",          action="store_true",help="Return raw chunks (no LLM)")
    parser.add_argument("--sentiment",       metavar="TICKER",   help="Sentiment analysis")
    parser.add_argument("--signals",         metavar="TICKER",   help="Investment signals")
    parser.add_argument("--earnings",        metavar="TICKER",   help="Earnings summary")
    parser.add_argument("--demo",            metavar="TICKER",   help="Full demo")
    parser.add_argument("--status",          action="store_true",help="Show KB status")
    parser.add_argument("--validate",        action="store_true",help="Validate environment")
    args = parser.parse_args()

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + "  AI HEDGE FUND — PART 3: RAG & DATA INTELLIGENCE".center(58) + "║")
    print("║" + f"  {datetime.now():%Y-%m-%d %H:%M:%S}".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    if args.validate or args.status:
        validate_environment() if args.validate else None
        show_status() if args.status else None
        return

    validate_environment()

    if args.ingest:
        ingest_ticker(args.ingest, refresh=args.refresh)
    elif args.ingest_universe:
        ingest_universe(args.ingest_universe)
    elif args.query:
        run_query(args.query, ticker=args.ticker, use_llm=not args.no_llm)
    elif args.sentiment:
        run_sentiment(args.sentiment)
    elif args.signals:
        run_signals(args.signals)
    elif args.earnings:
        run_earnings(args.earnings)
    elif args.demo:
        run_demo(args.demo)
    else:
        # Default: demo with AAPL
        print("\n  No command specified — running demo with AAPL")
        print("  Run 'python run_part3.py --help' for options\n")
        run_demo("AAPL")

    print("\n✅ Part 3 complete.")
    print("   Next: Part 4 — Execution Engine (Almgren-Chriss + IB integration)")


if __name__ == "__main__":
    main()
