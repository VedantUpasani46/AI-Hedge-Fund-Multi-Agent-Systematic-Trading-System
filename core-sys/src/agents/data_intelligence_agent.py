"""
AI Hedge Fund — Part 3: RAG & Data Intelligence
=================================================
data_intelligence_agent.py — RAG-Powered Data Intelligence Agent

The Data Intelligence Agent is the hedge fund's research librarian.
It gives every other agent access to the full document knowledge base.

Role in the multi-agent system:
    Part 1: PM Agent makes decisions based on price signals
    Part 2: Risk/Research agents add risk checks and thesis
    Part 3: Data Agent adds real document intelligence

    Before Part 3:
        PM Agent → "Based on momentum and GARCH vol..."
        Research Analyst → "Based on price action..."

    After Part 3:
        PM Agent → "Based on momentum AND Apple's Q4 transcript
                    where Tim Cook said 'AI is our biggest opportunity'..."
        Research Analyst → "Based on price action AND Apple's 10-K
                           which lists competition from Google as a key risk..."

    The document grounding transforms the LLM from a guesser
    into a reader of real source material.

Communication (via MessageBus from Part 2):
    Receives:
        document_query        — Any agent asking a question about documents
        ingest_ticker         — Request to fetch and index documents
        sentiment_analysis    — Request sentiment for a ticker
        investment_signals    — Request investment signal extraction
        earnings_summary      — Request earnings call summary
        risk_factors          — Request key risks from filings
        macro_context         — Request Fed/macro analysis

    Sends:
        document_query_result — Answer with source citations
        ingestion_complete    — Notification that indexing is done
        sentiment_result      — Structured sentiment analysis
        investment_signals_result — Structured signal extraction

Integration with PM Agent:
    The PM Agent can now call ask_agent("DataIntelligence", "sentiment_analysis")
    before making an allocation decision, getting real document-grounded context.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hedge_fund.data_intelligence")


@dataclass
class DocumentQueryRequest:
    """Structured request to the Data Intelligence Agent."""
    query_type:    str              # query / sentiment / signals / earnings / risks / macro
    query_text:    str
    ticker:        Optional[str] = None
    tickers:       Optional[List[str]] = None
    use_llm:       bool = True
    top_k:         int = 5


class DataIntelligenceAgent(BaseAgent):
    """
    RAG-powered Data Intelligence Agent.

    Provides every other agent with access to the document
    knowledge base via natural language queries.

    Tools:
        query_documents       — Semantic search over all documents
        ingest_ticker_docs    — Fetch and index documents for a ticker
        analyse_sentiment     — Sentiment analysis from real documents
        extract_signals       — Investment signal extraction
        summarise_earnings    — Earnings call summary
        get_risk_factors      — Risk factor extraction
        get_guidance          — Forward guidance extraction
        get_macro_context     — Fed minutes / macro reports
        compare_companies     — Cross-ticker comparison
        check_indexing_status — Which tickers are indexed
    """

    SYSTEM_PROMPT = """You are a financial data intelligence specialist at a hedge fund.

YOUR ROLE:
You have access to a comprehensive database of financial documents:
  - SEC 10-K/10-Q/8-K filings (audited financial reports)
  - Earnings call transcripts (CEO/CFO direct statements)
  - News articles (market-moving events)
  - FOMC meeting minutes (monetary policy)
  - Analyst reports and press releases

Your job is to answer financial research questions by:
1. RETRIEVING the most relevant document sections (use your tools)
2. SYNTHESISING a clear, factual answer from those sections
3. CITING your sources (document title, company, date)

WHEN TO USE EACH TOOL:
  query_documents      → General financial questions about any topic
  analyse_sentiment    → Understanding management tone and outlook
  extract_signals      → Getting structured bull/bear investment signals
  summarise_earnings   → Understanding what happened on earnings call
  get_risk_factors     → Understanding company-specific risks
  get_guidance         → Extracting forward guidance from filings
  get_macro_context    → Understanding Fed/macro environment
  compare_companies    → Comparing multiple companies

ALWAYS:
- Use tools to retrieve actual document content before answering
- Distinguish between facts (reported results) and forecasts (guidance)
- Note when documents are recent vs older (relevance to current situation)
- If a ticker isn't indexed, use ingest_ticker_docs to fetch its documents

NEVER:
- Answer without checking the document database first
- State facts about a company without citing a source document
- Make predictions — report what management actually said

Your answers must be traceable to real source documents."""

    def __init__(
        self,
        rag_engine=None,
        config=None,
    ):
        # Import here to avoid circular imports
        from src.rag.rag_engine import RAGEngine, get_rag_engine

        self.rag = rag_engine or get_rag_engine()

        from src.agents.base_agent import AgentConfig
        cfg = config or AgentConfig(
            name        = "DataIntelligence",
            model       = "claude-sonnet-4-6",
            temperature = 0.1,
            max_tokens  = 4096,
        )
        super().__init__(cfg)

        logger.info(
            f"DataIntelligenceAgent ready | "
            f"indexed: {self.rag.get_indexed_tickers()}"
        )

    def _get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def _get_tools(self) -> List[Tool]:
        return [
            Tool(
                name = "query_documents",
                func = self._tool_query,
                description = (
                    "Search the financial document database for relevant information. "
                    "Use this for any question about a company's financials, strategy, "
                    "news, or management statements. "
                    "Input: JSON with 'query' (str) and optional 'ticker' (str) and 'top_k' (int). "
                    "Example: {\"query\": \"Apple AI revenue growth\", \"ticker\": \"AAPL\", \"top_k\": 5}"
                ),
                param_schema={
                    "type": "object",
                    "properties": {
                        "query":  {"type": "string"},
                        "ticker": {"type": "string"},
                        "top_k":  {"type": "integer", "default": 5}
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name = "analyse_sentiment",
                func = self._tool_sentiment,
                description = (
                    "Analyse sentiment and management tone in recent documents for a ticker. "
                    "Returns structured positive/negative signals with source quotes. "
                    "Input: ticker symbol (e.g. 'AAPL')"
                ),
                param_schema={
                    "type": "object",
                    "properties": {"ticker": {"type": "string"}},
                    "required": ["ticker"]
                }
            ),
            Tool(
                name = "extract_investment_signals",
                func = self._tool_signals,
                description = (
                    "Extract structured investment signals (bullish/bearish) "
                    "from all available documents for a ticker. "
                    "Covers revenue trends, margins, guidance, risks, management tone. "
                    "Input: ticker symbol"
                ),
                param_schema={
                    "type": "object",
                    "properties": {"ticker": {"type": "string"}},
                    "required": ["ticker"]
                }
            ),
            Tool(
                name = "summarise_earnings_call",
                func = self._tool_earnings,
                description = (
                    "Summarise the most recent earnings call for a ticker. "
                    "Covers key results, guidance, management commentary, analyst Q&A. "
                    "Input: ticker symbol"
                ),
                param_schema={
                    "type": "object",
                    "properties": {"ticker": {"type": "string"}},
                    "required": ["ticker"]
                }
            ),
            Tool(
                name = "get_risk_factors",
                func = self._tool_risks,
                description = (
                    "Get key risk factors from SEC filings for a ticker. "
                    "What does the company itself say could hurt their business? "
                    "Input: ticker symbol"
                ),
                param_schema={
                    "type": "object",
                    "properties": {"ticker": {"type": "string"}},
                    "required": ["ticker"]
                }
            ),
            Tool(
                name = "get_guidance",
                func = self._tool_guidance,
                description = (
                    "Get forward guidance and outlook for a ticker. "
                    "Revenue, EPS, margins, qualitative commentary. "
                    "Input: ticker symbol"
                ),
                param_schema={
                    "type": "object",
                    "properties": {"ticker": {"type": "string"}},
                    "required": ["ticker"]
                }
            ),
            Tool(
                name = "get_macro_context",
                func = self._tool_macro,
                description = (
                    "Get macro context from FOMC minutes and economic reports. "
                    "Interest rates, inflation, economic conditions, Fed policy. "
                    "Input: question about macro conditions"
                ),
                param_schema={
                    "type": "object",
                    "properties": {"question": {"type": "string"}},
                    "required": ["question"]
                }
            ),
            Tool(
                name = "compare_companies",
                func = self._tool_compare,
                description = (
                    "Compare two or more companies on a specific dimension "
                    "using their financial documents. "
                    "Input: JSON with 'tickers' (list) and 'question' (str). "
                    "Example: {\"tickers\": [\"AAPL\", \"MSFT\"], \"question\": \"cloud growth\"}"
                ),
                param_schema={
                    "type": "object",
                    "properties": {
                        "tickers":  {"type": "array", "items": {"type": "string"}},
                        "question": {"type": "string"}
                    },
                    "required": ["tickers", "question"]
                }
            ),
            Tool(
                name = "ingest_ticker_docs",
                func = self._tool_ingest,
                description = (
                    "Fetch and index financial documents for a ticker. "
                    "Call this if a ticker isn't in the knowledge base yet. "
                    "Takes 30-60 seconds. "
                    "Input: ticker symbol"
                ),
                param_schema={
                    "type": "object",
                    "properties": {"ticker": {"type": "string"}},
                    "required": ["ticker"]
                }
            ),
            Tool(
                name = "check_indexing_status",
                func = self._tool_status,
                description = (
                    "Check which tickers are indexed in the knowledge base "
                    "and how many documents each has. "
                    "Input: anything (e.g. 'all' or specific ticker)"
                ),
                param_schema={
                    "type": "object",
                    "properties": {"ticker": {"type": "string", "default": "all"}},
                    "required": []
                }
            ),
        ]

    # ── Tool implementations ──────────────────────────────────────────────────

    def _tool_query(self, query: str, ticker: str = None, top_k: int = 5) -> str:
        ticker_filter = [ticker] if ticker else None
        result = self.rag.query(
            query_text    = query,
            ticker_filter = ticker_filter,
            top_k         = top_k,
            use_llm       = False,   # Return raw chunks (LLM handles synthesis)
        )
        if not result.retrieved:
            return f"No relevant documents found for: '{query}'"

        parts = [f"Found {result.num_sources} relevant document sections:"]
        for r in result.retrieved:
            dt  = r.chunk.published_at or r.chunk.filing_date
            dt_str = dt.strftime("%Y-%m-%d") if dt else "Date unknown"
            parts.append(
                f"\n[{r.rank}] Relevance: {r.similarity:.2f} | "
                f"{r.chunk.ticker} {r.chunk.doc_type.value} | {dt_str}\n"
                f"Title: {r.chunk.title}\n"
                f"Content: {r.chunk.text[:500]}..."
            )
        return "\n".join(parts)

    def _tool_sentiment(self, ticker: str) -> str:
        if not self.rag.is_ticker_indexed(ticker):
            return (
                f"{ticker} is not indexed. "
                f"Use ingest_ticker_docs('{ticker}') first."
            )
        result = self.rag.analyse_sentiment(ticker)
        return result.synthesis[:2000] if result.synthesis else "No sentiment data available."

    def _tool_signals(self, ticker: str) -> str:
        if not self.rag.is_ticker_indexed(ticker):
            return f"{ticker} not indexed. Use ingest_ticker_docs first."
        result = self.rag.extract_investment_signals(ticker)
        return result.synthesis[:2000] if result.synthesis else "No signals extracted."

    def _tool_earnings(self, ticker: str) -> str:
        if not self.rag.is_ticker_indexed(ticker):
            return f"{ticker} not indexed. Use ingest_ticker_docs first."
        result = self.rag.summarise_earnings_call(ticker)
        return result.synthesis[:2000] if result.synthesis else "No earnings data available."

    def _tool_risks(self, ticker: str) -> str:
        if not self.rag.is_ticker_indexed(ticker):
            return f"{ticker} not indexed. Use ingest_ticker_docs first."
        result = self.rag.get_risk_factors(ticker)
        return result.synthesis[:2000] if result.synthesis else "No risk factors found."

    def _tool_guidance(self, ticker: str) -> str:
        if not self.rag.is_ticker_indexed(ticker):
            return f"{ticker} not indexed. Use ingest_ticker_docs first."
        result = self.rag.get_guidance(ticker)
        return result.synthesis[:2000] if result.synthesis else "No guidance found."

    def _tool_macro(self, question: str) -> str:
        result = self.rag.macro_context_query(question)
        if not result.retrieved:
            return (
                "No macro documents indexed. "
                "Use ingest_ticker_docs('MACRO') to fetch FOMC minutes."
            )
        return result.synthesis[:2000] if result.synthesis else "No macro data available."

    def _tool_compare(self, tickers: List[str], question: str) -> str:
        not_indexed = [t for t in tickers if not self.rag.is_ticker_indexed(t)]
        if not_indexed:
            return (
                f"Tickers not indexed: {not_indexed}. "
                "Use ingest_ticker_docs for each."
            )
        result = self.rag.compare_tickers(tickers, question)
        return result.synthesis[:2000] if result.synthesis else "Comparison failed."

    def _tool_ingest(self, ticker: str) -> str:
        logger.info(f"Ingesting documents for {ticker}...")
        stats = self.rag.ingest_ticker(ticker)
        return (
            f"Ingestion complete for {ticker}: "
            f"{stats['documents']} documents → "
            f"{stats['chunks']} chunks indexed. "
            f"Errors: {stats.get('errors', 0)}"
        )

    def _tool_status(self, ticker: str = "all") -> str:
        stats = self.rag.store_stats()
        indexed = stats["indexed_tickers"]
        if not indexed:
            return "No tickers indexed yet. Use ingest_ticker_docs to add documents."

        lines = [
            f"Knowledge base status:",
            f"  Total chunks: {stats['total_chunks']:,}",
            f"  Embedding:    {stats['embedding_model']}",
            f"  Indexed tickers: {', '.join(indexed)}",
        ]
        if ticker != "all" and ticker in indexed:
            count = self.rag.store.count(ticker)
            lines.append(f"  {ticker}: {count} chunks")

        return "\n".join(lines)

    # ── MessageBus handler ────────────────────────────────────────────────────

    def handle_message(self, message) -> Optional[Dict[str, Any]]:
        """Process incoming research requests from the bus."""
        subject = message.subject.lower()
        payload = message.payload
        ticker  = payload.get("ticker", "")

        logger.info(
            f"DataIntelligence handling: {message.subject} from {message.sender}"
        )

        if "ingest" in subject:
            stats = self.rag.ingest_ticker(ticker)
            return {"status": "complete", "ticker": ticker, **stats}

        elif "sentiment" in subject:
            if not self.rag.is_ticker_indexed(ticker):
                self.rag.ingest_ticker(ticker)
            result = self.rag.analyse_sentiment(ticker)
            return result.to_dict()

        elif "signal" in subject or "investment" in subject:
            if not self.rag.is_ticker_indexed(ticker):
                self.rag.ingest_ticker(ticker)
            result = self.rag.extract_investment_signals(ticker)
            return result.to_dict()

        elif "earnings" in subject:
            if not self.rag.is_ticker_indexed(ticker):
                self.rag.ingest_ticker(ticker)
            result = self.rag.summarise_earnings_call(ticker)
            return result.to_dict()

        elif "risk" in subject:
            if not self.rag.is_ticker_indexed(ticker):
                self.rag.ingest_ticker(ticker)
            result = self.rag.get_risk_factors(ticker)
            return result.to_dict()

        elif "guidance" in subject:
            if not self.rag.is_ticker_indexed(ticker):
                self.rag.ingest_ticker(ticker)
            result = self.rag.get_guidance(ticker)
            return result.to_dict()

        elif "macro" in subject:
            question = payload.get("question", "Current macro environment")
            result   = self.rag.macro_context_query(question)
            return result.to_dict()

        elif "query" in subject or "document" in subject:
            query_text = payload.get("query", payload.get("question", ""))
            if not query_text:
                return {"error": "No query text provided"}

            # Auto-ingest if not indexed
            if ticker and not self.rag.is_ticker_indexed(ticker):
                self.rag.ingest_ticker(ticker)

            ticker_filter = [ticker] if ticker else payload.get("tickers")
            result = self.rag.query(
                query_text    = query_text,
                ticker_filter = ticker_filter,
                use_llm       = True,
                top_k         = payload.get("top_k", 5),
            )
            return result.to_dict()

        else:
            # General research question — let LLM decide which tools to use
            query = payload.get("query", message.subject)
            context = f"Research request from {message.sender}: {query}"
            if ticker:
                context += f"\nTicker of interest: {ticker}"

            response_text, _ = self.think(
                user_message = context,
                use_tools    = True,
                purpose      = "data_intelligence_query",
            )
            return {
                "response":  response_text,
                "timestamp": datetime.now().isoformat(),
            }

    # ── High-level convenience methods ───────────────────────────────────────

    def enrich_pm_context(self, ticker: str) -> Dict[str, str]:
        """
        Enrich Portfolio Manager context with document intelligence.

        Called by AgentCoordinator before PM makes allocation decision.
        Returns a dict of {context_type: text} to inject into PM prompt.
        """
        if not self.rag.is_ticker_indexed(ticker):
            logger.info(f"Auto-ingesting {ticker} for PM context enrichment...")
            self.rag.ingest_ticker(ticker)

        enrichment = {}

        # Recent news sentiment
        try:
            from src.rag.document_models import DocumentType, RAGQuery
            news_query = RAGQuery(
                query_text    = f"{ticker} recent news performance outlook",
                ticker_filter = [ticker],
                doc_type_filter= [DocumentType.NEWS_ARTICLE],
                top_k          = 3,
            )
            q_emb = self.rag.embedder.embed_query(news_query.query_text)
            news_chunks = self.rag.store.search(q_emb, news_query)
            if news_chunks:
                enrichment["recent_news"] = "\n".join(
                    r.chunk.text[:300] for r in news_chunks[:2]
                )
        except Exception as e:
            logger.debug(f"News enrichment failed: {e}")

        # Management guidance
        try:
            guidance_result = self.rag.get_guidance(ticker)
            if guidance_result.retrieved:
                enrichment["management_guidance"] = (
                    guidance_result.synthesis[:500]
                    if guidance_result.synthesis
                    else guidance_result.retrieved[0].chunk.text[:300]
                )
        except Exception as e:
            logger.debug(f"Guidance enrichment failed: {e}")

        return enrichment


# Import BaseAgent at bottom to avoid circular
from src.agents.base_agent import BaseAgent, Tool, AgentConfig


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  Data Intelligence Agent — Test")
    print("=" * 60)

    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  No LLM API key — testing RAG components only")

        from src.rag.rag_engine import RAGEngine
        engine = RAGEngine()

        print("\n1. Ingesting AAPL news...")
        stats = engine.ingest_ticker("AAPL", include_sec=False, include_earnings=False)
        print(f"   Stats: {stats}")

        print("\n2. Testing retrieval (no LLM)...")
        result = engine.query(
            "Apple revenue earnings",
            ticker_filter = ["AAPL"],
            use_llm       = False,
            top_k         = 3,
        )
        print(f"   Retrieved {result.num_sources} chunks")
        for r in result.retrieved[:2]:
            print(f"   [{r.rank}] {r.chunk.ticker} | sim={r.similarity:.3f}")
            print(f"         {r.chunk.text[:150]}...")

        print("\n✅ RAG components work. Add LLM key for full agent test.")
    else:
        agent = DataIntelligenceAgent()

        print("\n1. Ingesting AAPL documents...")
        stats = agent.rag.ingest_ticker("AAPL")
        print(f"   {stats}")

        print("\n2. Querying: 'What did Apple say about AI?'")
        result = agent.rag.query(
            "What has Apple said about artificial intelligence and AI products?",
            ticker_filter = ["AAPL"],
            top_k         = 4,
        )
        print(f"\n   ANSWER:\n{result.synthesis[:800]}")
        print(f"\n   Sources: {result.num_sources} | Cost: ${result.cost_usd:.5f}")
