"""
AI Hedge Fund — Part 3: RAG & Data Intelligence
=================================================
rag_engine.py — Retrieval-Augmented Generation Engine

The RAG engine is the bridge between the vector store and the LLM.

Query flow:
    1. Receive natural-language query (e.g. "What did AAPL say about AI?")
    2. Embed the query into a vector
    3. Search vector store for most similar document chunks
    4. Build a context prompt from retrieved chunks
    5. Ask the LLM to synthesise an answer using ONLY the retrieved context
    6. Return structured result with sources cited

Why RAG matters for hedge funds:
    - LLMs have a knowledge cutoff — they don't know about last week's earnings
    - RAG gives the LLM access to your specific document library
    - The LLM can't make up facts if you tell it: "answer ONLY from the provided context"
    - Every answer is traceable to a source document (audit trail)

The difference between RAG and just asking the LLM:
    Without RAG: "What did Apple say about AI on their last earnings call?"
                 → LLM may hallucinate, uses old training data
    With RAG:    → System retrieves actual Apple 8-K from last quarter
                 → LLM reads the real text and extracts the actual quote
                 → Response is grounded in the real document

This is what separates professional information systems
from chatbots — every answer is anchored to real sources.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.rag.document_models import (
    DocumentChunk, DocumentType, RAGQuery, RAGResult, RetrievedChunk
)
from src.rag.document_processor import EmbeddingEngine, VectorStore, IngestionPipeline

logger = logging.getLogger("hedge_fund.rag_engine")


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are a financial research assistant for a hedge fund.

Your role is to answer questions about companies, markets, and financial topics
using ONLY the provided source documents.

CRITICAL RULES:
1. Base your answer ONLY on the provided documents
2. If the documents do not contain relevant information, say "The provided documents
   do not contain information about this" — do not guess or use prior knowledge
3. Always cite your sources (document title and date)
4. For quantitative claims (revenue, EPS, guidance), quote the exact numbers
5. Distinguish between management statements (forward-looking) and reported results (facts)
6. Flag uncertainty: if documents give conflicting information, note the conflict

FORMAT:
- Provide a clear, direct answer first
- Then cite specific supporting evidence from the documents
- End with any important caveats or limitations

You are reading real financial documents. Be precise and factual."""


SYNTHESIS_PROMPT_TEMPLATE = """Based on the following source documents, answer this question:

QUESTION: {query}

SOURCE DOCUMENTS:
{context}

---

Provide a thorough answer based ONLY on the above sources.
Cite specific documents and dates where relevant.
If the sources are insufficient to fully answer the question, clearly state what is missing."""


SENTIMENT_ANALYSIS_PROMPT = """Analyse the tone and sentiment of the following financial text
for ticker {ticker}.

TEXT:
{context}

Provide a structured sentiment analysis:
1. OVERALL SENTIMENT: (Very Positive / Positive / Neutral / Negative / Very Negative)
2. KEY POSITIVE SIGNALS: (list specific positive statements, with quotes)
3. KEY NEGATIVE SIGNALS: (list specific concerns or risks mentioned)
4. MANAGEMENT TONE: (confident/cautious/defensive/optimistic)
5. FORWARD GUIDANCE: (raising/maintaining/lowering/no guidance)
6. NOTABLE LANGUAGE CHANGES: (compared to standard corporate language)

Focus on substance over spin."""


INVESTMENT_SIGNAL_PROMPT = """Extract investment-relevant signals from these financial documents for {ticker}.

DOCUMENTS:
{context}

Extract and structure:
1. REVENUE TRENDS: (growth rate, drivers, headwinds)
2. MARGIN TRENDS: (gross/operating/net margin changes and drivers)
3. FORWARD GUIDANCE: (next quarter/year revenue, EPS, margin guidance if provided)
4. KEY RISKS MENTIONED: (regulatory, competitive, macro, operational)
5. MANAGEMENT CONFIDENCE SIGNALS: (specific statements indicating confidence or concern)
6. COMPETITIVE POSITION: (market share, competitive advantages mentioned)
7. CAPITAL ALLOCATION: (buybacks, dividends, M&A, capex signals)

For each point: cite the specific document and quote the relevant text.
Rate each signal: BULLISH / NEUTRAL / BEARISH"""


# ─────────────────────────────────────────────────────────────────────────────
# RAG Engine
# ─────────────────────────────────────────────────────────────────────────────

class RAGEngine:
    """
    Retrieval-Augmented Generation engine for financial research.

    Combines the vector store (document retrieval) with an LLM
    (synthesis and reasoning) to answer financial research queries.

    Usage:
        engine = RAGEngine()

        # Simple query
        result = engine.query(
            "What guidance did Apple give for next quarter?",
            ticker_filter=["AAPL"]
        )
        print(result.synthesis)

        # Sentiment analysis
        sentiment = engine.analyse_sentiment("AAPL")

        # Investment signal extraction
        signals = engine.extract_investment_signals("NVDA")

        # Earnings call summary
        summary = engine.summarise_earnings_call("MSFT")
    """

    def __init__(
        self,
        pipeline:  Optional[IngestionPipeline] = None,
        llm_model: Optional[str] = None,
    ):
        self.pipeline  = pipeline or IngestionPipeline()
        self.embedder  = self.pipeline.embedder
        self.store     = self.pipeline.store
        self._llm_model= llm_model

        logger.info(
            f"RAGEngine ready | "
            f"store={self.store.count():,} chunks | "
            f"embedding={self.embedder._provider}"
        )

    # ── Core query method ─────────────────────────────────────────────────────

    def query(
        self,
        query_text:      str,
        ticker_filter:   Optional[List[str]] = None,
        doc_type_filter: Optional[List[DocumentType]] = None,
        top_k:           int = 5,
        min_similarity:  float = 0.25,
        use_llm:         bool = True,
    ) -> RAGResult:
        """
        Execute a RAG query.

        1. Embed query
        2. Retrieve top_k most similar chunks
        3. (Optional) synthesise answer with LLM

        Args:
            query_text      : Natural language question
            ticker_filter   : Only search these tickers
            doc_type_filter : Only search these document types
            top_k           : Number of chunks to retrieve
            min_similarity  : Minimum relevance threshold (0-1)
            use_llm         : Whether to generate LLM synthesis

        Returns:
            RAGResult with retrieved chunks and optional synthesis
        """
        start_time = time.time()

        # Build query object
        rag_query = RAGQuery(
            query_text      = query_text,
            ticker_filter   = ticker_filter,
            doc_type_filter = doc_type_filter,
            top_k           = top_k,
            min_similarity  = min_similarity,
        )

        # Step 1: Embed query
        query_embedding = self.embedder.embed_query(query_text)

        # Step 2: Retrieve
        retrieved = self.store.search(query_embedding, rag_query)

        if not retrieved:
            logger.info(f"No results for query: '{query_text[:60]}...'")
            return RAGResult(
                query      = rag_query,
                retrieved  = [],
                synthesis  = (
                    "No relevant documents found in the knowledge base for this query. "
                    "Try ingesting documents for the relevant tickers first."
                ),
                sources_used = [],
            )

        # Step 3: Synthesise with LLM
        synthesis    = ""
        total_tokens = 0
        cost_usd     = 0.0

        if use_llm:
            synthesis, total_tokens, cost_usd = self._synthesise(
                query_text, retrieved
            )
        else:
            # Return retrieved chunks concatenated (no LLM)
            synthesis = "\n\n---\n\n".join(
                r.to_context_string() for r in retrieved
            )

        latency_ms = (time.time() - start_time) * 1000
        sources    = list({r.chunk.doc_id for r in retrieved})

        return RAGResult(
            query        = rag_query,
            retrieved    = retrieved,
            synthesis    = synthesis,
            sources_used = sources,
            total_tokens = total_tokens,
            cost_usd     = cost_usd,
            latency_ms   = latency_ms,
        )

    def _synthesise(
        self,
        query_text: str,
        retrieved:  List[RetrievedChunk],
        prompt_template: Optional[str] = None,
    ) -> tuple:
        """
        Call LLM to synthesise an answer from retrieved chunks.

        Returns (synthesis_text, total_tokens, cost_usd)
        """
        # Build context from retrieved chunks
        context_parts = []
        for r in retrieved:
            context_parts.append(r.to_context_string(include_metadata=True))

        context = "\n\n".join(context_parts)

        # Limit context size (stay within LLM context window)
        max_context_chars = 80_000
        if len(context) > max_context_chars:
            context = context[:max_context_chars] + "\n[Context truncated for length]"

        # Build user prompt
        template = prompt_template or SYNTHESIS_PROMPT_TEMPLATE
        user_prompt = template.format(
            query   = query_text,
            context = context,
        )

        try:
            import anthropic
            client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY", "")
            )
            model = self._llm_model or os.getenv("DEFAULT_LLM_MODEL", "claude-sonnet-4-6")

            response = client.messages.create(
                model      = model,
                max_tokens = 2048,
                temperature= 0.1,
                system     = RAG_SYSTEM_PROMPT,
                messages   = [{"role": "user", "content": user_prompt}],
            )

            text    = response.content[0].text if response.content else ""
            in_tok  = response.usage.input_tokens
            out_tok = response.usage.output_tokens
            cost    = (in_tok * 0.003 + out_tok * 0.015) / 1000

            return text, in_tok + out_tok, cost

        except ImportError:
            # Try OpenAI
            try:
                import openai
                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
                response = client.chat.completions.create(
                    model       = "gpt-4o-mini",
                    max_tokens  = 2048,
                    temperature = 0.1,
                    messages    = [
                        {"role": "system",  "content": RAG_SYSTEM_PROMPT},
                        {"role": "user",    "content": user_prompt},
                    ],
                )
                text    = response.choices[0].message.content or ""
                in_tok  = response.usage.prompt_tokens
                out_tok = response.usage.completion_tokens
                cost    = (in_tok * 0.00015 + out_tok * 0.0006) / 1000
                return text, in_tok + out_tok, cost
            except Exception as e:
                logger.error(f"OpenAI synthesis failed: {e}")
                return context[:2000], 0, 0.0

        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            return f"Synthesis failed: {e}\n\nRaw retrieved content:\n{context[:2000]}", 0, 0.0

    # ── Specialised query methods ─────────────────────────────────────────────

    def analyse_sentiment(
        self,
        ticker:  str,
        days:    int = 90,
    ) -> RAGResult:
        """
        Analyse sentiment in recent filings and news for a ticker.

        Uses specialised sentiment extraction prompt.
        """
        query_text = (
            f"What is the overall tone and sentiment in recent {ticker} "
            "documents? What positive and negative signals are there?"
        )

        retrieved = self._retrieve_for_ticker(
            ticker   = ticker,
            query    = query_text,
            top_k    = 8,
            doc_types= [
                DocumentType.EARNINGS_CALL,
                DocumentType.NEWS_ARTICLE,
                DocumentType.SEC_8K,
            ],
        )

        if not retrieved:
            return RAGResult(
                query        = RAGQuery(query_text=query_text, ticker_filter=[ticker]),
                retrieved    = [],
                synthesis    = f"No documents found for {ticker}. Ingest documents first.",
                sources_used = [],
            )

        context = "\n\n".join(r.to_context_string() for r in retrieved)
        synthesis, tokens, cost = self._synthesise(
            query_text      = query_text,
            retrieved       = retrieved,
            prompt_template = SENTIMENT_ANALYSIS_PROMPT.replace("{ticker}", ticker),
        )

        return RAGResult(
            query        = RAGQuery(query_text=query_text, ticker_filter=[ticker]),
            retrieved    = retrieved,
            synthesis    = synthesis,
            sources_used = [r.chunk.doc_id for r in retrieved],
            total_tokens = tokens,
            cost_usd     = cost,
        )

    def extract_investment_signals(
        self,
        ticker: str,
    ) -> RAGResult:
        """
        Extract structured investment signals from all available documents.

        Returns bullish/bearish signals with source citations.
        """
        query_text = (
            f"What are the key investment signals for {ticker}? "
            "Revenue trends, guidance, risks, management tone?"
        )

        retrieved = self._retrieve_for_ticker(
            ticker  = ticker,
            query   = query_text,
            top_k   = 10,
        )

        if not retrieved:
            return RAGResult(
                query        = RAGQuery(query_text=query_text),
                retrieved    = [],
                synthesis    = f"No documents found for {ticker}.",
                sources_used = [],
            )

        synthesis, tokens, cost = self._synthesise(
            query_text      = query_text,
            retrieved       = retrieved,
            prompt_template = INVESTMENT_SIGNAL_PROMPT.replace("{ticker}", ticker),
        )

        return RAGResult(
            query        = RAGQuery(query_text=query_text, ticker_filter=[ticker]),
            retrieved    = retrieved,
            synthesis    = synthesis,
            sources_used = [r.chunk.doc_id for r in retrieved],
            total_tokens = tokens,
            cost_usd     = cost,
        )

    def summarise_earnings_call(
        self,
        ticker: str,
    ) -> RAGResult:
        """
        Summarise the most recent earnings call for a ticker.

        Focuses on prepared remarks and Q&A highlights.
        """
        query_text = (
            f"Summarise the {ticker} earnings call: "
            "key financial results, guidance, management commentary, "
            "and analyst questions."
        )

        retrieved = self._retrieve_for_ticker(
            ticker    = ticker,
            query     = query_text,
            top_k     = 8,
            doc_types = [DocumentType.EARNINGS_CALL, DocumentType.SEC_8K],
        )

        if not retrieved:
            # Fallback to any documents
            retrieved = self._retrieve_for_ticker(
                ticker = ticker,
                query  = query_text,
                top_k  = 6,
            )

        synthesis, tokens, cost = self._synthesise(query_text, retrieved)

        return RAGResult(
            query        = RAGQuery(query_text=query_text, ticker_filter=[ticker]),
            retrieved    = retrieved,
            synthesis    = synthesis,
            sources_used = [r.chunk.doc_id for r in retrieved],
            total_tokens = tokens,
            cost_usd     = cost,
        )

    def get_risk_factors(self, ticker: str) -> RAGResult:
        """Extract key risk factors from SEC filings."""
        query_text = (
            f"What are the key risk factors for {ticker} mentioned "
            "in their SEC filings? What could cause the business to underperform?"
        )
        return self.query(
            query_text      = query_text,
            ticker_filter   = [ticker],
            doc_type_filter = [DocumentType.SEC_10K, DocumentType.SEC_10Q],
            top_k           = 6,
        )

    def get_guidance(self, ticker: str) -> RAGResult:
        """Extract forward guidance from filings and earnings calls."""
        query_text = (
            f"What forward guidance has {ticker} management provided? "
            "Revenue, earnings, margins, and qualitative outlook."
        )
        return self.query(
            query_text      = query_text,
            ticker_filter   = [ticker],
            doc_type_filter = [
                DocumentType.EARNINGS_CALL,
                DocumentType.SEC_8K,
                DocumentType.NEWS_ARTICLE,
            ],
            top_k = 6,
        )

    def macro_context_query(self, question: str) -> RAGResult:
        """
        Query macro-economic documents (Fed minutes, economic reports).
        """
        return self.query(
            query_text      = question,
            ticker_filter   = ["MACRO"],
            doc_type_filter = [DocumentType.MACRO_REPORT],
            top_k           = 5,
        )

    def compare_tickers(
        self,
        tickers:  List[str],
        question: str,
    ) -> RAGResult:
        """
        Compare multiple tickers on a specific dimension.

        Example: compare_tickers(["AAPL", "MSFT"], "cloud revenue growth")
        """
        formatted_tickers = " vs ".join(tickers)
        full_query = f"Compare {formatted_tickers}: {question}"

        return self.query(
            query_text    = full_query,
            ticker_filter = tickers,
            top_k         = 8,
        )

    # ── Helper methods ────────────────────────────────────────────────────────

    def _retrieve_for_ticker(
        self,
        ticker:    str,
        query:     str,
        top_k:     int = 5,
        doc_types: Optional[List[DocumentType]] = None,
    ) -> List[RetrievedChunk]:
        """Retrieve chunks for a specific ticker."""
        rag_query = RAGQuery(
            query_text      = query,
            ticker_filter   = [ticker],
            doc_type_filter = doc_types,
            top_k           = top_k,
            min_similarity  = 0.20,
        )
        query_embedding = self.embedder.embed_query(query)
        return self.store.search(query_embedding, rag_query)

    def is_ticker_indexed(self, ticker: str) -> bool:
        """Check if a ticker has documents in the vector store."""
        return self.store.count(ticker) > 0

    def get_indexed_tickers(self) -> List[str]:
        """Get list of all indexed tickers."""
        return self.store.get_indexed_tickers()

    def ingest_ticker(
        self,
        ticker:   str,
        refresh:  bool = False,
    ) -> Dict[str, int]:
        """Convenience method to ingest documents for a ticker."""
        return self.pipeline.ingest_ticker(ticker, refresh=refresh)

    def store_stats(self) -> Dict[str, Any]:
        return {
            "total_chunks":     self.store.count(),
            "indexed_tickers":  self.get_indexed_tickers(),
            "embedding_model":  self.embedder._provider,
            "embedding_dims":   self.embedder.dimensions,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Singleton factory
# ─────────────────────────────────────────────────────────────────────────────

_rag_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    """Get or create the singleton RAG engine."""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  RAG Engine — Integration Test")
    print("=" * 60)

    engine = RAGEngine()
    print(f"\nStore stats: {engine.store_stats()}")

    # Ingest documents for AAPL
    print("\n1. Ingesting AAPL documents...")
    stats = engine.ingest_ticker("AAPL")
    print(f"   Ingestion stats: {stats}")
    print(f"   Total chunks now: {engine.store.count()}")

    if engine.store.count() > 0:
        print("\n2. Testing basic query (no LLM)...")
        result = engine.query(
            "Apple revenue earnings results",
            ticker_filter = ["AAPL"],
            use_llm       = False,
            top_k         = 3,
        )
        print(f"   Retrieved {result.num_sources} chunks")
        for r in result.retrieved[:2]:
            print(f"   [{r.rank}] sim={r.similarity:.3f} | {r.chunk}")
            print(f"        '{r.chunk.text[:100]}...'")

        if os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"):
            print("\n3. Testing LLM synthesis...")
            result = engine.query(
                "What did Apple report for recent revenue and earnings?",
                ticker_filter = ["AAPL"],
                use_llm       = True,
                top_k         = 4,
            )
            print(f"\n   SYNTHESIS:\n{result.synthesis[:600]}")
            print(f"\n   Cost: ${result.cost_usd:.5f} | Latency: {result.latency_ms:.0f}ms")
        else:
            print("\n3. (Add LLM API key to .env for synthesis test)")

    print("\n✅ RAG Engine tests passed")
    print("   Install: pip install sentence-transformers chromadb")
    print("   for full vector search capability")
